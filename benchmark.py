import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import evaluate

# --- Configuration ---
# Replace with your Hugging Face dataset name
DATASET_NAME = "prepared_data"
DATSET_SPLIT = "test"
# Replace with the desired Whisper model name
MODEL_NAME = "whisper-large-v3-turbo-lora-pl-med-asr-45s"
# The name of the audio column in your dataset
AUDIO_COLUMN_NAME = "path"
# The name of the text column in your dataset
TEXT_COLUMN_NAME = "text"
# The language of the audio data (if known, otherwise it will be detected)
LANGUAGE = "Polish"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Batch size for processing
BATCH_SIZE = 32
# Results directory
RESULTS_DIR = "benchmark_results"
# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

def create_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")

def process_batch(model, processor, audio_batch: List[Dict], device: str) -> List[str]:
    """
    Process a batch of audio samples and return transcriptions.

    Args:
        model: The Whisper model
        processor: The Whisper processor
        audio_batch: List of audio dictionaries
        device: Device to run inference on

    Returns:
        List of transcriptions
    """
    # Extract audio arrays and sampling rates
    audio_arrays = [item["array"] for item in audio_batch]
    sampling_rates = [item["sampling_rate"] for item in audio_batch]

    # Ensure all samples have the same sampling rate
    assert all(sr == sampling_rates[0] for sr in sampling_rates), "All audio samples must have the same sampling rate"

    # Process all audio samples in the batch
    input_features = processor(
        audio_arrays,
        sampling_rate=sampling_rates[0],
        return_tensors="pt"
    ).input_features

    input_features = input_features.to(device)

    # Generate transcriptions for the batch
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=448)

    # Decode all transcriptions
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcriptions

def save_detailed_results(results: Dict[str, Any], model_name: str) -> str:
    """
    Save detailed results to a JSON file.

    Args:
        results: Dictionary containing all results
        model_name: Name of the model

    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    filename = f"{safe_model_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {filepath}")
    return filepath

def update_leaderboard(results: Dict[str, Any]) -> None:
    """
    Update the leaderboard with new results.

    Args:
        results: Dictionary containing benchmark results
    """
    leaderboard_path = os.path.join(RESULTS_DIR, LEADERBOARD_FILE)

    # Load existing leaderboard or create new one
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'r', encoding='utf-8') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Create leaderboard entry
    entry = {
        "model_name": results["model_name"],
        "dataset_name": results["dataset_name"],
        "wer": results["wer"],
        "cer": results["cer"],
        "timestamp": results["timestamp"],
        "num_samples": results["num_samples"],
        "batch_size": results["batch_size"],
        "device": results["device"],
        "language": results["language"],
        "detailed_results_file": results["detailed_results_file"]
    }

    # Add to leaderboard
    leaderboard.append(entry)

    # Sort by WER (lower is better)
    leaderboard.sort(key=lambda x: x["wer"])

    # Save updated leaderboard
    with open(leaderboard_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Leaderboard updated: {leaderboard_path}")

def display_leaderboard(top_n: int = 10) -> None:
    """
    Display the current leaderboard.

    Args:
        top_n: Number of top results to display
    """
    leaderboard_path = os.path.join(RESULTS_DIR, LEADERBOARD_FILE)

    if not os.path.exists(leaderboard_path):
        print("No leaderboard found. Run some benchmarks first!")
        return

    with open(leaderboard_path, 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)

    if not leaderboard:
        print("Leaderboard is empty.")
        return

    print(f"\n{'='*80}")
    print(f"{'ASR MODEL LEADERBOARD':^80}")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Model':<30} {'WER':<8} {'CER':<8} {'Samples':<8} {'Date':<12}")
    print(f"{'-'*80}")

    for i, entry in enumerate(leaderboard[:top_n], 1):
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d")
        model_short = entry["model_name"][:28] + "..." if len(entry["model_name"]) > 30 else entry["model_name"]
        print(f"{i:<5} {model_short:<30} {entry['wer']:<8.4f} {entry['cer']:<8.4f} {entry['num_samples']:<8} {timestamp:<12}")

    print(f"{'='*80}\n")

def main():
    """
    Main function to run the ASR benchmark with batch processing and result saving.
    """
    # Create results directory
    create_results_directory()

    # --- 1. Load Dataset ---
    print("Loading dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split="test", trust_remote_code=True)
        print(f"Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset '{DATASET_NAME}'. Error: {e}")
        return

    # Ensure the audio column is in the correct format (16kHz sampling rate)
    dataset = dataset.cast_column(AUDIO_COLUMN_NAME, Audio(sampling_rate=16000))

    # --- 2. Load Model and Processor ---
    print(f"Loading model and processor: {MODEL_NAME}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, task="transcribe")
        model.to(DEVICE)
        print(f"Model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"Failed to load model '{MODEL_NAME}'. Error: {e}")
        return

    # --- 3. Run Batch Inference ---
    predictions = []
    references = []

    print(f"\nStarting batch inference (batch_size={BATCH_SIZE})...")

    # Process in batches with progress bar
    num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(dataset), desc="Processing samples") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))

            # Get batch data
            batch_items = [dataset[i] for i in range(start_idx, end_idx)]
            audio_batch = [item[AUDIO_COLUMN_NAME] for item in batch_items]
            batch_references = [item[TEXT_COLUMN_NAME] for item in batch_items]

            try:
                # Process batch
                batch_predictions = process_batch(model, processor, audio_batch, DEVICE)

                # Collect results
                predictions.extend(batch_predictions)
                references.extend(batch_references)

                pbar.update(len(batch_items))

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # Add empty predictions for failed batch to maintain alignment
                predictions.extend([""] * len(batch_items))
                references.extend(batch_references)
                pbar.update(len(batch_items))

    print("Batch inference complete.")

    # --- 4. Calculate Metrics ---
    print("Calculating metrics...")

    # Filter out empty predictions for metric calculation
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
    valid_predictions = [p for p, r in valid_pairs]
    valid_references = [r for p, r in valid_pairs]

    # Calculate WER (Word Error Rate) using evaluate library
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(references=valid_references, predictions=valid_predictions)

    # Calculate CER (Character Error Rate) using evaluate library
    cer_metric = evaluate.load("cer")
    cer_score = cer_metric.compute(references=valid_references, predictions=valid_predictions)

    # --- 5. Prepare Results ---
    timestamp = datetime.now().isoformat()

    results = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "language": LANGUAGE,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "num_samples": len(dataset),
        "num_valid_predictions": len(valid_predictions),
        "wer": wer_score,
        "cer": cer_score,
        "timestamp": timestamp,
        "predictions": predictions,
        "references": references,
        "examples": [
            {
                "reference": references[i],
                "prediction": predictions[i],
                "sample_idx": i
            }
            for i in range(min(10, len(predictions)))
        ]
    }

    # --- 6. Save Results ---
    detailed_results_file = save_detailed_results(results, MODEL_NAME)
    results["detailed_results_file"] = os.path.basename(detailed_results_file)

    # Update leaderboard
    update_leaderboard(results)

    # --- 7. Display Results ---
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Total Samples: {len(dataset)}")
    print(f"Valid Predictions: {len(valid_predictions)}")
    print(f"Word Error Rate (WER): {wer_score:.4f}")
    print(f"Character Error Rate (CER): {cer_score:.4f}")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Show some examples
    print("\nExample Transcriptions:")
    print("-" * 40)
    for i, example in enumerate(results["examples"][:5]):
        print(f"Example {i+1}:")
        print(f"Reference:  {example['reference']}")
        print(f"Prediction: {example['prediction']}")
        print()

    # Display current leaderboard
    display_leaderboard()

if __name__ == "__main__":
    main()