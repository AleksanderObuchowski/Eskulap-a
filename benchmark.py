"""Multi-test benchmark for ASR models.

Evaluates models on multiple test sets separately, computing per-test
and aggregate metrics for comprehensive performance analysis.

Usage:
    # Use default model (Qwen3-ASR with LoRA)
    python benchmark.py --config configs/default.yaml

    # Use base model (no LoRA)
    python benchmark.py --model Qwen/Qwen3-ASR-1.7B

    # Use a LoRA adapter (auto-detected)
    python benchmark.py --model models/my-lora-checkpoint

    # Use a fine-tuned model from HuggingFace
    python benchmark.py --model username/my-finetuned-model

    # Use Whisper
    python benchmark.py --model-type whisper --model openai/whisper-large-v3-turbo
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from eval.llm_rescore import (
    build_hypotheses,
    deduplicate_by_text,
    get_whisper_nbest,
    load_bielik,
    score_with_causal_lm,
)

# --- Configuration ---
LANGUAGE = "Polish"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
RESULTS_DIR = "benchmark_results"

# Default model configuration
DEFAULT_MODEL_TYPE = "qwen3"

# Whisper models
WHISPER_MODEL_NAME = "AleksanderObuchowski/whisper-large-v3-turbo-med-pl-lora"
WHISPER_BASE_MODEL = "openai/whisper-large-v3-turbo"

# Qwen3-ASR models
QWEN3_BASE_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN3_LORA_PATH = "models/Qwen3-ASR-1.7B-med-pl-lora-decoder-only/checkpoint-7350"


def is_lora_model(model_path: str) -> bool:
    """Detect if a model path contains a LoRA adapter.

    Checks for adapter_config.json which is the standard marker for PEFT/LoRA adapters.
    Works with both local paths and HuggingFace model IDs.

    Args:
        model_path: Local path or HuggingFace model ID

    Returns:
        True if the path contains a LoRA adapter, False otherwise
    """
    # Check local path first
    local_path = Path(model_path)
    if local_path.exists():
        return (local_path / "adapter_config.json").exists()

    # Check HuggingFace Hub
    try:
        hf_hub_download(repo_id=model_path, filename="adapter_config.json")
        return True
    except Exception:
        return False


def resolve_model_config(
    model_path: str | None, model_type: str
) -> tuple[str, str | None]:
    """Resolve model path into base model and optional LoRA path.

    Automatically detects whether the provided path is a base model,
    fine-tuned model, or LoRA adapter.

    Args:
        model_path: Path to model (local or HuggingFace). None uses defaults.
        model_type: "whisper" or "qwen3"

    Returns:
        Tuple of (base_model_path, lora_path or None)
    """
    if model_type == "qwen3":
        if model_path is None:
            # Use default: base + LoRA
            return QWEN3_BASE_MODEL, QWEN3_LORA_PATH

        if is_lora_model(model_path):
            # It's a LoRA adapter - use with default base
            return QWEN3_BASE_MODEL, model_path
        else:
            # It's a base or fine-tuned model - use directly, no LoRA
            return model_path, None
    else:
        # Whisper
        if model_path is None:
            return WHISPER_MODEL_NAME, None

        if is_lora_model(model_path):
            # Whisper LoRA - use with base model
            return WHISPER_BASE_MODEL, model_path
        else:
            # Base or fine-tuned Whisper
            return model_path, None


@dataclass
class TestSetResult:
    """Results for a single test set."""

    name: str
    wer: float
    cer: float
    num_samples: int
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results across all test sets."""

    model_name: str
    timestamp: str
    test_sets: list[TestSetResult]
    avg_wer_unweighted: float
    avg_cer_unweighted: float
    avg_wer_weighted: float
    avg_cer_weighted: float
    total_samples: int


def create_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def load_qwen3_model(base_model: str, lora_path: str | None = None):
    """Load Qwen3-ASR model with optional LoRA adapter.

    Args:
        base_model: Base model path (local or HuggingFace)
        lora_path: Optional LoRA adapter path (local or HuggingFace)

    Returns:
        Tuple of (model, processor)
    """
    os.environ["AUDIO_DECODER_BACKEND"] = "soundfile"
    import warnings

    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    print(f"Loading Qwen3-ASR model: {base_model}")
    wrapper = Qwen3ASRModel.from_pretrained(base_model, device_map=None)
    processor = wrapper.processor

    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*missing adapter keys.*")
            model = PeftModel.from_pretrained(wrapper.model, lora_path)
        model = model.merge_and_unload()
        model = model.thinker
    else:
        print("Using model without LoRA adapter")
        model = wrapper.model.thinker

    model.to(DEVICE)
    model.eval()
    return model, processor


def transcribe_qwen3_single(model, processor, audio_array: dict) -> str:
    """Transcribe a single audio sample with Qwen3-ASR."""
    arr = audio_array["array"]
    tokenizer = processor.tokenizer

    messages = [{"role": "user", "content": [{"type": "audio", "audio": arr}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt += f"language {LANGUAGE}<asr_text>"

    inputs = processor(text=prompt, audio=arr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def process_batch_whisper(
    model, processor, audio_batch: list[dict], device: str
) -> list[str]:
    """Process a batch of audio samples with Whisper."""
    audio_arrays = [item["array"] for item in audio_batch]
    sampling_rates = [item["sampling_rate"] for item in audio_batch]

    assert all(sr == sampling_rates[0] for sr in sampling_rates)

    input_features = processor(
        audio_arrays, sampling_rate=sampling_rates[0], return_tensors="pt"
    ).input_features
    input_features = input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=448)

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


def _whisper_nbest_simple(
    model,
    processor,
    audio: np.ndarray,
    num_beams: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Generate Whisper n-best using sequences_scores (avoids compute_transition_scores CUDA bugs)."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="polish", task="transcribe"
    )

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=444,
            return_dict_in_generate=True,
            output_scores=True,
            forced_decoder_ids=forced_decoder_ids,
        )

    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

    # Use sequences_scores (length-normalized log-probs from beam search)
    asr_avg = outputs.sequences_scores.cpu().float().numpy().astype(np.float32)

    # Estimate token counts from non-pad tokens
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        asr_count = outputs.sequences.ne(pad_id).sum(dim=1).cpu().numpy().astype(np.int64)
    else:
        asr_count = np.full(len(texts), outputs.sequences.shape[1], dtype=np.int64)
    asr_count = np.maximum(1, asr_count)
    asr_sum = (asr_avg * asr_count.astype(np.float32)).astype(np.float32)

    return texts, asr_sum, asr_avg, asr_count


def transcribe_whisper_with_rescore(
    model,
    processor,
    audio_array: dict,
    lm_model,
    lm_tokenizer,
    rescore_config: dict,
) -> str:
    """Transcribe a single audio sample using Whisper n-best + LM rescoring.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_array: Dict with 'array' and 'sampling_rate' keys
        lm_model: Loaded causal LM for rescoring
        lm_tokenizer: Tokenizer for the LM
        rescore_config: Dict with 'lm_weight', 'num_beams', 'lm_batch_size'

    Returns:
        Best hypothesis text after rescoring
    """
    audio = audio_array["array"].astype(np.float32)
    num_beams = rescore_config["num_beams"]

    try:
        texts, asr_sum, asr_avg, asr_count = _whisper_nbest_simple(
            model, processor, audio, num_beams
        )

        texts, asr_sum, asr_avg, asr_count = deduplicate_by_text(
            texts, asr_sum, asr_avg, asr_count
        )

        lm_sum, lm_avg, lm_count = score_with_causal_lm(
            texts=texts,
            tokenizer=lm_tokenizer,
            model=lm_model,
            device=DEVICE,
            batch_size=rescore_config["lm_batch_size"],
        )

        hypotheses = build_hypotheses(
            texts=texts,
            asr_sum=asr_sum,
            asr_avg=asr_avg,
            asr_count=asr_count,
            lm_sum=lm_sum,
            lm_avg=lm_avg,
            lm_count=lm_count,
            lm_weight=rescore_config["lm_weight"],
            word_bonus=0.0,
        )

        best = max(hypotheses, key=lambda h: h.combined_score)
        return best.text
    except Exception as e:
        # Fallback: greedy 1-best
        print(f"  Rescoring failed, falling back to greedy: {e}")
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=448)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()


def load_test_set(
    dataset_name: str,
    max_samples: int | None = None,
    cache_dir: str | None = None,  # Kept for backward compatibility, ignored
    test_sets_dir: str | None = None,  # Kept for backward compatibility, ignored
) -> Dataset | None:
    """Load a test set from pre-processed clean splits.

    Args:
        dataset_name: Name of the dataset (e.g., "admed_anoni", "admed_human")
        max_samples: Maximum number of test samples to use (None = all)
        cache_dir: Deprecated, ignored
        test_sets_dir: Deprecated, ignored

    Returns:
        Dataset containing test samples, or None if not found
    """
    from data import load_single_dataset

    try:
        test_dataset = load_single_dataset(
            dataset_name, split="test", max_samples=max_samples
        )
        return test_dataset
    except Exception as e:
        print(f"  Warning: Could not load test set for {dataset_name}: {e}")
        return None


def evaluate_test_set(
    model,
    processor,
    dataset: Dataset,
    dataset_name: str,
    model_type: str,
    rescore_config: dict | None = None,
    lm_model=None,
    lm_tokenizer=None,
) -> TestSetResult:
    """Evaluate model on a single test set.

    Args:
        model: Loaded model
        processor: Model processor
        dataset: Test dataset
        dataset_name: Name for display
        model_type: "whisper" or "qwen3"
        rescore_config: Optional dict with rescoring params (lm_weight, num_beams, lm_batch_size)
        lm_model: Loaded causal LM (required when rescore_config is set)
        lm_tokenizer: LM tokenizer (required when rescore_config is set)

    Returns:
        TestSetResult with metrics and predictions
    """
    predictions = []
    references = []

    use_rescore = rescore_config is not None and model_type == "whisper"

    if model_type == "qwen3":
        for i in tqdm(range(len(dataset)), desc=f"  {dataset_name}", leave=False):
            item = dataset[i]
            references.append(item["text"])
            try:
                pred = transcribe_qwen3_single(model, processor, item["audio"])
                predictions.append(pred)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                predictions.append("")
    elif use_rescore:
        for i in tqdm(range(len(dataset)), desc=f"  {dataset_name} (rescore)", leave=False):
            item = dataset[i]
            references.append(item["text"])
            try:
                pred = transcribe_whisper_with_rescore(
                    model, processor, item["audio"],
                    lm_model, lm_tokenizer, rescore_config,
                )
                predictions.append(pred.strip())
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                predictions.append("")
    else:
        num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in tqdm(
            range(num_batches), desc=f"  {dataset_name}", leave=False
        ):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))

            batch_items = [dataset[i] for i in range(start_idx, end_idx)]
            audio_batch = [item["audio"] for item in batch_items]
            batch_references = [item["text"] for item in batch_items]

            try:
                batch_predictions = process_batch_whisper(
                    model, processor, audio_batch, DEVICE
                )
                predictions.extend([p.strip() for p in batch_predictions])
                references.extend(batch_references)
            except Exception as e:
                print(f"Error on batch {batch_idx}: {e}")
                predictions.extend([""] * len(batch_items))
                references.extend(batch_references)

    # Normalize predictions and references for consistent evaluation
    from data.text_normalize import normalize_text_for_asr

    predictions = [normalize_text_for_asr(p) for p in predictions]
    references = [normalize_text_for_asr(r) for r in references]

    # Filter empty predictions for metrics
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
    valid_predictions = [p for p, _ in valid_pairs]
    valid_references = [r for _, r in valid_pairs]

    # Calculate metrics
    if not valid_predictions:
        print(f"  WARNING: No valid predictions for {dataset_name} — all samples failed")
        wer = 1.0
        cer = 1.0
    else:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        wer = wer_metric.compute(references=valid_references, predictions=valid_predictions)
        cer = cer_metric.compute(references=valid_references, predictions=valid_predictions)

    return TestSetResult(
        name=dataset_name,
        wer=wer,
        cer=cer,
        num_samples=len(dataset),
        predictions=predictions,
        references=references,
    )


class MultiTestBenchmark:
    """Benchmark runner that evaluates on multiple test sets."""

    def __init__(
        self,
        model_type: str = "qwen3",
        model_path: str | None = None,
        cache_dir: str | None = None,  # Deprecated, ignored
        test_sets_dir: str | None = None,  # Deprecated, ignored
        rescore_config: dict | None = None,
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.rescore_config = rescore_config
        self.lm_model = None
        self.lm_tokenizer = None

    def load_model(self):
        """Load the model based on configuration.

        Automatically detects whether model_path is a base model,
        fine-tuned model, or LoRA adapter.
        """
        base_model, lora_path = resolve_model_config(self.model_path, self.model_type)

        if self.model_type == "qwen3":
            self.model, self.processor = load_qwen3_model(base_model, lora_path)
            self.model_name = base_model
            if lora_path:
                lora_name = (
                    Path(lora_path).name if Path(lora_path).exists() else lora_path
                )
                self.model_name += f"+{lora_name}"
        else:
            if lora_path:
                # Whisper with LoRA
                from peft import PeftModel

                base = WhisperForConditionalGeneration.from_pretrained(base_model)
                self.model = PeftModel.from_pretrained(base, lora_path)
                self.model = self.model.merge_and_unload()
                self.model_name = f"{base_model}+{Path(lora_path).name}"
            else:
                # Base or fine-tuned Whisper
                self.model = WhisperForConditionalGeneration.from_pretrained(base_model)
                self.model_name = base_model

            self.processor = WhisperProcessor.from_pretrained(
                WHISPER_BASE_MODEL, language="Polish", task="transcribe"
            )
            self.model.to(DEVICE)

    def load_lm(self):
        """Load the LM for rescoring."""
        lm_model_id = self.rescore_config["lm_model"]
        print(f"Loading LM for rescoring: {lm_model_id}")
        self.lm_tokenizer, self.lm_model = load_bielik(lm_model_id, DEVICE)

    def run(self, test_datasets: list[str] | list[dict[str, any]]) -> BenchmarkResults:
        """Run benchmark on specified test sets.

        Args:
            test_datasets: List of dataset names (strings) or dicts with
                          'name' and optional 'max_samples' keys

        Returns:
            BenchmarkResults with per-test and aggregate metrics
        """
        if self.model is None:
            self.load_model()

        if self.rescore_config is not None and self.lm_model is None:
            self.load_lm()

        # Normalize to list of dicts
        normalized = []
        for item in test_datasets:
            if isinstance(item, str):
                normalized.append({"name": item, "max_samples": None})
            else:
                normalized.append(item)

        print("=" * 70)
        print("MULTI-TEST BENCHMARK")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Model type: {self.model_type}")
        if self.rescore_config:
            print(f"LM rescoring: {self.rescore_config['lm_model']} (weight={self.rescore_config['lm_weight']}, beams={self.rescore_config['num_beams']})")
        print(f"Test sets: {', '.join(d['name'] for d in normalized)}")
        print()

        test_results = []

        for ds in normalized:
            name = ds["name"]
            max_samples = ds.get("max_samples")

            print(f"\nEvaluating: {name}")
            if max_samples:
                print(f"  (limited to {max_samples} samples)")
            print("-" * 40)

            dataset = load_test_set(name, max_samples)
            if dataset is None:
                print(f"  Skipping {name} - no data found")
                continue

            print(f"  Loaded {len(dataset)} test samples")

            result = evaluate_test_set(
                self.model,
                self.processor,
                dataset,
                name,
                self.model_type,
                rescore_config=self.rescore_config,
                lm_model=self.lm_model,
                lm_tokenizer=self.lm_tokenizer,
            )
            test_results.append(result)
            print(f"  WER: {result.wer:.4f}, CER: {result.cer:.4f}")

        # Calculate aggregate metrics
        if test_results:
            # Unweighted average (each test set counts equally)
            avg_wer_unweighted = sum(r.wer for r in test_results) / len(test_results)
            avg_cer_unweighted = sum(r.cer for r in test_results) / len(test_results)

            # Weighted average (by sample count)
            total_samples = sum(r.num_samples for r in test_results)
            avg_wer_weighted = (
                sum(r.wer * r.num_samples for r in test_results) / total_samples
            )
            avg_cer_weighted = (
                sum(r.cer * r.num_samples for r in test_results) / total_samples
            )
        else:
            avg_wer_unweighted = avg_cer_unweighted = 0.0
            avg_wer_weighted = avg_cer_weighted = 0.0
            total_samples = 0

        results = BenchmarkResults(
            model_name=self.model_name,
            timestamp=datetime.now().isoformat(),
            test_sets=test_results,
            avg_wer_unweighted=avg_wer_unweighted,
            avg_cer_unweighted=avg_cer_unweighted,
            avg_wer_weighted=avg_wer_weighted,
            avg_cer_weighted=avg_cer_weighted,
            total_samples=total_samples,
        )

        return results

    def display_results(self, results: BenchmarkResults):
        """Display results in a formatted table."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Model: {results.model_name}")
        print(f"Timestamp: {results.timestamp}")
        print()

        # Results table
        print(f"{'Test Set':<25} {'WER':<12} {'CER':<12} {'Samples':<10}")
        print("-" * 70)

        for ts in results.test_sets:
            print(f"{ts.name:<25} {ts.wer:<12.4f} {ts.cer:<12.4f} {ts.num_samples:<10}")

        print("-" * 70)
        print(
            f"{'Average (unweighted)':<25} {results.avg_wer_unweighted:<12.4f} "
            f"{results.avg_cer_unweighted:<12.4f}"
        )
        print(
            f"{'Average (weighted)':<25} {results.avg_wer_weighted:<12.4f} "
            f"{results.avg_cer_weighted:<12.4f} {results.total_samples:<10}"
        )
        print("=" * 70)

    def save_results(self, results: BenchmarkResults) -> str:
        """Save detailed results to JSON file.

        Returns:
            Path to saved file
        """
        create_results_directory()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = results.model_name.replace("/", "_")
        filename = f"multi_test_{safe_model_name}_{timestamp}.json"
        filepath = os.path.join(RESULTS_DIR, filename)

        # Convert to serializable format
        data = {
            "model_name": results.model_name,
            "timestamp": results.timestamp,
            "avg_wer_unweighted": results.avg_wer_unweighted,
            "avg_cer_unweighted": results.avg_cer_unweighted,
            "avg_wer_weighted": results.avg_wer_weighted,
            "avg_cer_weighted": results.avg_cer_weighted,
            "total_samples": results.total_samples,
            "test_sets": [
                {
                    "name": ts.name,
                    "wer": ts.wer,
                    "cer": ts.cer,
                    "num_samples": ts.num_samples,
                    "predictions": ts.predictions,
                    "references": ts.references,
                }
                for ts in results.test_sets
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main entry point for benchmark."""
    parser = argparse.ArgumentParser(description="Multi-test ASR benchmark")
    parser.add_argument(
        "--config",
        help="Path to training config YAML (uses test_datasets from config)",
    )
    parser.add_argument(
        "--test-sets",
        nargs="+",
        help="Test set names to evaluate (overrides config)",
    )
    parser.add_argument(
        "--model-type",
        choices=["whisper", "qwen3"],
        default=DEFAULT_MODEL_TYPE,
        help="Model type to use",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        help="Model path (HuggingFace ID or local). Auto-detects base/finetuned/LoRA.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detailed results to file",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Enable LM rescoring (n-best + causal LM)",
    )
    parser.add_argument(
        "--lm-model",
        default="speakleash/Bielik-1.5B-v3",
        help="Causal LM for rescoring (default: speakleash/Bielik-1.5B-v3)",
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=0.35,
        help="LM interpolation weight for rescoring (default: 0.35)",
    )
    parser.add_argument(
        "--num-beams-rescore",
        type=int,
        default=8,
        help="Number of beams for n-best generation in rescoring (default: 8)",
    )
    parser.add_argument(
        "--lm-batch-size",
        type=int,
        default=8,
        help="Batch size for LM scoring (default: 8)",
    )

    args = parser.parse_args()

    # Determine test sets to evaluate
    if args.test_sets:
        # CLI args are just names (no max_samples)
        test_datasets = args.test_sets
    elif args.config:
        from data import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.config)
        # Get enabled test datasets from config
        test_datasets = [
            {"name": name, "max_samples": ds_config.max_samples}
            for name, ds_config in config.test_datasets.items()
            if ds_config.enabled
        ]
    else:
        # Default test sets
        test_datasets = ["admed_anoni", "admed_human"]

    # Build rescore config if enabled
    rescore_config = None
    if args.rescore:
        rescore_config = {
            "lm_model": args.lm_model,
            "lm_weight": args.lm_weight,
            "num_beams": args.num_beams_rescore,
            "lm_batch_size": args.lm_batch_size,
        }

    # Run benchmark
    benchmark = MultiTestBenchmark(
        model_type=args.model_type,
        model_path=args.model_path,
        rescore_config=rescore_config,
    )

    results = benchmark.run(test_datasets)
    benchmark.display_results(results)

    if args.save:
        benchmark.save_results(results)


if __name__ == "__main__":
    main()
