import gradio as gr
import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

FINETUNED_MODEL_NAME = "AleksanderObuchowski/whisper-large-v3-turbo-med-pl-lora"
BASE_MODEL_NAME = "openai/whisper-large-v3-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor (shared between both models)
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL_NAME, language="Polish", task="transcribe"
)

# Load finetuned model
print(f"Loading finetuned model: {FINETUNED_MODEL_NAME}")
finetuned_model = WhisperForConditionalGeneration.from_pretrained(FINETUNED_MODEL_NAME)
finetuned_model.to(DEVICE)

# Load base model
print(f"Loading base model: {BASE_MODEL_NAME}")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
base_model.to(DEVICE)

print(f"Both models loaded on {DEVICE}")


def preprocess_audio(audio):
    """Preprocess audio to the format expected by Whisper."""
    sample_rate, audio_array = audio

    # Convert stereo to mono if needed
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Convert to float32 and normalize
    audio_array = audio_array.astype("float32")
    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(
            audio_array, orig_sr=sample_rate, target_sr=16000
        )

    return audio_array


def transcribe_with_model(audio_array, model):
    """Transcribe audio using a specific model."""
    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=448)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def transcribe(audio):
    """Transcribe audio with both finetuned and base models."""
    if audio is None:
        return "Please provide an audio file.", "Please provide an audio file."

    audio_array = preprocess_audio(audio)

    finetuned_result = transcribe_with_model(audio_array, finetuned_model)
    base_result = transcribe_with_model(audio_array, base_model)

    return finetuned_result, base_result


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="numpy", label="Upload audio or record"),
    outputs=[
        gr.Textbox(label="Finetuned Model (Medical Polish)", lines=10),
        gr.Textbox(label="Standard Whisper", lines=10),
    ],
    title="Whisper Transcription Comparison",
    description="Compare transcriptions between the finetuned Polish medical model and standard Whisper. Upload an audio file or record from your microphone.",
)

if __name__ == "__main__":
    demo.launch(share=True)
