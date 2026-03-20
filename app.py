import os

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

os.environ["AUDIO_DECODER_BACKEND"] = "soundfile"

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE = "Polish"

# Whisper models
WHISPER_FINETUNED = "AleksanderObuchowski/whisper-large-v3-med-pl-lora-decoder-only"
WHISPER_BASE = "openai/whisper-large-v3-turbo"

# Qwen3-ASR models
QWEN3_BASE = "Qwen/Qwen3-ASR-1.7B"
QWEN3_LORA_PATH = "models/Qwen3-ASR-1.7B-med-pl-lora-decoder-only/checkpoint-7350"


def load_whisper_models():
    """Load Whisper models."""
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE, language=LANGUAGE, task="transcribe"
    )

    print(f"Loading finetuned Whisper: {WHISPER_FINETUNED}")
    finetuned = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_FINETUNED, force_download=False
    )
    finetuned.to(DEVICE)

    print(f"Loading base Whisper: {WHISPER_BASE}")
    base = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
    base.to(DEVICE)

    return processor, finetuned, base


def load_qwen3_model():
    """Load Qwen3-ASR model with LoRA adapter."""
    import warnings
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    print(f"Loading Qwen3-ASR base: {QWEN3_BASE}")
    wrapper = Qwen3ASRModel.from_pretrained(QWEN3_BASE, device_map=None)
    processor = wrapper.processor

    if QWEN3_LORA_PATH and os.path.exists(QWEN3_LORA_PATH):
        print(f"Loading LoRA adapter: {QWEN3_LORA_PATH}")
        # Load LoRA onto wrapper.model (Qwen3ASREncoder), not wrapper.model.thinker
        # The adapter keys include "thinker" in the path
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*missing adapter keys.*")
            model = PeftModel.from_pretrained(wrapper.model, QWEN3_LORA_PATH)
        model = model.merge_and_unload()
        # After merge, use .thinker for generation
        model = model.thinker
    else:
        model = wrapper.model.thinker

    model.to(DEVICE)
    model.eval()
    return processor, model


# Load all models at startup
print("=" * 60)
print("Loading models...")
print("=" * 60)

whisper_processor, whisper_finetuned, whisper_base = load_whisper_models()
qwen3_processor, qwen3_model = load_qwen3_model()

print(f"\nAll models loaded on {DEVICE}")
print("=" * 60)


def preprocess_audio(audio):
    """Preprocess audio to 16kHz float32."""
    sample_rate, audio_array = audio

    # Convert stereo to mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Convert to float32 and normalize to [-1, 1]
    audio_array = audio_array.astype("float32")
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val

    # Resample to 16kHz
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    return audio_array


def transcribe_whisper(audio_array, model):
    """Transcribe with Whisper model."""
    input_features = whisper_processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    # Get forced decoder IDs for language and task conditioning
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language=LANGUAGE.lower(), task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_length=448,
            forced_decoder_ids=forced_decoder_ids,
        )

    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def transcribe_qwen3(audio_array):
    """Transcribe with Qwen3-ASR model."""
    tokenizer = qwen3_processor.tokenizer
    messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]
    prompt = qwen3_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt += f"language {LANGUAGE}<asr_text>"

    inputs = qwen3_processor(text=prompt, audio=audio_array, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = qwen3_model.generate(
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


def transcribe(audio):
    """Transcribe audio with all models."""
    if audio is None:
        empty = "Please provide an audio file."
        return empty, empty, empty

    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, audio_data = audio
        if audio_data is None or len(audio_data) == 0:
            empty = "Audio data is empty. Please try recording again."
            return empty, empty, empty
    else:
        empty = "Invalid audio format. Please try again."
        return empty, empty, empty

    audio_array = preprocess_audio(audio)

    # Transcribe with all models
    whisper_finetuned_result = transcribe_whisper(audio_array, whisper_finetuned)
    whisper_base_result = transcribe_whisper(audio_array, whisper_base)
    qwen3_result = transcribe_qwen3(audio_array)

    return qwen3_result, whisper_finetuned_result, whisper_base_result


with gr.Blocks(title="ASR Model Comparison") as demo:
    gr.Markdown("# ASR Model Comparison")
    gr.Markdown(
        "Compare transcriptions between Qwen3-ASR (finetuned), Whisper (finetuned), "
        "and base Whisper. Upload an audio file or record from your microphone."
    )

    with gr.Row():
        audio_input = gr.Audio(
            type="numpy",
            label="Upload audio or record",
            sources=["microphone", "upload"],
        )

    submit_btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        qwen3_output = gr.Textbox(label="Qwen3-ASR (Finetuned)", lines=10)
        whisper_finetuned_output = gr.Textbox(label="Whisper (Finetuned)", lines=10)
        whisper_base_output = gr.Textbox(label="Whisper (Base)", lines=10)

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[qwen3_output, whisper_finetuned_output, whisper_base_output],
    )

    audio_input.stop_recording(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[qwen3_output, whisper_finetuned_output, whisper_base_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
