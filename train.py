# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.50.0",
#     "torch>=2.0.0",
#     "datasets>=4.3.0",
#     "evaluate>=0.4.6",
#     "accelerate>=1.0.0",
#     "peft>=0.18.0",
#     "wandb>=0.22.3",
#     "python-dotenv>=1.2.1",
#     "numpy<2.0.0",
#     "huggingface-hub>=0.23.0",
#     "jiwer>=4.0.0",
#     "levenshtein>=0.27.3",
#     "rich>=13.0.0",
#     "torchcodec>=0.4.0",
# ]
# ///

"""
Fine-tuning script for Whisper and GLM-ASR on Polish medical ASR.

For Qwen3-ASR, use train_qwen3_asr.py instead.

Usage:
    uv run train.py
    uv run train.py --config configs/experiment.yaml
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Parse arguments early
parser = argparse.ArgumentParser(description="Fine-tune Whisper/GLM-ASR")
parser.add_argument(
    "--config", default="configs/experiment.yaml", help="Path to experiment config"
)
args = parser.parse_args()
CONFIG_PATH = args.config

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import wandb

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "Polish"

# Training
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

# LoRA
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Data
MAX_AUDIO_SECONDS = 30
EVAL_SUBSET_SIZE = 50000

# Options
FREEZE_ENCODER = True
PUSH_TO_HUB = True
WANDB_PROJECT = "eskulap-a"
MAX_STEPS = None  # Set to int for quick testing


# =============================================================================
# Model type detection and batch size config
# =============================================================================


def get_model_type(model_name: str) -> str:
    """Detect model type from model name."""
    model_name_lower = model_name.lower()
    if "whisper" in model_name_lower:
        return "whisper"
    elif "glm-asr" in model_name_lower:
        return "glm-asr"
    else:
        raise ValueError(
            f"Unknown model type: {model_name}. Use train_qwen3_asr.py for Qwen3-ASR."
        )


MODEL_TYPE = get_model_type(BASE_MODEL_NAME)

# Batch size based on model
if MODEL_TYPE == "glm-asr":
    BATCH_SIZE = 1  # GLM-ASR is 1.5B params, needs small batch
    GRADIENT_ACCUMULATION_STEPS = 16
else:  # whisper
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1

# Sweep overrides from environment
if os.environ.get("SWEEP_LR"):
    LEARNING_RATE = float(os.environ["SWEEP_LR"])
if os.environ.get("SWEEP_LORA_R"):
    LORA_R = int(os.environ["SWEEP_LORA_R"])
if os.environ.get("SWEEP_BATCH_SIZE"):
    BATCH_SIZE = int(os.environ["SWEEP_BATCH_SIZE"])
if os.environ.get("SWEEP_MAX_STEPS"):
    MAX_STEPS = int(os.environ["SWEEP_MAX_STEPS"])
if os.environ.get("SWEEP_NO_HUB"):
    PUSH_TO_HUB = False

# Derived names
RUN_NAME = f"{BASE_MODEL_NAME.split('/')[-1]}-med-pl"
if USE_LORA:
    RUN_NAME += "-lora"
if FREEZE_ENCODER:
    RUN_NAME += "-decoder-only"

# =============================================================================
# Data paths
# =============================================================================

from data.experiment_config import ExperimentConfig
from data.training_dataset import (
    compute_training_cache_fingerprint,
    resolve_processed_data_path,
)

PROCESSED_DATA_DIR = os.environ.get(
    "PROCESSED_DATA_DIR", "/mnt/data/Eskulap-a/processed_data"
)
_experiment_cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
_training_cache_fp = compute_training_cache_fingerprint(
    config_path=CONFIG_PATH,
    model_family=MODEL_TYPE,
    base_model_name=BASE_MODEL_NAME,
    inner_dev_from_train=_experiment_cfg.evaluation.inner_dev_from_train,
)
PROCESSED_DATA_PATH = resolve_processed_data_path(
    PROCESSED_DATA_DIR, BASE_MODEL_NAME.split("/")[-1], _training_cache_fp
)

# =============================================================================
# Load processor and tokenizer
# =============================================================================

print(f"Model type: {MODEL_TYPE}")
print(f"Loading processor from {BASE_MODEL_NAME}")

if MODEL_TYPE == "whisper":
    tokenizer = WhisperTokenizer.from_pretrained(
        BASE_MODEL_NAME, language="Polish", task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_NAME, language="Polish", task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
elif MODEL_TYPE == "glm-asr":
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

print(f"Tokenizer: eos={tokenizer.eos_token}, pad={tokenizer.pad_token}")

# =============================================================================
# Dataset preparation
# =============================================================================


def prepare_whisper_batch(batch):
    """Process batch for Whisper model."""
    valid = [
        i
        for i in range(len(batch["audio"]))
        if batch["audio"][i] is not None and batch["text"][i] is not None
    ]
    if not valid:
        return {"input_features": [], "labels": []}
    audio_arrays = [batch["audio"][i]["array"] for i in valid]
    texts = [batch["text"][i] for i in valid]
    return {
        "input_features": feature_extractor(
            audio_arrays, sampling_rate=16000
        ).input_features,
        "labels": tokenizer(texts).input_ids,
    }


def prepare_glm_asr_batch(batch):
    """Process batch for GLM-ASR model."""
    valid = [
        i
        for i in range(len(batch["audio"]))
        if batch["audio"][i] is not None and batch["text"][i] is not None
    ]
    audio_arrays = [batch["audio"][i]["array"] for i in valid]
    texts = [batch["text"][i] for i in valid]
    eos_token_id = tokenizer.eos_token_id

    all_input_features = []
    all_input_features_mask = []
    all_input_ids = []
    all_labels = []

    for audio, text in zip(audio_arrays, texts):
        inputs = processor.apply_transcription_request(audio)
        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(prompt_ids)

        # Tokenize target and add EOS
        target_ids = tokenizer(text, add_special_tokens=False).input_ids
        target_ids = target_ids + [eos_token_id]

        # Full sequence: prompt + target
        full_input_ids = prompt_ids + target_ids
        labels = [-100] * prompt_len + target_ids

        all_input_features.append(inputs["input_features"][0].numpy())
        all_input_features_mask.append(inputs["input_features_mask"][0].numpy())
        all_input_ids.append(full_input_ids)
        all_labels.append(labels)

    batch["input_features"] = all_input_features
    batch["input_features_mask"] = all_input_features_mask
    batch["input_ids"] = all_input_ids
    batch["labels"] = all_labels
    return batch


# =============================================================================
# Load or process dataset
# =============================================================================

from data import load_test_data, load_train_data
from data.training_dataset import load_or_build_processed_dataset

print(f"Loading data from config: {CONFIG_PATH}")
raw_train = load_train_data(config_path=CONFIG_PATH)
print(f"Raw train: {len(raw_train)} samples")

inner_dev = _experiment_cfg.evaluation.inner_dev_from_train
raw_test_dict = None
if not inner_dev:
    raw_test_dict = load_test_data(config_path=CONFIG_PATH)
    if not raw_test_dict:
        print(
            "Warning: no enabled test datasets in config; "
            "falling back to inner_dev_from_train split."
        )
        inner_dev = True

if inner_dev:
    # Legacy: eval split carved from training concat by unique text
    # (same text can appear multiple times with different audio)
    all_texts = raw_train["text"]
    unique_texts = list(set(all_texts))
    random.seed(42)
    random.shuffle(unique_texts)
    dev_text_count = max(1, min(500, len(unique_texts) // 10))
    dev_texts = set(unique_texts[:dev_text_count])
    dev_indices = [i for i, t in enumerate(all_texts) if t in dev_texts]
    train_indices = [i for i, t in enumerate(all_texts) if t not in dev_texts]
    raw_splits = DatasetDict({
        "train": raw_train.select(train_indices),
        "test": raw_train.select(dev_indices),
    })
else:
    raw_test = concatenate_datasets(list(raw_test_dict.values()))
    print(f"Raw test (held-out from config): {len(raw_test)} samples")
    raw_splits = DatasetDict({"train": raw_train, "test": raw_test})

prepare_fn = (
    prepare_whisper_batch if MODEL_TYPE == "whisper" else prepare_glm_asr_batch
)
dataset = load_or_build_processed_dataset(
    processed_data_path=PROCESSED_DATA_PATH,
    raw_splits=raw_splits,
    backend="map_batched",
    prepare_batch=prepare_fn,
    map_batch_size=50,
    map_num_proc=8,
    required_cache_columns=None,
)
print(f"Processed dataset at {PROCESSED_DATA_PATH}")

# Filter long audio for GLM-ASR (variable length features)
if MODEL_TYPE == "glm-asr":
    MAX_AUDIO_FRAMES = 2000  # ~20 seconds at 100 frames/sec
    initial = len(dataset["train"])
    dataset = dataset.filter(
        lambda x: len(x["input_features"]) <= MAX_AUDIO_FRAMES,
        num_proc=8,
    )
    print(
        f"Filtered: {initial} -> {len(dataset['train'])} samples (max {MAX_AUDIO_FRAMES} frames)"
    )

# Create eval subset
eval_dataset = dataset["test"]
if len(eval_dataset) > EVAL_SUBSET_SIZE:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
print(f"Train: {len(dataset['train'])}, Eval: {len(eval_dataset)} samples")

# =============================================================================
# Model setup
# =============================================================================

print(f"Loading model: {BASE_MODEL_NAME}")

if MODEL_TYPE == "whisper":
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.float16
    )
    model.generation_config.language = "polish"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
elif MODEL_TYPE == "glm-asr":
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id


def get_whisper_lora_target_modules(
    model: WhisperForConditionalGeneration, decoder_only: bool
) -> List[str]:
    """Select Whisper LoRA target modules.

    When decoder_only=True, only decoder projection layers are selected.
    """
    prefix = "model.decoder." if decoder_only else "model."
    target_modules = []
    for module_name, module in model.named_modules():
        if not module_name.startswith(prefix):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(
            module_name.endswith(suffix)
            for suffix in ("q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2")
        ):
            target_modules.append(module_name)

    if not target_modules:
        scope = "decoder" if decoder_only else "model"
        raise RuntimeError(f"No Whisper LoRA target modules found in {scope} scope")

    return sorted(set(target_modules))

# Freeze encoder
if FREEZE_ENCODER:
    if MODEL_TYPE == "whisper":
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    elif MODEL_TYPE == "glm-asr":
        if hasattr(model, "audio_tower"):
            for param in model.audio_tower.parameters():
                param.requires_grad = False
        elif hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif hasattr(model, "model") and hasattr(model.model, "encoder"):
            for param in model.model.encoder.parameters():
                param.requires_grad = False
    print("Encoder frozen - only decoder will be trained")

# Apply LoRA
if USE_LORA:
    if MODEL_TYPE == "whisper":
        lora_target_modules = get_whisper_lora_target_modules(
            model, decoder_only=FREEZE_ENCODER
        )
    else:  # glm-asr
        lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=lora_target_modules,
            bias="none",
            task_type=None,
        ),
    )
    model.enable_input_require_grads()

    if FREEZE_ENCODER and MODEL_TYPE == "whisper":
        for param_name, param in model.named_parameters():
            if ".encoder." in param_name:
                param.requires_grad = False

        encoder_trainable = [
            param_name
            for param_name, param in model.named_parameters()
            if param.requires_grad and ".encoder." in param_name
        ]
        if encoder_trainable:
            raise RuntimeError(
                "Strict decoder-only mode failed: encoder parameters are still trainable"
            )
        print("Strict decoder-only mode enabled - no trainable encoder parameters")

    model.print_trainable_parameters()

# =============================================================================
# Data collator
# =============================================================================


@dataclass
class ASRDataCollator:
    processor: Any
    model_type: str = "whisper"
    decoder_start_token_id: int = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if self.model_type == "glm-asr":
            return self._collate_glm_asr(features)
        else:
            return self._collate_whisper(features)

    def _collate_whisper(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )
        # Cast to float16 to match model dtype when using fp16 training
        batch["input_features"] = batch["input_features"].half()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if self.decoder_start_token_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

        # Build decoder_input_ids explicitly (shift labels right).
        # Required when label_smoothing > 0 because the Trainer pops
        # "labels" before calling model.forward(), so the model can no
        # longer derive decoder_input_ids from labels on its own.
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
        decoder_input_ids = torch.cat(
            [
                torch.full(
                    (decoder_input_ids.shape[0], 1),
                    self.decoder_start_token_id,
                    dtype=decoder_input_ids.dtype,
                ),
                decoder_input_ids[:, :-1],
            ],
            dim=1,
        )

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        return batch

    def _collate_glm_asr(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad input_features_mask
        max_time_len = batch["input_features"].shape[-1]
        input_features_masks = []
        for f in features:
            mask = f["input_features_mask"]
            if len(mask) < max_time_len:
                padded_mask = np.pad(
                    mask, (0, max_time_len - len(mask)), mode="constant"
                )
            else:
                padded_mask = mask[:max_time_len]
            input_features_masks.append(padded_mask)
        batch["input_features_mask"] = torch.tensor(
            np.stack(input_features_masks), dtype=torch.int32
        )

        # Pad input_ids and labels
        max_seq_len = max(len(f["input_ids"]) for f in features)
        pad_token_id = self.processor.tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            seq_len = len(input_ids)
            padding_len = max_seq_len - seq_len

            padded_input_ids.append(input_ids + [pad_token_id] * padding_len)
            padded_labels.append(labels + [-100] * padding_len)
            attention_masks.append([1] * seq_len + [0] * padding_len)

        batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)

        return batch


# Initialize data collator
decoder_start_token_id = getattr(
    model.config,
    "decoder_start_token_id",
    getattr(model.config, "bos_token_id", tokenizer.bos_token_id),
)
data_collator = ASRDataCollator(
    processor=processor,
    model_type=MODEL_TYPE,
    decoder_start_token_id=decoder_start_token_id,
)

# =============================================================================
# Metrics
# =============================================================================

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    from data.text_normalize import normalize_text_for_asr

    preds = [normalize_text_for_asr(p) for p in preds]
    refs = [normalize_text_for_asr(r) for r in refs]

    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    cer = 100 * cer_metric.compute(predictions=preds, references=refs)

    # Log examples to wandb
    indices = random.sample(range(len(preds)), min(50, len(preds)))
    table = wandb.Table(columns=["Reference", "Prediction"])
    for i in indices:
        table.add_data(refs[i], preds[i])
    wandb.log({"examples": table})

    return {"wer": wer, "cer": cer}


# =============================================================================
# Training
# =============================================================================

num_batches = len(dataset["train"]) // BATCH_SIZE

wandb.init(
    project=WANDB_PROJECT,
    config={
        "model": BASE_MODEL_NAME,
        "model_type": MODEL_TYPE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "lora": USE_LORA,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "freeze_encoder": FREEZE_ENCODER,
    },
)

training_kwargs = {
    "output_dir": f"./models/{RUN_NAME}",
    "hub_model_id": RUN_NAME,
    "run_name": RUN_NAME,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": max(1, BATCH_SIZE // 2),
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing_factor": LABEL_SMOOTHING,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": MAX_STEPS or -1,
    "eval_strategy": "steps" if MAX_STEPS else "epoch",
    "eval_steps": 5 if MAX_STEPS else None,
    "save_strategy": "steps" if MAX_STEPS else "epoch",
    "save_steps": 5 if MAX_STEPS else None,
    "predict_with_generate": True,
    "generation_max_length": 225,
    "gradient_checkpointing": True,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "logging_steps": max(1, int(num_batches * 0.1)),
    "report_to": ["wandb"],
    "push_to_hub": PUSH_TO_HUB,
    "label_names": ["labels"],
    "remove_unused_columns": False,
}

# Model-specific settings
if MODEL_TYPE == "whisper":
    training_kwargs["fp16"] = True
elif MODEL_TYPE == "glm-asr":
    # GLM-ASR has dtype issues with mixed precision
    training_kwargs["generation_max_length"] = 1024
    training_kwargs["predict_with_generate"] = False

training_args = Seq2SeqTrainingArguments(**training_kwargs)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# Baseline evaluation
print("Calculating baseline metrics...")
baseline_metrics = trainer.evaluate()
print(f"Baseline: {baseline_metrics}")
wandb.log({"baseline": baseline_metrics})

# Train
trainer.train()

# Push to hub
if PUSH_TO_HUB:
    trainer.push_to_hub(
        commit_message="Training complete",
        language="pl",
        finetuned_from=BASE_MODEL_NAME,
        tasks="automatic-speech-recognition",
    )

print("Done!")
