# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "qwen-asr==0.0.6",
#     "transformers==4.57.6",
#     "torch>=2.0.0",
#     "datasets>=4.3.0",
#     "evaluate>=0.4.6",
#     "accelerate==1.12.0",
#     "peft>=0.18.0",
#     "wandb>=0.22.3",
#     "python-dotenv>=1.2.1",
#     "nagisa==0.2.11",
#     "soynlp==0.0.493",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "numpy<2.0.0",
#     "huggingface-hub>=0.23.0",
#     "jiwer>=4.0.0",
#     "levenshtein>=0.27.3",
#     "qwen-omni-utils",
#     "rich>=13.0.0",
#     "torchcodec>=0.3.0",
# ]
# ///

"""
Fine-tuning script for Qwen3-ASR on Polish medical ASR.

Usage:
    uv run train_qwen3_asr.py
    uv run train_qwen3_asr.py --config configs/experiment.yaml
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AUDIO_DECODER_BACKEND"] = "soundfile"
import random
import sys

# Parse arguments early
parser = argparse.ArgumentParser(description="Fine-tune Qwen3-ASR")
parser.add_argument(
    "--config", default="configs/experiment.yaml", help="Path to experiment config"
)
args = parser.parse_args()
CONFIG_PATH = args.config
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import wandb

# Environment setup


# =============================================================================
# Configuration
# =============================================================================

# BASE_MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
BASE_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
LANGUAGE = "Polish"

# Training
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 10
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.1

# LoRA
USE_LORA = True
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Data
MAX_AUDIO_FRAMES = 1500
EVAL_SUBSET_SIZE = 200

# Options
FREEZE_ENCODER = True
PUSH_TO_HUB = True
WANDB_PROJECT = "eskulap-a"
MAX_STEPS = None  # Set to int for quick testing

# Derived
RUN_NAME = f"{BASE_MODEL_NAME.split('/')[-1]}-med-pl"
if USE_LORA:
    RUN_NAME += "-lora"
if FREEZE_ENCODER:
    RUN_NAME += "-decoder-only"

# =============================================================================
# Compatibility patches for transformers
# =============================================================================

if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS.get(
        "linear", ROPE_INIT_FUNCTIONS.get("llama3", lambda *args, **kwargs: None)
    )
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRTextConfig,
)

Qwen3ASRTextConfig.convert_rope_params_to_dict = lambda self, **kwargs: kwargs


def _standardize_rope_params(self):
    if getattr(self, "_rope_standardized", False):
        return
    self.rope_parameters = {
        "rope_theta": getattr(self, "rope_theta", 5000000.0),
        "factor": 1.0,
        "rope_type": "default",
        "head_dim": getattr(self, "head_dim", None)
        or getattr(self, "hidden_size", 896)
        // getattr(self, "num_attention_heads", 14),
    }
    self._rope_standardized = True


Qwen3ASRTextConfig.standardize_rope_params = _standardize_rope_params

# =============================================================================
# Data paths
# =============================================================================

sys.path.insert(0, str(Path(__file__).parent))

from data.experiment_config import ExperimentConfig
from data.training_dataset import (
    compute_training_cache_fingerprint,
    load_or_build_processed_dataset,
    resolve_processed_data_path,
)

PROCESSED_DATA_DIR = os.environ.get(
    "PROCESSED_DATA_DIR", "/mnt/data/Eskulap-a/processed_data"
)
_experiment_cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
_training_cache_fp = compute_training_cache_fingerprint(
    config_path=CONFIG_PATH,
    model_family="qwen3-asr",
    base_model_name=BASE_MODEL_NAME,
    inner_dev_from_train=_experiment_cfg.evaluation.inner_dev_from_train,
)
PROCESSED_DATA_PATH = resolve_processed_data_path(
    PROCESSED_DATA_DIR, BASE_MODEL_NAME.split("/")[-1], _training_cache_fp
)

# =============================================================================
# Load processor and tokenizer
# =============================================================================

print(f"Loading processor from {BASE_MODEL_NAME}")
_wrapper = Qwen3ASRModel.from_pretrained(BASE_MODEL_NAME, device_map=None)
processor = _wrapper.processor
tokenizer = processor.tokenizer
del _wrapper
print(f"Tokenizer: eos={tokenizer.eos_token}, pad={tokenizer.pad_token}")

# =============================================================================
# Dataset preparation
# =============================================================================


def prepare_single(example):
    """Process single audio sample into model inputs with proper prompt format."""
    audio = example["audio"]["array"]
    text = example["text"]

    # Build prompt with language tag
    messages = [{"role": "user", "content": [{"type": "audio", "audio": audio}]}]
    base_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt = base_prompt + f"language {LANGUAGE}<asr_text>"

    # Process audio features
    audio_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )

    # Tokenize prompt to get its length
    prompt_ids = processor.tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Full sequence: prompt + transcription + EOS
    full_text = prompt + text + tokenizer.eos_token
    full_ids = processor.tokenizer(full_text, return_tensors="pt")["input_ids"]

    # Create labels with prompt masked
    labels = full_ids[0].clone()
    labels[:prompt_len] = -100

    return {
        "input_features": audio_features["input_features"][0].numpy(),
        "input_ids": full_ids[0].tolist(),
        "labels": labels.tolist(),
        "prompt_length": prompt_len,
    }


# Import the new data loading API
from data import load_test_data, load_train_data

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

# Chunked loop avoids dataset.map hangs with this processor
dataset = load_or_build_processed_dataset(
    processed_data_path=PROCESSED_DATA_PATH,
    raw_splits=raw_splits,
    backend="chunked_loop",
    prepare_single=prepare_single,
    chunked_size=500,
    required_cache_columns=["prompt_length"],
)
print(f"Processed dataset at {PROCESSED_DATA_PATH}")

# Filter long samples and create eval subset
initial = len(dataset["train"])
print(f"Filtering samples longer than {MAX_AUDIO_FRAMES} frames...")
# Use select instead of filter (filter has same hang issue as map)
from tqdm import tqdm

train_indices = [
    i
    for i in tqdm(range(len(dataset["train"])), desc="Filter train")
    if len(dataset["train"][i]["input_features"]) <= MAX_AUDIO_FRAMES
]
test_indices = [
    i
    for i in tqdm(range(len(dataset["test"])), desc="Filter test")
    if len(dataset["test"][i]["input_features"]) <= MAX_AUDIO_FRAMES
]
dataset = DatasetDict(
    {
        "train": dataset["train"].select(train_indices),
        "test": dataset["test"].select(test_indices),
    }
)
print(f"Filtered: {initial} -> {len(dataset['train'])} samples")

eval_dataset = dataset["test"]
if len(eval_dataset) > EVAL_SUBSET_SIZE:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
print(f"Eval: {len(eval_dataset)} samples")

# =============================================================================
# Model setup
# =============================================================================

print(f"Loading model: {BASE_MODEL_NAME}")
model = Qwen3ASRModel.from_pretrained(
    BASE_MODEL_NAME, dtype=torch.bfloat16, device_map=None
).model


# Patch forward for HF Trainer
def _forward(
    self, input_ids=None, attention_mask=None, input_features=None, labels=None, **kw
):
    return self.thinker.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        labels=labels,
        **kw,
    )


model.__class__.forward = _forward

# Configure token IDs
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
if hasattr(model, "generation_config"):
    model.generation_config.max_new_tokens = 128
    model.generation_config.max_length = None
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# Freeze encoder
if FREEZE_ENCODER and hasattr(model.thinker, "audio_tower"):
    for param in model.thinker.audio_tower.parameters():
        param.requires_grad = False
    print("Encoder frozen")


# Patch for LoRA compatibility
def _get_input_embeddings(self):
    if hasattr(self, "thinker") and hasattr(self.thinker, "model"):
        return self.thinker.model.embed_tokens
    return None


model.__class__.get_input_embeddings = _get_input_embeddings

# Apply LoRA
if USE_LORA:
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type=None,
        ),
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

# =============================================================================
# Data collator
# =============================================================================


@dataclass
class ASRDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad audio features (right padding)
        feats = [np.array(f["input_features"]) for f in features]
        lengths = [f.shape[-1] for f in feats]
        max_len = max(lengths)

        padded_feats = []
        feat_masks = []
        for feat, orig_len in zip(feats, lengths):
            pad = max_len - orig_len
            padded_feats.append(np.pad(feat, ((0, 0), (0, pad))) if pad > 0 else feat)
            feat_masks.append([1] * orig_len + [0] * pad)

        # Pad text (right padding for training)
        max_seq = max(len(f["input_ids"]) for f in features)
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids, labels, attn_masks = [], [], []
        for f in features:
            seq_len = len(f["input_ids"])
            pad = max_seq - seq_len
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn_masks.append([1] * seq_len + [0] * pad)

        batch = {
            "input_features": torch.tensor(np.stack(padded_feats), dtype=torch.float32),
            "feature_attention_mask": torch.tensor(feat_masks, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
        }

        if "prompt_length" in features[0]:
            batch["prompt_length"] = torch.tensor(
                [f["prompt_length"] for f in features], dtype=torch.long
            )

        return batch


# =============================================================================
# Metrics
# =============================================================================

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions.copy()
    label_ids = pred.label_ids.copy()

    # Replace invalid tokens with pad (token 0 is '!', not padding!)
    vocab_size = len(tokenizer)
    pred_ids[(pred_ids < 0) | (pred_ids >= vocab_size)] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize text for consistent evaluation
    from data.text_normalize import normalize_text_for_asr

    preds = [normalize_text_for_asr(p) or "<empty>" for p in preds]
    refs = [normalize_text_for_asr(r) or "<empty>" for r in refs]

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
# Trainer
# =============================================================================


class ASRTrainer(Seq2SeqTrainer):
    """Trainer with left-padding for generation (required for decoder-only models)."""

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype:
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=dtype)
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )

        inputs = self._prepare_inputs(inputs)
        prompt_lengths = inputs.pop("prompt_length", None)
        labels = inputs.get("labels")

        # Compute loss
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Generate with left-padded prompts
        generated = None
        if prompt_lengths is not None:
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            max_prompt = prompt_lengths.max().item()
            pad_id = tokenizer.pad_token_id

            # Apply LEFT padding for each sequence
            padded_ids, padded_mask = [], []
            for i, plen in enumerate(prompt_lengths):
                plen = plen.item()
                ids = input_ids[i, :plen]
                mask = attn_mask[i, :plen]
                pad = max_prompt - plen
                if pad > 0:
                    ids = torch.cat(
                        [
                            torch.full(
                                (pad,), pad_id, dtype=ids.dtype, device=ids.device
                            ),
                            ids,
                        ]
                    )
                    mask = torch.cat(
                        [torch.zeros(pad, dtype=mask.dtype, device=mask.device), mask]
                    )
                if pad > 0:
                    ids = torch.cat(
                        [
                            torch.full(
                                (pad,), pad_id, dtype=ids.dtype, device=ids.device
                            ),
                            ids,
                        ]
                    )
                    mask = torch.cat(
                        [torch.zeros(pad, dtype=mask.dtype, device=mask.device), mask]
                    )
                padded_ids.append(ids)
                padded_mask.append(mask)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=torch.stack(padded_ids),
                    attention_mask=torch.stack(padded_mask),
                    input_features=inputs["input_features"],
                    feature_attention_mask=inputs.get("feature_attention_mask"),
                    max_new_tokens=self.args.generation_max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    repetition_penalty=1.2,
                )

        return loss, generated, labels


# Patch generate to return only new tokens
_orig_generate = model.generate


def _generate(input_ids=None, **kwargs):
    input_len = input_ids.shape[1] if input_ids is not None else 0
    out = _orig_generate(input_ids=input_ids, **kwargs)
    seqs = out.sequences if hasattr(out, "sequences") else out
    return seqs[:, input_len:] if input_len > 0 else seqs


model.generate = _generate

# =============================================================================
# Training
# =============================================================================

wandb.init(
    project=WANDB_PROJECT,
    config={
        "model": BASE_MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_TRAIN_EPOCHS,
        "lora": USE_LORA,
        "freeze_encoder": FREEZE_ENCODER,
    },
)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{RUN_NAME}",
    hub_model_id=RUN_NAME,
    run_name=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS or -1,
    eval_strategy="steps" if MAX_STEPS else "epoch",
    eval_steps=5 if MAX_STEPS else None,
    save_strategy="steps" if MAX_STEPS else "epoch",
    save_steps=5 if MAX_STEPS else None,
    predict_with_generate=True,
    generation_max_length=128,
    eval_accumulation_steps=5,
    gradient_checkpointing=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_steps=50,
    report_to=["wandb"],
    push_to_hub=PUSH_TO_HUB,
    label_names=["labels"],
    remove_unused_columns=False,
    bf16=True,
)

trainer = ASRTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
    data_collator=ASRDataCollator(processor=processor),
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print(f"Training: {len(dataset['train'])} samples, Eval: {len(eval_dataset)} samples")
trainer.train()

if PUSH_TO_HUB:
    trainer.push_to_hub(
        commit_message="Training complete",
        language="pl",
        finetuned_from=BASE_MODEL_NAME,
        tasks="automatic-speech-recognition",
    )

print("Done!")
