"""
Fine-tuning script for Cohere Transcribe on Polish medical ASR.

Usage:
    uv run train_cohere_asr.py
    uv run train_cohere_asr.py --config configs/experiment.yaml
"""

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
import wandb
from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from data import load_train_data

load_dotenv(Path(__file__).parent / ".env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


parser = argparse.ArgumentParser(description="Fine-tune Cohere ASR")
parser.add_argument(
    "--config", default="configs/experiment.yaml", help="Path to experiment config"
)
args = parser.parse_args()
CONFIG_PATH = args.config


# =============================================================================
# Configuration
# =============================================================================
def _getenv_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


BASE_MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"
LANGUAGE = "pl"

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
EVAL_SUBSET_SIZE = _getenv_int("EVAL_SUBSET_SIZE", 50000)
TRAIN_SAMPLE_CAP = _getenv_int("TRAIN_SAMPLE_CAP")
WER_MAX_SAMPLES = _getenv_int("WER_MAX_SAMPLES", 200)
EVAL_STEPS = _getenv_int("EVAL_STEPS", 5)
SAVE_STEPS = _getenv_int("SAVE_STEPS", EVAL_STEPS)
RUN_SUFFIX = os.environ.get("RUN_SUFFIX", "").strip()
CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "").strip()

# Options
FREEZE_ENCODER = True
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "1") == "1"
WANDB_PROJECT = "eskulap-a"
MAX_STEPS = int(os.environ["SWEEP_MAX_STEPS"]) if os.environ.get("SWEEP_MAX_STEPS") else None

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

RUN_NAME = f"{BASE_MODEL_NAME.split('/')[-1]}-med-pl"
if USE_LORA:
    RUN_NAME += "-lora"
if FREEZE_ENCODER:
    RUN_NAME += "-decoder-only"
if TRAIN_SAMPLE_CAP:
    RUN_NAME += f"-n{TRAIN_SAMPLE_CAP}"
if RUN_SUFFIX:
    RUN_NAME += f"-{RUN_SUFFIX}"

PROCESSED_DATA_DIR = os.environ.get(
    "PROCESSED_DATA_DIR", "/mnt/data/Eskulap-a/processed_data"
)
processed_data_name = f"processed_data_{BASE_MODEL_NAME.split('/')[-1]}"
if TRAIN_SAMPLE_CAP:
    processed_data_name += f"_n{TRAIN_SAMPLE_CAP}"
if CACHE_SUFFIX:
    processed_data_name += f"_{CACHE_SUFFIX}"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, processed_data_name)


print(f"Loading processor from {BASE_MODEL_NAME}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor
print(f"Tokenizer: eos={tokenizer.eos_token}, pad={tokenizer.pad_token}")


def build_cohere_decoder_prompt_text(language: str, punctuation: bool = True) -> str:
    """Match CohereAsrForConditionalGeneration.build_prompt (Cohere remote modeling code)."""
    pnc_token = "<|pnc|>" if punctuation else "<|nopnc|>"
    task_token = "<|noitn|>"
    return (
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        f"<|{language}|><|{language}|>{pnc_token}{task_token}<|notimestamp|><|nodiarize|>"
    )


def get_decoder_prompt_ids(language: str) -> List[int]:
    """Tokenize the same decoder prefix inference uses (requires ``text=`` — ``language=`` alone is ignored)."""
    dummy_audio = np.zeros(16000, dtype=np.float32)
    prompt_text = build_cohere_decoder_prompt_text(language)
    sr = int(feature_extractor.sampling_rate)
    prompt_inputs = processor(
        audio=[dummy_audio],
        text=[prompt_text],
        sampling_rate=sr,
        return_tensors="pt",
    )
    ids = prompt_inputs.get("input_ids")
    if ids is None or ids.numel() == 0:
        return []
    return ids[0].tolist()


DECODER_PROMPT_IDS = get_decoder_prompt_ids(LANGUAGE)


def prepare_cohere_batch(batch):
    valid = [
        i
        for i in range(len(batch["audio"]))
        if batch["audio"][i] is not None and batch["text"][i] is not None
    ]
    if not valid:
        return {"input_features": [], "length": [], "labels": []}

    audio_arrays = [batch["audio"][i]["array"] for i in valid]
    texts = [batch["text"][i] for i in valid]

    sr = int(feature_extractor.sampling_rate)
    feats = feature_extractor(audio_arrays, sampling_rate=sr)
    enc = tokenizer(texts, add_special_tokens=False)
    feat_lengths = feats["length"].tolist() if hasattr(feats["length"], "tolist") else list(feats["length"])
    eos_id = tokenizer.eos_token_id
    label_rows = []
    for ids in enc.input_ids:
        row = list(ids)
        if eos_id is not None:
            row.append(eos_id)
        label_rows.append(row)
    return {
        "input_features": feats["input_features"],
        "length": feat_lengths,
        "labels": label_rows,
    }


def process_split(split_dataset, save_path: str, desc="Processing"):
    """Process split and save incrementally to Arrow to avoid OOM."""
    import gc
    import tempfile

    import pyarrow as pa
    from datasets.arrow_writer import ArrowWriter
    from tqdm import tqdm

    chunk_size = 32
    tmp_arrow = os.path.join(tempfile.gettempdir(), f"_cohere_{desc.replace(' ', '_')}.arrow")

    schema = None
    writer = None
    total_rows = 0

    for start in tqdm(range(0, len(split_dataset), chunk_size), desc=desc):
        end = min(start + chunk_size, len(split_dataset))
        batch = split_dataset.select(range(start, end))
        prepared = prepare_cohere_batch(batch)
        if len(prepared["input_features"]) == 0:
            continue
        chunk = Dataset.from_dict(prepared)
        if writer is None:
            writer = ArrowWriter(path=tmp_arrow, schema=chunk.data.schema)
        writer.write_table(chunk.data.table)
        total_rows += len(chunk)
        del chunk, prepared, batch
        gc.collect()

    if writer is None or total_rows == 0:
        return Dataset.from_dict({"input_features": [], "length": [], "labels": []})
    writer.finalize()
    ds = Dataset.from_file(tmp_arrow)
    return ds


def check_cache_valid() -> bool:
    return os.path.exists(PROCESSED_DATA_PATH)


dataset = None
if check_cache_valid():
    print(f"Loading cached dataset from {PROCESSED_DATA_PATH}")
    dataset = load_from_disk(PROCESSED_DATA_PATH)

if dataset is None:
    import shutil

    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)

    print(f"Loading data from config: {CONFIG_PATH}")
    raw_train = load_train_data(config_path=CONFIG_PATH)
    if TRAIN_SAMPLE_CAP:
        train_cap = min(len(raw_train), TRAIN_SAMPLE_CAP)
        raw_train = raw_train.shuffle(seed=42).select(range(train_cap))
        print(f"Subset mode: capped raw train to {train_cap} samples")
    elif MAX_STEPS:
        # For smoke runs, cap preprocessing size so verification is fast/stable.
        smoke_cap = min(len(raw_train), 3000)
        raw_train = raw_train.shuffle(seed=42).select(range(smoke_cap))
        print(f"MAX_STEPS mode: capped raw train to {smoke_cap} samples")
    print(f"Raw train: {len(raw_train)} samples")

    all_texts = raw_train["text"]
    unique_texts = list(set(all_texts))
    random.seed(42)
    random.shuffle(unique_texts)
    dev_text_count = max(1, min(500, len(unique_texts) // 10))
    dev_texts = set(unique_texts[:dev_text_count])

    dev_indices = [i for i, t in enumerate(all_texts) if t in dev_texts]
    train_indices = [i for i, t in enumerate(all_texts) if t not in dev_texts]

    split = DatasetDict(
        {
            "train": raw_train.select(train_indices),
            "test": raw_train.select(dev_indices),
        }
    )

    train_processed = process_split(split["train"], PROCESSED_DATA_PATH + "_train_tmp", desc="Processing train")
    test_processed = process_split(split["test"], PROCESSED_DATA_PATH + "_test_tmp", desc="Processing test")
    dataset = DatasetDict({"train": train_processed, "test": test_processed})
    dataset.save_to_disk(PROCESSED_DATA_PATH)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

eval_dataset = dataset["test"]
if len(eval_dataset) > EVAL_SUBSET_SIZE:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
print(f"Train: {len(dataset['train'])}, Eval: {len(eval_dataset)} samples")


print(f"Loading model: {BASE_MODEL_NAME}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float32,
)


def get_cohere_lora_target_modules(model, decoder_only: bool) -> List[str]:
    if any(name.startswith("transf_decoder.") for name, _ in model.named_modules()):
        prefix = "transf_decoder." if decoder_only else ""
        suffixes = (
            "query_net",
            "key_net",
            "value_net",
            "out_projection",
            "dense_in",
            "dense_out",
        )
    else:
        prefix = "model.decoder." if decoder_only else "model."
        suffixes = ("q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2")

    target_modules = []
    for module_name, module in model.named_modules():
        if not module_name.startswith(prefix):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(module_name.endswith(suffix) for suffix in suffixes):
            target_modules.append(module_name)
    if not target_modules:
        scope = "decoder" if decoder_only else "model"
        raise RuntimeError(f"No Cohere LoRA target modules found in {scope} scope")
    return sorted(set(target_modules))


if FREEZE_ENCODER:
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    elif hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        raise RuntimeError("Could not find encoder module to freeze.")
    print("Encoder frozen - only decoder will be trained")

if USE_LORA:
    lora_target_modules = get_cohere_lora_target_modules(
        model, decoder_only=FREEZE_ENCODER
    )
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
    try:
        model.enable_input_require_grads()
    except NotImplementedError:
        # Remote-code Cohere implementation may not expose get_input_embeddings.
        pass

    if FREEZE_ENCODER:
        for param_name, param in model.named_parameters():
            if ".encoder." in param_name:
                param.requires_grad = False
        encoder_trainable = [
            n for n, p in model.named_parameters() if p.requires_grad and ".encoder." in n
        ]
        if encoder_trainable:
            raise RuntimeError(
                "Strict decoder-only mode failed: encoder parameters are still trainable"
            )
        print("Strict decoder-only mode enabled - no trainable encoder parameters")
    model.print_trainable_parameters()


def _actual_feat_len(feat: torch.Tensor) -> int:
    """Return the number of non-padding frames in a mel spectrogram [n_mels, time]."""
    nonzero = feat.any(dim=0)
    if not nonzero.any():
        return 1
    return int(nonzero.nonzero()[-1].item()) + 1


def _coerce_feature_length(value: Any, feat: torch.Tensor) -> int:
    """Coerce serialized feature length to a safe int, falling back to feature inspection."""
    if value is None:
        return _actual_feat_len(feat)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return _actual_feat_len(feat)
    return max(1, min(int(value), feat.shape[-1]))


@dataclass
class CohereASRDataCollator:
    tokenizer: Any
    decoder_prompt_ids: List[int]

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        feat_tensors = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        feature_lengths = [
            _coerce_feature_length(f.get("length"), feat) for f, feat in zip(features, feat_tensors)
        ]
        max_feat_len = max(feature_lengths)

        padded_feats = []
        for feat, flen in zip(feat_tensors, feature_lengths):
            feat = feat[:, :flen]
            if flen < max_feat_len:
                feat = torch.nn.functional.pad(feat, (0, max_feat_len - flen), value=0.0)
            padded_feats.append(feat)

        label_lists = [list(f["labels"]) for f in features]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Cohere tokenizer must define pad_token_id for batching")

        P = len(self.decoder_prompt_ids)
        if P == 0:
            raise ValueError("decoder_prompt_ids is empty — processor prompt tokenization failed")
        prompt_row = torch.tensor(self.decoder_prompt_ids, dtype=torch.long)

        dec_lens = [P + len(t) - 1 for t in label_lists]
        max_dec = max(dec_lens)
        batch_n = len(features)

        decoder_input_ids = torch.full((batch_n, max_dec), pad_id, dtype=torch.long)
        labels = torch.full((batch_n, max_dec), -100, dtype=torch.long)
        decoder_attention_mask = torch.zeros((batch_n, max_dec), dtype=torch.long)

        for i, t in enumerate(label_lists):
            if not t:
                t = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [0]
            dlen = P + len(t) - 1
            decoder_input_ids[i, :P] = prompt_row
            if len(t) > 1:
                decoder_input_ids[i, P:dlen] = torch.tensor(t[:-1], dtype=torch.long)
            labels[i, : P - 1] = -100
            labels[i, P - 1 : P - 1 + len(t)] = torch.tensor(t, dtype=torch.long)
            decoder_attention_mask[i, :dlen] = 1

        return {
            "input_features": torch.stack(padded_feats),
            "length": torch.tensor(feature_lengths, dtype=torch.long),
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


data_collator = CohereASRDataCollator(
    tokenizer=tokenizer,
    decoder_prompt_ids=DECODER_PROMPT_IDS,
)


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

    indices = random.sample(range(len(preds)), min(50, len(preds)))
    table = wandb.Table(columns=["Reference", "Prediction"])
    for i in indices:
        table.add_data(refs[i], preds[i])
    wandb.log({"examples": table})
    return {"wer": wer, "cer": cer}


num_batches = max(1, len(dataset["train"]) // BATCH_SIZE)
wandb.init(
    project=WANDB_PROJECT,
    config={
        "model": BASE_MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "lora": USE_LORA,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "freeze_encoder": FREEZE_ENCODER,
        "language": LANGUAGE,
    },
)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{RUN_NAME}",
    hub_model_id=RUN_NAME,
    run_name=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1, BATCH_SIZE // 2),
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    label_smoothing_factor=LABEL_SMOOTHING,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS or -1,
    eval_strategy="steps" if MAX_STEPS else "epoch",
    eval_steps=EVAL_STEPS if MAX_STEPS else None,
    save_strategy="steps" if MAX_STEPS else "epoch",
    save_steps=SAVE_STEPS if MAX_STEPS else None,
    predict_with_generate=False,
    generation_max_length=256,
    gradient_checkpointing=False,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=max(1, int(num_batches * 0.1)),
    report_to=["wandb"],
    push_to_hub=PUSH_TO_HUB,
    label_names=["labels"],
    remove_unused_columns=False,
    fp16=False,
)

class WerEvalCallback(TrainerCallback):
    """Compute WER/CER via manual generation at the end of each evaluation."""

    def __init__(self, eval_ds, prompt_ids, tok, feat_ext, max_samples=200):
        self.eval_ds = eval_ds
        self.prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        self.prompt_mask = torch.ones(1, len(prompt_ids), dtype=torch.long)
        self.tok = tok
        self.feat_ext = feat_ext
        self.max_samples = max_samples
        sample_count = min(self.max_samples, len(self.eval_ds))
        self.eval_indices = random.Random(42).sample(range(len(self.eval_ds)), sample_count)

    def on_evaluate(self, args, state, control, model, **kwargs):
        from data.text_normalize import normalize_text_for_asr

        model.eval()
        device = next(model.parameters()).device
        prompt = self.prompt_ids.to(device)
        prompt_mask = self.prompt_mask.to(device)

        n = len(self.eval_indices)
        preds, refs = [], []
        for i in self.eval_indices:
            sample = self.eval_ds[i]
            feat = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0)
            flen = _coerce_feature_length(sample.get("length"), feat[0])
            length = torch.tensor([flen], dtype=torch.long)
            with torch.no_grad():
                out = model.generate(
                    input_features=feat.to(device),
                    length=length.to(device),
                    decoder_input_ids=prompt,
                    decoder_attention_mask=prompt_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=4,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )
            hyp = self.tok.decode(out[0], skip_special_tokens=True).strip()
            ref_ids = [t for t in sample["labels"] if t != -100]
            ref = self.tok.decode(ref_ids, skip_special_tokens=True).strip()
            preds.append(normalize_text_for_asr(hyp))
            refs.append(normalize_text_for_asr(ref))

        wer_val = 100 * wer_metric.compute(predictions=preds, references=refs)
        cer_val = 100 * cer_metric.compute(predictions=preds, references=refs)
        print(f"\n>>> WER: {wer_val:.2f}%  CER: {cer_val:.2f}%  (on {n} samples)")
        wandb.log({"eval_wer": wer_val, "eval_cer": cer_val, "trainer/global_step": state.global_step})

        table = wandb.Table(columns=["Reference", "Prediction"])
        for i in random.sample(range(len(preds)), min(50, len(preds))):
            table.add_data(refs[i], preds[i])
        wandb.log({"eval_examples": table})


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=None,
    processing_class=None,
    callbacks=[
        WerEvalCallback(
            eval_dataset,
            DECODER_PROMPT_IDS,
            tokenizer,
            feature_extractor,
            max_samples=WER_MAX_SAMPLES,
        )
    ],
)

print("Calculating baseline metrics...")
baseline_metrics = trainer.evaluate()
print(f"Baseline: {baseline_metrics}")
wandb.log({"baseline": baseline_metrics})

def generate_eval_hypothesis(sample: Dict[str, Any], generation_model) -> str:
    generation_model.eval()
    prompt = torch.tensor(DECODER_PROMPT_IDS, dtype=torch.long).unsqueeze(0)
    prompt_mask = torch.ones_like(prompt)
    feat = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0)
    flen = _coerce_feature_length(sample.get("length"), feat[0])
    length = torch.tensor([flen], dtype=torch.long)
    with torch.no_grad():
        out = generation_model.generate(
            input_features=feat.to(generation_model.device),
            length=length.to(generation_model.device),
            decoder_input_ids=prompt.to(generation_model.device),
            decoder_attention_mask=prompt_mask.to(generation_model.device),
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def repetition_score(text: str, ngram_size: int = 4) -> int:
    words = text.lower().split()
    if len(words) < ngram_size:
        return 0
    counts: Dict[tuple[str, ...], int] = {}
    for idx in range(len(words) - ngram_size + 1):
        ngram = tuple(words[idx : idx + ngram_size])
        counts[ngram] = counts.get(ngram, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def preview_transcriptions(label: str, generation_model, num_samples: int = 3):
    print(f"{label} preview transcriptions:")
    for i in range(min(num_samples, len(eval_dataset))):
        sample = eval_dataset[i]
        hyp = generate_eval_hypothesis(sample, generation_model)
        ref_ids = [t for t in sample["labels"] if t != -100]
        ref = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()
        print(f"[{i}] REF: {ref}")
        print(f"[{i}] HYP: {hyp}")
        print(f"[{i}] repetition_4gram: {repetition_score(hyp)}")


preview_transcriptions("Baseline", model)

trainer.train()

preview_transcriptions("Final", model)

if PUSH_TO_HUB:
    trainer.push_to_hub(
        commit_message="Training complete",
        language="pl",
        finetuned_from=BASE_MODEL_NAME,
        tasks="automatic-speech-recognition",
    )

print("Done!")
