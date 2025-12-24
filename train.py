import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import wandb
from data.config import prepared_data_dir, processed_data_dir

# Configuration
BASE_MODEL_NAME = "openai/whisper-large-v3"
USE_LORA = True
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 10

# Setups
RUN_NAME = BASE_MODEL_NAME.split("/")[-1] + "-med-pl"
if USE_LORA:
    RUN_NAME += "-lora"

# Processed data cache path (includes model name to handle different feature extractors)
PROCESSED_DATA_PATH = os.path.join(
    processed_data_dir, f"processed_data_{BASE_MODEL_NAME.split('/')[-1]}"
)

# Load tokenizer/processor (needed for both loading and processing)
tokenizer = WhisperTokenizer.from_pretrained(
    BASE_MODEL_NAME, language="Polish", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL_NAME, language="Polish", task="transcribe"
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    BASE_MODEL_NAME, language="Polish", task="transcribe"
)

# Load or process dataset with caching
if os.path.exists(PROCESSED_DATA_PATH):
    print(f"Loading cached processed dataset from {PROCESSED_DATA_PATH}")
    dataset = load_from_disk(PROCESSED_DATA_PATH)
else:
    print("Processing dataset (this will be cached for future runs)...")
    dataset = load_from_disk(prepared_data_dir)

    def prepare_dataset(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]

        batch["input_features"] = feature_extractor(
            audio_arrays, sampling_rate=16000
        ).input_features
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        batched=True,
        batch_size=50,
        num_proc=8,
    )
    dataset.save_to_disk(PROCESSED_DATA_PATH)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

# Load and configure model
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
model.generation_config.language = "polish"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Optional LoRA setup
if USE_LORA:
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=64,s
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # Required for LoRA + gradient checkpointing
    model.print_trainable_parameters()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Process input features
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Process labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Handle padding and special tokens
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# Initialize data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Metrics computation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Handle padding tokens
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    # Log 50 random examples to wandb
    num_examples = min(50, len(pred_str))
    indices = random.sample(range(len(pred_str)), num_examples)
    examples_table = wandb.Table(columns=["Reference", "Prediction"])
    for idx in indices:
        examples_table.add_data(label_str[idx], pred_str[idx])
    wandb.log({"eval_examples": examples_table})

    return {"wer": wer, "cer": cer}


# Calculate training steps
num_batches = len(dataset["train"]) // BATCH_SIZE

# Initialize WandB with training config
wandb_config = {
    "base_model": BASE_MODEL_NAME,
    "use_lora": USE_LORA,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
}
if USE_LORA:
    wandb_config["lora_r"] = lora_config.r
    wandb_config["lora_alpha"] = lora_config.lora_alpha
    wandb_config["lora_dropout"] = lora_config.lora_dropout
    wandb_config["lora_target_modules"] = lora_config.target_modules
    wandb_config["lora_bias"] = lora_config.bias

wandb.init(project="eskulap-a", config=wandb_config)

# Training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{RUN_NAME}",
    hub_model_id=RUN_NAME,
    run_name=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE // 2,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=int(num_batches * 0.5),
    eval_steps=int(num_batches * 0.5),
    logging_steps=int(num_batches * 0.1),
    save_total_limit=5,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    label_names=["labels"],  # Required for PEFT to identify label columns correctly
    remove_unused_columns=False,  # Prevent trainer from removing input_features
)


# Initialize logging callback (removed - examples now logged in compute_metrics)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# Train and push to hub

# Calculate baseline metrics before training
print("Calculating baseline metrics...")
baseline_metrics = trainer.evaluate()
print(f"Baseline metrics: {baseline_metrics}")
wandb.log({"baseline": baseline_metrics})

trainer.train()

model_card_kwargs = {
    "language": "pl",
    "finetuned_from": BASE_MODEL_NAME,
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(commit_message="Training complete", **model_card_kwargs)
