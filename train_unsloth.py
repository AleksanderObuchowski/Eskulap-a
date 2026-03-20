"""Legacy Unsloth Whisper training path.

Deprecated: uses hardcoded ``prepared_data`` instead of ``data.load_train_data`` /
``configs/experiment.yaml``. Prefer ``train.py`` (Whisper) or align this script with
the unified pipeline in ``docs/data_pipeline.md`` before relying on it for new work.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
import wandb
from datasets import Audio, load_from_disk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration
from unsloth import FastModel, is_bf16_supported

# Configuration
BASE_MODEL_NAME = "unsloth/whisper-large-v3-turbo"
USE_4BIT = False
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 2

# Setup
RUN_NAME = "whisper-large-v3-turbo-lora-pl-med-asr"

# Processed data cache path
PROCESSED_DATA_PATH = f"processed_data_{BASE_MODEL_NAME.split('/')[-1]}"

# Load model and tokenizer with Unsloth
model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    dtype=None,
    load_in_4bit=USE_4BIT,
    auto_model=WhisperForConditionalGeneration,
    whisper_language="Polish",
    whisper_task="transcribe",
)

# Apply LoRA with Unsloth
model = FastModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    task_type="SEQ_2_SEQ_LM",
)

# Configure generation settings
model.generation_config.language = "<|pl|>"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
    language="pl", task="transcribe"
)

# Load or process dataset with caching
if os.path.exists(PROCESSED_DATA_PATH):
    print(f"Loading cached processed dataset from {PROCESSED_DATA_PATH}")
    dataset = load_from_disk(PROCESSED_DATA_PATH)
else:
    print("Processing dataset (this will be cached for future runs)...")
    dataset = load_from_disk("prepared_data")
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["path"]
        features = tokenizer.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        )
        batch["input_features"] = features.input_features[0]
        batch["labels"] = tokenizer.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=32
    )
    dataset.save_to_disk(PROCESSED_DATA_PATH)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

# Metrics computation
wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# Initialize data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)

# Calculate training steps
num_batches = len(dataset["train"]) // BATCH_SIZE

# Initialize WandB
wandb.init(project="eskulap-a")

# Training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{RUN_NAME}",
    run_name=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=1,
    save_steps=int(num_batches * 0.5),
    eval_steps=int(num_batches * 0.5),
    logging_steps=int(num_batches * 0.1) or 1,
    save_total_limit=5,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    seed=3407,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer.feature_extractor,
)

# Show memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save model
model.save_pretrained(RUN_NAME)
tokenizer.save_pretrained(RUN_NAME)
model.push_to_hub_merged(f"lion-ai/{RUN_NAME}", tokenizer, save_method="merged_16bit")
