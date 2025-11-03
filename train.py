from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bf16_supported
import wandb

MODEL_NAME = "whisper-large-v3-turbo-lora-pl-med-asr-45s"

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/whisper-large-v3-turbo",
    dtype = None, # Leave as None for auto detection
    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "Polish",
    whisper_task = "transcribe",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = None, # ** MUST set this for Whisper **
)

import numpy as np
import tqdm
import random



def filter_by_duration(example):
    try:
        audio = example["path"]
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
    except (KeyError, TypeError):
        return False
    if array is None or sampling_rate is None or sampling_rate <= 0:
        return False
    duration_seconds = len(array) / sampling_rate
    return duration_seconds < MAX_DURATION_SECONDS

#Set this to the language you want to train on
model.generation_config.language = "<|pl|>"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="pl", task="transcribe"
)
from transformers import WhisperConfig
default_cfg = WhisperConfig.from_pretrained("unsloth/whisper-large-v3-turbo")
model.generation_config.suppress_tokens = default_cfg.suppress_tokens
model.generation_config.begin_suppress_tokens = default_cfg.begin_suppress_tokens

def formatting_prompts_func(example):
    try:
        audio_arrays = example['path']['array']
        sampling_rate = example["path"]["sampling_rate"]
        features = tokenizer.feature_extractor(
            audio_arrays, sampling_rate=sampling_rate
        )
        tokenized_text = tokenizer.tokenizer(example["text"])
        return {
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        }
    except Exception as e:
        return {
            "input_features": None,
            "labels": None,
        }
dataset = load_dataset("prepared_data")

train_dataset =  [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['train'], desc="Processing train dataset")]
test_dataset =  [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['test'], desc="Processing test dataset")]

# @title Create compute_metrics and datacollator
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pdb

metric = evaluate.load("wer")
def compute_metrics(pred):
    # When predict_with_generate=True, predictions are already token IDs
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Calculate overall WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    # Log random samples with predictions
    num_samples_to_log = min(10, len(pred_str))
    if num_samples_to_log > 0:
        # Get random indices
        sample_indices = random.sample(range(len(pred_str)), num_samples_to_log)
        
        # Create a table for WandB
        sample_data = []
        for idx in sample_indices:
            # Calculate WER for individual sample
            sample_wer = 100 * metric.compute(
                predictions=[pred_str[idx]],
                references=[label_str[idx]]
            )
            sample_data.append([
                idx,
                label_str[idx],
                pred_str[idx],
                round(sample_wer, 2)
            ])
        
        # Log to WandB as a table
        wandb.log({
            "sample_predictions": wandb.Table(
                columns=["Sample Index", "Reference", "Prediction", "WER (%)"],
                data=sample_data
            )
        })
    
    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

wandb.init(project="eskulap-a")  # Initialize WandB

trainer = Seq2SeqTrainer(
    model = model,
    train_dataset = train_dataset,
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
    eval_dataset = test_dataset,
    tokenizer = tokenizer.feature_extractor,
    compute_metrics=compute_metrics,
    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 1,
        warmup_ratio=0.1,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps = 120,
        learning_rate = 1e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        weight_decay = 0.01,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = 50,
        eval_strategy="steps",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use TrackIO/WandB etc
        run_name = MODEL_NAME,
        load_best_model_at_end = True,
        metric_for_best_model="wer",
        generation_max_length=225,          # typical for Whisper
        generation_num_beams=1,
    ),
)# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained(MODEL_NAME)  # Local saving
tokenizer.save_pretrained(MODEL_NAME)
model.push_to_hub_merged(f"lion-ai/{MODEL_NAME}", tokenizer, save_method = "merged_16bit")  # Push LoRA merged model to hub