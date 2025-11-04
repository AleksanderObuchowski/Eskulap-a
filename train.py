from datasets import load_dataset, Audio, load_from_disk
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
import torch
import wandb

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

BASE_MODEL_NAME = "openai/whisper-small"
USE_LORA = False  # Set to True to use LoRA adapters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 3

RUN_NAME = BASE_MODEL_NAME.split("/")[-1] + "-med-pl-30s"
if USE_LORA:
    RUN_NAME += "-lora"

metric = evaluate.load("wer")
dataset = load_from_disk("prepared_data")
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_NAME)

tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_NAME, language="Polish", task="transcribe")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language="Polish", task="transcribe")


input_str = dataset["train"][0]["text"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


dataset = dataset.cast_column("path", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["path"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    batch["decoded"] = tokenizer.decode(batch["labels"], skip_special_tokens=False)
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
print(dataset)

model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

model.generation_config.language = "polish"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

if USE_LORA:
    from peft import get_peft_model, LoraConfig

    config = LoraConfig(
        r=64,  # Rank of LoRA decomposition
        lora_alpha=64,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention projections
        lora_dropout=0,  # Dropout applied to LoRA layers
        bias="none",  # Don't adapt bias terms
        task_type=None
    )

    # Wrap the base model with LoRA using the above config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Print which parameters are trainable


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Seq2SeqTrainingArguments

no_of_batches = len(dataset["train"]) // BATCH_SIZE

wandb.init(project="eskulap-a")  # Initialize WandB

training_args = Seq2SeqTrainingArguments(
    run_name=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    warmup_ratio=0.1,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=int(no_of_batches * 0.5),  # Save every 50% of an epoch
    eval_steps=int(no_of_batches * 0.5),
    logging_steps=int(no_of_batches * 0.1),
    save_total_limit=5,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()
kwargs = {
    "language": "pl",
    "model_name": RUN_NAME,  # a 'pretty' name for your model
    "finetuned_from": BASE_MODEL_NAME,
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(**kwargs)