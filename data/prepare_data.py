from config import (
    admed_anoni_dataset,
    admed_human_dataset,
    gemini_dataset,
    prepared_data_dir,
    youtube_dataset,
)
from datasets import Audio, DatasetDict, concatenate_datasets
from dotenv import load_dotenv

load_dotenv()
MAX_DURATION_SECONDS = 30.0


def filter_by_duration(example):
    try:
        audio = example["audio"]
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
    except (KeyError, TypeError):
        return False
    if array is None or sampling_rate is None or sampling_rate <= 0:
        return False
    duration_seconds = len(array) / sampling_rate
    return duration_seconds <= MAX_DURATION_SECONDS


# Step 1: Set apart 3k examples from admed_anoni_dataset for test
print("Setting apart test examples...")
admed_anoni_split = admed_anoni_dataset.train_test_split(test_size=3000, seed=42)
admed_anoni_test = admed_anoni_split["test"]
admed_anoni_remaining = admed_anoni_split["train"]
print(
    f"Admed anoni - Test: {len(admed_anoni_test)}, Remaining: {len(admed_anoni_remaining)}"
)

# Step 2: Set apart 2k examples from admed_human_dataset for test
admed_human_split = admed_human_dataset.train_test_split(test_size=2000, seed=42)
admed_human_test = admed_human_split["test"]
admed_human_remaining = admed_human_split["train"]
print(
    f"Admed human - Test: {len(admed_human_test)}, Remaining: {len(admed_human_remaining)}"
)

# Step 3: Create combined test dataset
test_dataset = concatenate_datasets([admed_anoni_test, admed_human_test])
print(f"Total test examples: {len(test_dataset)}")

# Step 4: Prepare all datasets for merging (normalize columns and cast audio)
print("\nPreparing datasets for merging...")
youtube_dataset = youtube_dataset.remove_columns(["id"])
youtube_dataset = youtube_dataset.cast_column("audio", Audio(sampling_rate=16000))
youtube_dataset = youtube_dataset.rename_column("sentence", "text")

gemini_dataset = gemini_dataset.rename_column("file_name", "audio")
gemini_dataset = gemini_dataset.remove_columns(["prompt"])
gemini_dataset = gemini_dataset.cast_column("audio", Audio(sampling_rate=16000))

admed_anoni_remaining = admed_anoni_remaining.cast_column(
    "audio", Audio(sampling_rate=16000)
)
admed_human_remaining = admed_human_remaining.cast_column(
    "audio", Audio(sampling_rate=16000)
)

test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))


# Step 5: Merge all training datasets
print("\nMerging all training datasets...")
combined_dataset = concatenate_datasets(
    [
        youtube_dataset,
        gemini_dataset,
        admed_anoni_remaining,
        admed_human_remaining,
    ]
)
print(f"Total training examples before filtering: {len(combined_dataset)}")

# Step 6: Filter by duration
print(f"\nFiltering by duration (max {MAX_DURATION_SECONDS}s)...")
combined_dataset = combined_dataset.filter(
    filter_by_duration, load_from_cache_file=False, num_proc=4
)
test_dataset = test_dataset.filter(
    filter_by_duration, load_from_cache_file=False, num_proc=4
)
print(f"Training examples after filtering: {len(combined_dataset)}")
print(f"Test examples after filtering: {len(test_dataset)}")

# Step 7: Create train/dev split (0.1 = 10% dev)
print("\nCreating train/dev split...")
train_dev_split = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_dev_split["train"]
dev_dataset = train_dev_split["test"]

print(f"Train examples: {len(train_dataset)}")
print(f"Dev examples: {len(dev_dataset)}")
print(f"Test examples: {len(test_dataset)}")

# Step 8: Save dataset with all splits
final_dataset = DatasetDict(
    {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
)

print(f"\nSaving dataset to {prepared_data_dir}...")
final_dataset.save_to_disk(prepared_data_dir)
print("Dataset saved successfully!")
print(f"\nFinal dataset structure:")
print(f"  - train: {len(final_dataset['train'])} examples")
print(f"  - dev: {len(final_dataset['dev'])} examples")
print(f"  - test: {len(final_dataset['test'])} examples")
