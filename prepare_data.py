from datasets import Audio, concatenate_datasets, load_dataset

MAX_DURATION_SECONDS = 30.0


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
    return duration_seconds <= MAX_DURATION_SECONDS


youtube_dataset = load_dataset("lion-ai/youtube_asr_30", split="train")
gemini_dataset = load_dataset("lion-ai/pl_med_asr_test2", split="train")
youtube_dataset = youtube_dataset.remove_columns(["id"])  # Remove unused columns if any
gemini_dataset = gemini_dataset.rename_column("file_name", "path")
gemini_dataset = gemini_dataset.remove_columns(
    ["prompt"]
)  # Remove unused columns if any


youtube_dataset = youtube_dataset.cast_column("path", Audio(sampling_rate=16000))
gemini_dataset = gemini_dataset.cast_column("path", Audio(sampling_rate=16000))
combined_dataset = concatenate_datasets([youtube_dataset, gemini_dataset])
print("Total number of examples before filtering:", len(combined_dataset))
combined_dataset = combined_dataset.filter(
    filter_by_duration, load_from_cache_file=False, num_proc=4
)
print("Total number of examples after filtering:", len(combined_dataset))
combined_dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)
combined_dataset.save_to_disk("prepared_data")
