"""Create clean train/test splits for ASR datasets.

Creates fixed splits with NO text overlap between train and test,
suitable for publishing as a public benchmark on HuggingFace.

For datasets that share texts (admed_anoni/admed_human), splits by UNIQUE TEXTS
first to ensure no text leakage between train and test.

Usage:
    # Create splits for admed datasets (default)
    python -m data.create_clean_splits

    # Create splits for specific datasets
    python -m data.create_clean_splits --datasets youtube gemini bigos

    # Create and push to HuggingFace
    python -m data.create_clean_splits --push --repo-id lion-ai/admed_voice_clean
"""

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

from .quality_filter import FilterConfig, filter_dataset

# Default paths
DEFAULT_OUTPUT_DIR = os.environ.get(
    "CLEAN_SPLITS_DIR", "/mnt/data/Eskulap-a/clean_splits"
)
HF_HOME = os.environ.get("HF_HOME", None)

SEED = 42

# Dataset registry with HuggingFace paths and configuration
DATASET_CONFIGS = {
    "admed_anoni": {
        "hf_path": "lion-ai/admed_voice",
        "hf_split": "anoni",
        "local_name": "anoni",
        "needs_quality_filter": True,
        "has_existing_test": False,
        "max_train_samples": None,
        "max_test_samples": 1000,
    },
    "admed_human": {
        "hf_path": "lion-ai/admed_voice",
        "hf_split": "human",
        "local_name": "human",
        "needs_quality_filter": True,
        "has_existing_test": False,
        "max_train_samples": None,
        "max_test_samples": 1000,
    },
    "youtube": {
        "hf_path": "lion-ai/youtube_asr_30",
        "hf_split": "train",
        "local_name": "youtube",
        "text_column": "sentence",
        "needs_quality_filter": True,
        "has_existing_test": False,
        "max_train_samples": None,
        "max_test_samples": 500,
    },
    "gemini": {
        "hf_path": "lion-ai/pl_med_asr_test2",
        "hf_split": "train",
        "local_name": "gemini",
        "needs_quality_filter": True,
        "has_existing_test": False,
        "max_train_samples": None,
        "max_test_samples": 500,
        # max_duration_seconds: uses default 30.0s from SplitConfig
    },
    "bigos": {
        "hf_path": "lion-ai/bigos",
        "hf_split": "train",
        "hf_test_split": "validation",  # bigos has existing validation split
        "local_name": "bigos",
        "text_column": "sentence",
        "needs_quality_filter": False,  # already clean
        "has_existing_test": True,
        "max_train_samples": 10000,
        "max_test_samples": 800,
    },
}

# Datasets that share texts and need joint splitting
SHARED_TEXT_GROUPS = [
    ["admed_anoni", "admed_human"],  # These share the same transcriptions
]


@dataclass
class SplitConfig:
    """Configuration for creating train/test splits."""

    # Target number of unique texts in test set (for text-based splitting)
    test_unique_texts_target: int = 300

    # Test ratio for datasets without shared texts
    test_ratio: float = 0.1

    # Quality filter settings for training data
    filter_config: FilterConfig = None

    # Audio settings
    target_sample_rate: int = 16000

    # Max audio duration in seconds (None = no limit)
    max_duration_seconds: float | None = 30.0

    def __post_init__(self):
        if self.filter_config is None:
            self.filter_config = FilterConfig(
                min_words=3,
                deduplicate_text=True,
                max_text_duplicates=4,
                deduplicate_audio=True,
                max_audio_duplicates=1,
                use_similarity_filter=False,
                similarity_threshold=0.95,
                num_perm=128,
            )


def get_cache_dir():
    """Get HuggingFace cache directory."""
    return os.path.join(HF_HOME, "datasets") if HF_HOME else None


def load_raw_dataset(name: str, max_samples: int | None = None) -> Dataset:
    """Load a raw dataset from HuggingFace.

    Args:
        name: Dataset name from DATASET_CONFIGS
        max_samples: If set, truncate to first N samples (for dry-run)

    Returns:
        Dataset
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}")

    config = DATASET_CONFIGS[name]
    cache_dir = get_cache_dir()

    print(f"  Loading {name} from {config['hf_path']}...")
    ds = load_dataset(config["hf_path"], split=config["hf_split"], cache_dir=cache_dir)

    # Normalize text column name to "text"
    text_col = config.get("text_column", "text")
    if text_col != "text" and text_col in ds.column_names:
        ds = ds.rename_column(text_col, "text")

    # Normalize audio column (some datasets store audio in "file_name" or similar)
    if "audio" not in ds.column_names:
        for col in ("file_name", "path", "audio_path"):
            if col in ds.column_names and isinstance(ds.features.get(col), Audio):
                ds = ds.rename_column(col, "audio")
                print(f"    Renamed '{col}' -> 'audio'")
                break
    elif "file_name" in ds.column_names and isinstance(ds.features.get("file_name"), Audio):
        # Both exist — check if "audio" is broken (None) while file_name has data
        if ds[0]["audio"] is None and ds[0]["file_name"] is not None:
            ds = ds.remove_columns("audio")
            ds = ds.rename_column("file_name", "audio")
            print(f"    Fixed audio column (was in 'file_name')")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"    {len(ds)} samples")
    return ds


def load_existing_test_split(
    name: str, max_samples: int | None = None
) -> Dataset | None:
    """Load existing test split if available.

    Args:
        name: Dataset name from DATASET_CONFIGS

    Returns:
        Tuple of (anoni_text_to_indices, human_text_to_indices) dicts
    """
    config = DATASET_CONFIGS[name]
    if not config.get("has_existing_test"):
        return None

    cache_dir = get_cache_dir()
    test_split = config.get("hf_test_split", "test")

    print(f"  Loading existing test split for {name}...")
    ds = load_dataset(config["hf_path"], split=test_split, cache_dir=cache_dir)

    # Normalize text column name to "text"
    text_col = config.get("text_column", "text")
    if text_col != "text" and text_col in ds.column_names:
        ds = ds.rename_column(text_col, "text")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"    {len(ds)} test samples")
    return ds


def collect_unique_texts(
    datasets: dict[str, Dataset],
) -> dict[str, dict[str, list[int]]]:
    """Collect unique texts and their sample indices from datasets.

    Args:
        datasets: Dict mapping dataset name to Dataset

    Returns:
        Dict mapping dataset name to {text: [indices]} dict
    """
    print("\nCollecting unique texts...")

    result = {}
    for name, ds in datasets.items():
        text_col = _get_text_column(ds)
        text_to_indices = defaultdict(list)
        for idx, text in enumerate(ds[text_col]):
            text_to_indices[text].append(idx)
        result[name] = dict(text_to_indices)
        print(f"  {name}: {len(text_to_indices)} unique texts from {len(ds)} samples")

    return result


def select_test_texts_for_group(
    datasets: dict[str, Dataset],
    text_indices: dict[str, dict[str, list[int]]],
    config: SplitConfig,
) -> set[str]:
    """Select test texts for a group of datasets that share texts.

    Args:
        datasets: Dict of datasets in the group
        text_indices: Text to indices mapping for each dataset
        config: Split configuration

    Returns:
        Set of texts designated for test split
    """
    print(
        f"\nSelecting test texts (target: ~{config.test_unique_texts_target} unique texts)..."
    )

    random.seed(SEED)

    # Get all unique texts across the group
    all_texts = set()
    for indices in text_indices.values():
        all_texts.update(indices.keys())

    # Find shared texts (appear in multiple datasets)
    text_counts = defaultdict(int)
    for indices in text_indices.values():
        for text in indices:
            text_counts[text] += 1

    shared_texts = {t for t, c in text_counts.items() if c > 1}
    unique_texts = all_texts - shared_texts

    print(f"  Total unique texts: {len(all_texts)}")
    print(f"  Shared across datasets: {len(shared_texts)}")

    test_texts = set()

    # Prioritize shared texts for cross-dataset evaluation
    shared_list = list(shared_texts)
    random.shuffle(shared_list)

    for text in shared_list:
        if len(test_texts) >= config.test_unique_texts_target:
            break
        test_texts.add(text)

    # Add unique texts if needed
    if len(test_texts) < config.test_unique_texts_target:
        unique_list = list(unique_texts)
        random.shuffle(unique_list)
        for text in unique_list:
            if len(test_texts) >= config.test_unique_texts_target:
                break
            test_texts.add(text)

    print(f"  Selected {len(test_texts)} unique texts for test set")
    return test_texts


_UNSET = object()


def create_split_for_dataset(
    dataset: Dataset,
    text_to_indices: dict[str, list[int]],
    test_texts: set[str],
    name: str,
    config: SplitConfig,
    apply_quality_filter: bool = True,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    max_duration_seconds: float | None = _UNSET,
) -> DatasetDict:
    """Create train/test split for a single dataset.

    Args:
        dataset: The full dataset
        text_to_indices: Mapping from text to sample indices
        test_texts: Set of texts designated for test split
        name: Dataset name for logging
        config: Split configuration
        apply_quality_filter: Whether to apply quality filters to train
        max_train_samples: Max train samples for this dataset (None = no limit)
        max_test_samples: Max test samples for this dataset (None = no limit)

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    # Resolve per-dataset duration override
    if max_duration_seconds is _UNSET:
        max_duration_seconds = config.max_duration_seconds

    print(f"\nCreating splits for {name}...")

    # Separate indices into train and test based on text
    train_indices = []
    test_indices = []

    for text, indices in text_to_indices.items():
        if text in test_texts:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    print(f"  Raw split: {len(train_indices)} train, {len(test_indices)} test")

    # Create test split
    test_dataset = dataset.select(test_indices) if test_indices else None

    if test_dataset:
        test_dataset = test_dataset.cast_column(
            "audio", Audio(sampling_rate=config.target_sample_rate)
        )

        # Apply duration filter to test set
        if max_duration_seconds:
            before = len(test_dataset)
            test_dataset = test_dataset.filter(
                lambda x: x["audio"] is not None
                and len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
                <= max_duration_seconds
            )
            print(
                f"  Test after duration filter: {len(test_dataset)} (removed {before - len(test_dataset)})"
            )

        # Subsample test set if needed
        if max_test_samples and len(test_dataset) > max_test_samples:
            before = len(test_dataset)
            test_dataset = test_dataset.shuffle(seed=SEED).select(
                range(max_test_samples)
            )
            print(
                f"  Test after subsampling: {len(test_dataset)} (removed {before - len(test_dataset)})"
            )

    # Create train split
    train_dataset = dataset.select(train_indices)
    train_dataset = train_dataset.cast_column(
        "audio", Audio(sampling_rate=config.target_sample_rate)
    )

    # Apply duration filter to train set
    if max_duration_seconds:
        before = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: x["audio"] is not None
            and len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
            <= max_duration_seconds
        )
        print(
            f"  Train after duration filter: {len(train_dataset)} (removed {before - len(train_dataset)})"
        )

    # Apply quality filtering to train set only
    if apply_quality_filter:
        print(f"  Applying quality filters to train split...")
        train_dataset = filter_dataset(train_dataset, config=config.filter_config)

    # Subsample train set if needed
    if max_train_samples and len(train_dataset) > max_train_samples:
        before = len(train_dataset)
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(max_train_samples)
        )
        print(
            f"  Train after subsampling: {len(train_dataset)} (removed {before - len(train_dataset)})"
        )

    print(
        f"  Final: {len(train_dataset)} train, {len(test_dataset) if test_dataset else 0} test"
    )

    return (
        DatasetDict(
            {
                "train": train_dataset,
                "test": test_dataset,
            }
        )
        if test_dataset
        else DatasetDict({"train": train_dataset})
    )


def create_simple_split(
    dataset: Dataset,
    name: str,
    config: SplitConfig,
    apply_quality_filter: bool = True,
    existing_test: Dataset | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    max_duration_seconds: float | None = _UNSET,
) -> DatasetDict:
    """Create a simple train/test split for a standalone dataset.

    Args:
        dataset: The full dataset
        name: Dataset name for logging
        config: Split configuration
        apply_quality_filter: Whether to apply quality filters
        existing_test: Pre-existing test split (e.g., bigos validation)
        max_train_samples: Max train samples for this dataset (None = no limit)
        max_test_samples: Max test samples for this dataset (None = no limit)

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    # Resolve per-dataset duration override
    if max_duration_seconds is _UNSET:
        max_duration_seconds = config.max_duration_seconds

    print(f"\nCreating splits for {name}...")

    # Normalize audio
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=config.target_sample_rate)
    )

    if existing_test is not None:
        # Use existing test split, removing test texts from train
        test_dataset = existing_test.cast_column(
            "audio", Audio(sampling_rate=config.target_sample_rate)
        )
        text_col = _get_text_column(test_dataset)
        test_texts = set(test_dataset[text_col])
        train_text_col = _get_text_column(dataset)
        train_dataset = dataset.filter(lambda x: x[train_text_col] not in test_texts)
        print(
            f"  Using existing test split: {len(test_dataset)} samples"
            f" (removed {len(dataset) - len(train_dataset)} overlapping from train)"
        )
    else:
        # Create test split by random sampling
        test_size = min(
            int(len(dataset) * config.test_ratio),
            max_test_samples or int(len(dataset) * config.test_ratio),
        )
        split = dataset.train_test_split(test_size=test_size, seed=SEED)
        train_dataset = split["train"]
        test_dataset = split["test"]
        print(f"  Split: {len(train_dataset)} train, {len(test_dataset)} test")

    # Apply duration filter
    if max_duration_seconds:
        before_train = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: x["audio"] is not None
            and len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
            <= max_duration_seconds
        )
        print(
            f"  Train after duration filter: {len(train_dataset)} (removed {before_train - len(train_dataset)})"
        )

        before_test = len(test_dataset)
        test_dataset = test_dataset.filter(
            lambda x: x["audio"] is not None
            and len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
            <= max_duration_seconds
        )
        print(
            f"  Test after duration filter: {len(test_dataset)} (removed {before_test - len(test_dataset)})"
        )

    # Subsample test if needed
    if max_test_samples and len(test_dataset) > max_test_samples:
        before = len(test_dataset)
        test_dataset = test_dataset.shuffle(seed=SEED).select(range(max_test_samples))
        print(
            f"  Test after subsampling: {len(test_dataset)} (removed {before - len(test_dataset)})"
        )

    # Apply quality filtering to train
    if apply_quality_filter:
        print(f"  Applying quality filters to train split...")
        train_dataset = filter_dataset(train_dataset, config=config.filter_config)

    # Subsample train set if needed
    if max_train_samples and len(train_dataset) > max_train_samples:
        before = len(train_dataset)
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(max_train_samples)
        )
        print(
            f"  Train after subsampling: {len(train_dataset)} (removed {before - len(train_dataset)})"
        )

    print(f"  Final: {len(train_dataset)} train, {len(test_dataset)} test")

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def _get_text_column(dataset: Dataset) -> str:
    """Detect the text column name in a dataset."""
    for col in ("text", "sentence", "transcription"):
        if col in dataset.column_names:
            return col
    raise ValueError(f"No text column found. Columns: {dataset.column_names}")


def validate_no_text_overlap(splits: dict[str, DatasetDict]) -> bool:
    """Validate that there's no text overlap between train and test."""
    print("\nValidating no text overlap...")

    all_train_texts = set()
    all_test_texts = set()

    for name, dataset_dict in splits.items():
        if "test" not in dataset_dict:
            print(f"  {name}: no test split, skipping validation")
            continue

        text_col = _get_text_column(dataset_dict["train"])
        ds_train_texts = set(dataset_dict["train"][text_col])
        ds_test_texts = set(dataset_dict["test"][text_col])

        # Check within-dataset overlap
        overlap = ds_train_texts & ds_test_texts
        if overlap:
            print(f"  ERROR: {name} has {len(overlap)} overlapping texts!")
            return False
        print(f"  ✓ {name}: no within-dataset overlap")

        all_train_texts.update(ds_train_texts)
        all_test_texts.update(ds_test_texts)

    # Check cross-dataset overlap
    cross_overlap = all_train_texts & all_test_texts
    if cross_overlap:
        print(f"  ERROR: {len(cross_overlap)} texts appear in both train and test!")
        return False

    print(f"  ✓ No cross-dataset overlap")
    return True


def save_splits(splits: dict[str, DatasetDict], output_dir: str):
    """Save splits to local disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving splits to {output_path}...")

    for name, dataset_dict in splits.items():
        ds_path = output_path / DATASET_CONFIGS[name]["local_name"]
        dataset_dict.save_to_disk(str(ds_path))
        print(f"  Saved {name} to {ds_path}")

    # Save metadata
    metadata = {
        "seed": SEED,
        "splits": {
            name: {
                "train_samples": len(ds["train"]),
                "test_samples": len(ds["test"]) if "test" in ds else 0,
                "train_unique_texts": len(
                    set(ds["train"][_get_text_column(ds["train"])])
                ),
                "test_unique_texts": len(set(ds["test"][_get_text_column(ds["test"])]))
                if "test" in ds
                else 0,
            }
            for name, ds in splits.items()
        },
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")


def push_to_hub(splits: dict[str, DatasetDict], repo_id: str, private: bool = False):
    """Push splits to HuggingFace Hub."""
    print(f"\nPushing to HuggingFace Hub: {repo_id}...")

    for name, dataset_dict in splits.items():
        config_name = DATASET_CONFIGS[name]["local_name"]
        print(f"  Pushing {name} as config '{config_name}'...")
        dataset_dict.push_to_hub(repo_id, config_name=config_name, private=private)

    print(f"  ✓ Successfully pushed to {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Create clean train/test splits for ASR datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=["admed_anoni", "admed_human", "youtube", "gemini", "bigos"],
        help="Datasets to process (default: admed_anoni admed_human)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save splits (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-texts",
        type=int,
        default=500,
        help="Target number of unique texts in test set (default: 500)",
    )
    parser.add_argument(
        "--max-test-samples",
        nargs="*",
        default=[],
        help="Max test samples per dataset: DATASET=N ... (e.g. youtube=500 bigos=1000)",
    )
    parser.add_argument(
        "--max-train-samples",
        nargs="*",
        default=[],
        help="Max train samples per dataset: DATASET=N ... (e.g. youtube=3000 bigos=5000)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio for standalone datasets (default: 0.1)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        type=int,
        const=50,
        default=None,
        metavar="N",
        help="Quick validation run using first N samples per dataset (default: 50). Skips saving/pushing.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub after creating splits",
    )
    parser.add_argument(
        "--repo-id",
        default="lion-ai/admed_voice_clean",
        help="HuggingFace repo ID for pushing",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repo private",
    )

    args = parser.parse_args()

    # Parse per-dataset sample limits (format: DATASET=N)
    def parse_sample_limits(limit_args: list[str]) -> dict[str, int]:
        limits = {}
        for item in limit_args:
            if "=" not in item:
                parser.error(
                    f"Invalid sample limit format: '{item}'. Use DATASET=N (e.g. youtube=3000)"
                )
            name, value = item.split("=", 1)
            if name not in DATASET_CONFIGS:
                parser.error(
                    f"Unknown dataset: '{name}'. Available: {list(DATASET_CONFIGS.keys())}"
                )
            limits[name] = int(value)
        return limits

    train_limits = parse_sample_limits(args.max_train_samples)
    test_limits = parse_sample_limits(args.max_test_samples)

    # Apply limits to dataset configs
    for name, limit in train_limits.items():
        DATASET_CONFIGS[name]["max_train_samples"] = limit
    for name, limit in test_limits.items():
        DATASET_CONFIGS[name]["max_test_samples"] = limit

    config = SplitConfig(
        test_unique_texts_target=args.test_texts,
        test_ratio=args.test_ratio,
        max_duration_seconds=args.max_duration,
    )

    dry_run = args.dry_run

    print("=" * 60)
    print("CREATE CLEAN TRAIN/TEST SPLITS")
    if dry_run:
        print(f"  *** DRY RUN ({dry_run} samples per dataset) ***")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Test unique texts target: {config.test_unique_texts_target}")
    if train_limits:
        print(f"Max train samples: {train_limits}")
    if test_limits:
        print(f"Max test samples: {test_limits}")
    print(f"Test ratio (standalone): {config.test_ratio}")
    print(f"Max duration: {config.max_duration_seconds}s")
    print(f"Output directory: {args.output_dir}")
    if args.push:
        print(f"Will push to: {args.repo_id}")
    print()

    splits = {}

    # Find which shared groups are being processed
    datasets_to_process = set(args.datasets)

    for group in SHARED_TEXT_GROUPS:
        group_datasets = [d for d in group if d in datasets_to_process]
        if len(group_datasets) > 1:
            # Process as a group (shared text splitting)
            print(f"\nProcessing shared text group: {group_datasets}")

            # Load all datasets in the group
            raw_datasets = {
                name: load_raw_dataset(name, max_samples=dry_run)
                for name in group_datasets
            }

            # Collect unique texts
            text_indices = collect_unique_texts(raw_datasets)

            # Select test texts
            test_texts = select_test_texts_for_group(raw_datasets, text_indices, config)

            # Create splits for each dataset
            for name in group_datasets:
                ds_config = DATASET_CONFIGS[name]
                splits[name] = create_split_for_dataset(
                    raw_datasets[name],
                    text_indices[name],
                    test_texts,
                    name,
                    config,
                    apply_quality_filter=ds_config["needs_quality_filter"],
                    max_train_samples=ds_config.get("max_train_samples"),
                    max_test_samples=ds_config.get("max_test_samples"),
                    max_duration_seconds=ds_config.get("max_duration_seconds", _UNSET),
                )
                datasets_to_process.discard(name)

        elif len(group_datasets) == 1:
            # Single dataset from a group - process standalone
            pass  # Will be handled below

    # Process remaining datasets as standalone
    for name in datasets_to_process:
        ds_config = DATASET_CONFIGS[name]

        print(f"\nProcessing standalone dataset: {name}")
        raw_dataset = load_raw_dataset(name, max_samples=dry_run)

        # Check for existing test split
        existing_test = load_existing_test_split(name, max_samples=dry_run)

        splits[name] = create_simple_split(
            raw_dataset,
            name,
            config,
            apply_quality_filter=ds_config["needs_quality_filter"],
            existing_test=existing_test,
            max_train_samples=ds_config.get("max_train_samples"),
            max_test_samples=ds_config.get("max_test_samples"),
            max_duration_seconds=ds_config.get("max_duration_seconds", _UNSET),
        )

    # Validate
    if not validate_no_text_overlap(splits):
        print("\nERROR: Validation failed!")
        return 1

    if dry_run:
        print("\n*** DRY RUN complete - pipeline validated, skipping save/push ***")
    else:
        # Save locally
        save_splits(splits, args.output_dir)

        # Push to HuggingFace if requested
        if args.push:
            push_to_hub(splits, args.repo_id, args.private)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ds in splits.items():
        text_col = _get_text_column(ds["train"])
        print(f"\n{name}:")
        print(
            f"  Train: {len(ds['train'])} samples, {len(set(ds['train'][text_col]))} unique texts"
        )
        if "test" in ds:
            print(
                f"  Test:  {len(ds['test'])} samples, {len(set(ds['test'][text_col]))} unique texts"
            )

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
