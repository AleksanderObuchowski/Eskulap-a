"""Simple data loading for training and evaluation.

This module provides a clean interface for loading pre-processed datasets
from HuggingFace. No local preprocessing needed.

Usage:
    # In training script
    from data.load_data import load_train_data, load_test_data

    train = load_train_data("configs/experiment.yaml")
    test = load_test_data("configs/experiment.yaml")

    # Or programmatically
    train = load_train_data(
        admed_anoni=True,
        admed_human=True,
        bigos=5000,  # limit to 5000 samples
    )

CLI:
    # Show what would be loaded
    python -m data.load_data --config configs/experiment.yaml --info

    # Prepare training data and save locally
    python -m data.load_data --config configs/experiment.yaml --output data/train
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datasets import Audio, Dataset, concatenate_datasets, load_dataset, load_from_disk

# Dataset registry - maps names to HuggingFace paths and local cache names
DATASET_REGISTRY = {
    # Admed datasets - have local clean splits with train/test
    "admed_anoni": {
        "hf_path": "lion-ai/admed_voice_clean",
        "hf_config": "anoni",
        "local_name": "anoni",
        "description": "Medical reports - SALT anonymized audio",
    },
    "admed_human": {
        "hf_path": "lion-ai/admed_voice_clean",
        "hf_config": "human",
        "local_name": "human",
        "description": "Medical reports - human recordings",
    },
    # Other datasets - loaded from local clean splits or HuggingFace
    "youtube": {
        "hf_path": "lion-ai/youtube_asr_30",
        "hf_config": None,
        "local_name": "youtube",
        "description": "YouTube medical content",
    },
    "gemini": {
        "hf_path": "lion-ai/pl_med_asr_test2",
        "hf_config": None,
        "local_name": "gemini",
        "description": "Gemini-generated medical transcriptions",
    },
    "bigos": {
        "hf_path": "lion-ai/bigos",
        "hf_config": None,
        "local_name": "bigos",
        "description": "General domain Polish speech (BIGOS)",
    },
}

# Local cache directory
LOCAL_CACHE_DIR = Path(
    os.environ.get("DATASET_CACHE_DIR", "/mnt/data/Eskulap-a/clean_splits")
)


def load_single_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    use_local: bool = True,
    normalize: bool = True,
) -> Dataset:
    """Load a single dataset by name.

    Args:
        name: Dataset name (e.g., "admed_anoni")
        split: "train" or "test"
        max_samples: Optional limit on number of samples
        use_local: Try loading from local cache first
        normalize: Apply text normalization for ASR (default True)

    Returns:
        Dataset
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    ds = None
    info = DATASET_REGISTRY[name]

    # Try local cache first
    if use_local:
        local_name = info.get("local_name", name)
        local_path = LOCAL_CACHE_DIR / local_name
        if local_path.exists():
            try:
                ds_dict = load_from_disk(str(local_path))
                if split in ds_dict:
                    ds = ds_dict[split]
            except Exception as e:
                print(f"  Warning: Failed to load {name} from local cache: {e}")

    # Fall back to HuggingFace
    if ds is None:
        try:
            ds = load_dataset(info["hf_path"], info["hf_config"], split=split)
        except ValueError as e:
            # Split doesn't exist - try "train" split for datasets without test split
            if split == "test" and "train" in str(e):
                raise ValueError(
                    f"Dataset '{name}' doesn't have a 'test' split. "
                    f"Only admed_anoni and admed_human have separate test splits."
                )
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{name}' from HuggingFace ({info['hf_path']}). "
                f"Make sure it exists or use local cache. Error: {e}"
            )

    # Normalize text column to "text" (some HF datasets use "sentence")
    for col in ("sentence", "transcription"):
        if col in ds.column_names and "text" not in ds.column_names:
            ds = ds.rename_column(col, "text")
            break

    # Normalize audio column to "audio" (some datasets store audio in "file_name")
    if "audio" in ds.column_names and not isinstance(ds.features["audio"], Audio):
        ds = ds.remove_columns("audio")
    if "audio" not in ds.column_names:
        for col in ("file_name", "path", "audio_path"):
            if col in ds.column_names and isinstance(ds.features[col], Audio):
                ds = ds.rename_column(col, "audio")
                break
    elif "file_name" in ds.column_names and isinstance(ds.features["file_name"], Audio):
        # Both exist with Audio type — check if "audio" column is broken (first sample None)
        if ds[0]["audio"] is None and ds[0]["file_name"] is not None:
            ds = ds.remove_columns("audio")
            ds = ds.rename_column("file_name", "audio")

    # Filter out samples with missing audio or text
    if "audio" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda x: x["audio"] is not None and x["text"] is not None)
        if len(ds) < before:
            print(f"  Filtered {before - len(ds)} samples with missing audio/text")

    # Normalize text for ASR consistency
    if normalize and "text" in ds.column_names:
        from .text_normalize import normalize_text_for_asr

        ds = ds.map(
            lambda x: {"text": normalize_text_for_asr(x["text"])},
            desc="Normalizing text",
        )

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds


def load_train_data(
    config_path: str | None = None,
    use_local: bool = True,
    **dataset_limits: int | bool,
) -> Dataset:
    """Load training data from config or keyword arguments.

    Args:
        config_path: Path to YAML config file
        use_local: Try loading from local cache first
        **dataset_limits: Dataset names with True (include all) or int (limit samples)
            e.g., admed_anoni=True, bigos=5000

    Returns:
        Concatenated training dataset

    Examples:
        # From config file
        train = load_train_data("configs/experiment.yaml")

        # Programmatically
        train = load_train_data(
            admed_anoni=True,
            admed_human=True,
            bigos=5000,
        )
    """
    if config_path:
        from .experiment_config import ExperimentConfig

        config = ExperimentConfig.from_yaml(config_path)
        return config.load_train_data(use_local_cache=use_local)

    # Build from keyword arguments
    datasets = []
    for name, value in dataset_limits.items():
        if value is False:
            continue

        # Note: isinstance(True, int) is True in Python, so check bool first
        max_samples = (
            value if isinstance(value, int) and not isinstance(value, bool) else None
        )
        ds = load_single_dataset(
            name, split="train", max_samples=max_samples, use_local=use_local
        )
        print(f"  {name}: {len(ds)} samples")
        datasets.append(ds)

    if not datasets:
        raise ValueError("No datasets specified!")

    combined = concatenate_datasets(datasets)
    print(f"Total training samples: {len(combined)}")

    return combined


def load_test_data(
    config_path: str | None = None,
    use_local: bool = True,
    **dataset_limits: int | bool,
) -> dict[str, Dataset]:
    """Load test data from config or keyword arguments.

    Args:
        config_path: Path to YAML config file
        use_local: Try loading from local cache first
        **dataset_limits: Dataset names with True (include all) or int (limit samples)

    Returns:
        Dict mapping dataset name to test Dataset
    """
    if config_path:
        from .experiment_config import ExperimentConfig

        config = ExperimentConfig.from_yaml(config_path)
        return config.load_test_data(use_local_cache=use_local)

    # Build from keyword arguments
    test_data = {}
    for name, value in dataset_limits.items():
        if value is False:
            continue

        # Note: isinstance(True, int) is True in Python, so check bool first
        max_samples = (
            value if isinstance(value, int) and not isinstance(value, bool) else None
        )
        ds = load_single_dataset(
            name, split="test", max_samples=max_samples, use_local=use_local
        )
        print(f"  {name}: {len(ds)} samples")
        test_data[name] = ds

    return test_data


def list_datasets() -> None:
    """Print available datasets."""
    print("Available datasets:")
    print("-" * 60)
    for name, info in DATASET_REGISTRY.items():
        print(f"  {name}")
        print(f"    HF: {info['hf_path']}")
        if info["hf_config"]:
            print(f"    Config: {info['hf_config']}")
        print(f"    {info['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Load pre-processed datasets")
    parser.add_argument(
        "--config",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show config info without loading data",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output",
        help="Save combined training data to this path",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local cache, load directly from HuggingFace",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.info and args.config:
        from .experiment_config import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.config)
        print(config.summary())
        return

    if args.config:
        from .experiment_config import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.config)
        print(config.summary())
        print()

        train = config.load_train_data(use_local_cache=not args.no_local)

        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving to {output_path}...")
            train.save_to_disk(str(output_path))
            print("Done!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
