"""Shared disk cache and preprocessing for ASR training scripts.

Maps Arrow datasets (audio + text) to model-specific tensors. Supports:
- batched ``datasets.map`` (Whisper / GLM-ASR)
- chunked Python loops (Qwen3-ASR; avoids ``map`` hangs with some processors)

Cache paths incorporate a fingerprint of the experiment config and model id so
changes invalidate stale processed data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from typing import Any, Literal

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm

# Bump when preprocessing logic changes (forces cache rebuild).
PREP_VERSION = "1"

Backend = Literal["map_batched", "chunked_loop"]


def compute_training_cache_fingerprint(
    *,
    config_path: str,
    model_family: str,
    base_model_name: str,
    inner_dev_from_train: bool,
    prep_version: str = PREP_VERSION,
) -> str:
    """Stable hash for processed dataset cache directory names."""
    path = os.path.abspath(config_path)
    try:
        with open(path, "rb") as f:
            config_bytes = f.read()
    except OSError:
        config_bytes = path.encode()
    payload = {
        "prep_version": prep_version,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_path": path,
        "model_family": model_family,
        "base_model_name": base_model_name,
        "inner_dev_from_train": inner_dev_from_train,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def resolve_processed_data_path(
    processed_dir: str, model_slug: str, fingerprint_hex: str
) -> str:
    """Directory for ``DatasetDict.save_to_disk`` / ``load_from_disk``."""
    short = fingerprint_hex[:16]
    safe_slug = model_slug.replace("/", "_")
    return os.path.join(processed_dir, f"processed_{safe_slug}_{short}")


def cache_has_required_columns(
    dataset: DatasetDict, required: list[str] | None
) -> bool:
    if not required:
        return True
    for split in ("train", "test"):
        if split not in dataset:
            return False
        cols = dataset[split].column_names
        for c in required:
            if c not in cols:
                return False
    return True


def process_split_chunked(
    split_dataset: Dataset,
    prepare_single: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    chunk_size: int = 500,
    desc: str = "Processing",
) -> Dataset:
    """Row-wise prep in chunks (works around ``dataset.map`` hangs for Qwen processor)."""
    chunks: list[Dataset] = []
    pbar = tqdm(total=len(split_dataset), desc=desc)
    for start in range(0, len(split_dataset), chunk_size):
        end = min(start + chunk_size, len(split_dataset))
        results = []
        for i in range(start, end):
            results.append(prepare_single(split_dataset[i]))
        pbar.update(end - start)
        chunk = Dataset.from_dict(
            {k: [r[k] for r in results] for k in results[0].keys()}
        )
        chunks.append(chunk)
    pbar.close()
    return concatenate_datasets(chunks)


def process_split_map_batched(
    split_dataset: Dataset,
    prepare_batch: Callable[..., dict[str, Any]],
    *,
    batch_size: int = 50,
    num_proc: int = 8,
    desc: str = "Processing",
) -> Dataset:
    """Batched ``map`` preprocessing (Whisper / GLM-ASR)."""
    return split_dataset.map(
        prepare_batch,
        remove_columns=split_dataset.column_names,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=desc,
    )


def load_or_build_processed_dataset(
    *,
    processed_data_path: str,
    raw_splits: DatasetDict,
    backend: Backend,
    prepare_batch: Callable[..., dict[str, Any]] | None = None,
    prepare_single: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    map_batch_size: int = 50,
    map_num_proc: int = 8,
    chunked_size: int = 500,
    required_cache_columns: list[str] | None = None,
) -> DatasetDict:
    """Load cached ``DatasetDict`` or build from ``raw_splits`` (train + test)."""
    if os.path.exists(processed_data_path):
        dataset = load_from_disk(processed_data_path)
        if cache_has_required_columns(dataset, required_cache_columns):
            return dataset
        shutil.rmtree(processed_data_path)

    if backend == "map_batched":
        if prepare_batch is None:
            raise ValueError("prepare_batch required for map_batched backend")
        train_ds = process_split_map_batched(
            raw_splits["train"],
            prepare_batch,
            batch_size=map_batch_size,
            num_proc=map_num_proc,
            desc="Processing train",
        )
        test_ds = process_split_map_batched(
            raw_splits["test"],
            prepare_batch,
            batch_size=map_batch_size,
            num_proc=map_num_proc,
            desc="Processing test",
        )
    elif backend == "chunked_loop":
        if prepare_single is None:
            raise ValueError("prepare_single required for chunked_loop backend")
        train_ds = process_split_chunked(
            raw_splits["train"],
            prepare_single,
            chunk_size=chunked_size,
            desc="Processing train",
        )
        test_ds = process_split_chunked(
            raw_splits["test"],
            prepare_single,
            chunk_size=chunked_size,
            desc="Processing test",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    dataset.save_to_disk(processed_data_path)
    return dataset
