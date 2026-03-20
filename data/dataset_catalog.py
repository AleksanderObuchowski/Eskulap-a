"""Single source of truth for ASR dataset metadata.

Producer fields feed `data.create_clean_splits` (raw HF → clean splits).
Consumer fields feed `data.load_data` (clean HF / local cache → training).

Adding a dataset: append one `DatasetCatalogEntry` to `DATASET_CATALOG`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """One logical dataset (e.g. admed_anoni)."""

    id: str
    description: str
    # --- Consumer (clean splits / training load) ---
    clean_hf_path: str
    clean_hf_config: str | None
    local_name: str
    # --- Producer (raw HF → create_clean_splits) ---
    raw_hf_path: str
    raw_hf_split: str
    raw_hf_test_split: str | None
    text_column: str
    needs_quality_filter: bool
    has_existing_test: bool
    max_train_samples: int | None
    max_test_samples: int | None
    max_duration_seconds: float | None


DATASET_CATALOG: tuple[DatasetCatalogEntry, ...] = (
    DatasetCatalogEntry(
        id="admed_anoni",
        description="Medical reports - SALT anonymized audio",
        clean_hf_path="lion-ai/admed_voice_clean",
        clean_hf_config="anoni",
        local_name="anoni",
        raw_hf_path="lion-ai/admed_voice",
        raw_hf_split="anoni",
        raw_hf_test_split=None,
        text_column="text",
        needs_quality_filter=True,
        has_existing_test=False,
        max_train_samples=None,
        max_test_samples=1000,
        max_duration_seconds=None,
    ),
    DatasetCatalogEntry(
        id="admed_human",
        description="Medical reports - human recordings",
        clean_hf_path="lion-ai/admed_voice_clean",
        clean_hf_config="human",
        local_name="human",
        raw_hf_path="lion-ai/admed_voice",
        raw_hf_split="human",
        raw_hf_test_split=None,
        text_column="text",
        needs_quality_filter=True,
        has_existing_test=False,
        max_train_samples=None,
        max_test_samples=1000,
        max_duration_seconds=None,
    ),
    DatasetCatalogEntry(
        id="youtube",
        description="YouTube medical content",
        clean_hf_path="lion-ai/youtube_asr_30",
        clean_hf_config=None,
        local_name="youtube",
        raw_hf_path="lion-ai/youtube_asr_30",
        raw_hf_split="train",
        raw_hf_test_split=None,
        text_column="sentence",
        needs_quality_filter=True,
        has_existing_test=False,
        max_train_samples=None,
        max_test_samples=500,
        max_duration_seconds=None,
    ),
    DatasetCatalogEntry(
        id="gemini",
        description="Gemini-generated medical transcriptions",
        clean_hf_path="lion-ai/pl_med_asr_test2",
        clean_hf_config=None,
        local_name="gemini",
        raw_hf_path="lion-ai/pl_med_asr_test2",
        raw_hf_split="train",
        raw_hf_test_split=None,
        text_column="text",
        needs_quality_filter=True,
        has_existing_test=False,
        max_train_samples=None,
        max_test_samples=500,
        max_duration_seconds=None,
    ),
    DatasetCatalogEntry(
        id="bigos",
        description="General domain Polish speech (BIGOS)",
        clean_hf_path="lion-ai/bigos",
        clean_hf_config=None,
        local_name="bigos",
        raw_hf_path="lion-ai/bigos",
        raw_hf_split="train",
        raw_hf_test_split="validation",
        text_column="sentence",
        needs_quality_filter=False,
        has_existing_test=True,
        max_train_samples=10000,
        max_test_samples=800,
        max_duration_seconds=None,
    ),
)

# Datasets that share texts and need joint splitting in create_clean_splits.
SHARED_TEXT_GROUPS: list[list[str]] = [
    ["admed_anoni", "admed_human"],
]

_CATALOG_BY_ID: dict[str, DatasetCatalogEntry] = {e.id: e for e in DATASET_CATALOG}


def get_entry(dataset_id: str) -> DatasetCatalogEntry:
    if dataset_id not in _CATALOG_BY_ID:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Available: {list(_CATALOG_BY_ID.keys())}"
        )
    return _CATALOG_BY_ID[dataset_id]


def build_dataset_registry() -> dict[str, dict[str, Any]]:
    """Consumer registry for `load_data.load_single_dataset`."""
    return {
        e.id: {
            "hf_path": e.clean_hf_path,
            "hf_config": e.clean_hf_config,
            "local_name": e.local_name,
            "description": e.description,
        }
        for e in DATASET_CATALOG
    }


def build_dataset_configs() -> dict[str, dict[str, Any]]:
    """Producer configs for `create_clean_splits` (safe to mutate, e.g. CLI limits)."""
    configs: dict[str, dict[str, Any]] = {}
    for e in DATASET_CATALOG:
        d: dict[str, Any] = {
            "hf_path": e.raw_hf_path,
            "hf_split": e.raw_hf_split,
            "local_name": e.local_name,
            "needs_quality_filter": e.needs_quality_filter,
            "has_existing_test": e.has_existing_test,
            "max_train_samples": e.max_train_samples,
            "max_test_samples": e.max_test_samples,
        }
        if e.text_column != "text":
            d["text_column"] = e.text_column
        if e.raw_hf_test_split is not None:
            d["hf_test_split"] = e.raw_hf_test_split
        if e.max_duration_seconds is not None:
            d["max_duration_seconds"] = e.max_duration_seconds
        configs[e.id] = d
    return configs


def dataset_config_copy() -> dict[str, dict[str, Any]]:
    """Deep copy for create_clean_splits CLI (mutates max_* limits)."""
    return copy.deepcopy(build_dataset_configs())
