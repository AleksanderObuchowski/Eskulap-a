"""Data processing module for ASR training.

Main API:
    from data import load_train_data, load_test_data

    # Load all datasets
    train = load_train_data(admed_anoni=True, admed_human=True)

    # Load with sample limits
    train = load_train_data(admed_anoni=1000, admed_human=500)

    # Load from config file
    train = load_train_data(config_path="configs/experiment.yaml")
"""

from .experiment_config import DatasetConfig, EvaluationConfig, ExperimentConfig
from .load_data import (
    DATASET_REGISTRY,
    load_single_dataset,
    load_test_data,
    load_train_data,
)
from .text_normalize import normalize_text_for_asr

__all__ = [
    "load_train_data",
    "load_test_data",
    "load_single_dataset",
    "DATASET_REGISTRY",
    "ExperimentConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "normalize_text_for_asr",
]
