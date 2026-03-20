"""Simplified experiment configuration for training runs.

Loads pre-processed datasets directly from HuggingFace with minimal config.
No local preprocessing needed - just configure which datasets to use.

Example config (configs/experiment.yaml):

    name: "my_experiment"

    # Model settings
    model:
      type: "qwen3-asr"
      max_duration_seconds: 30.0

    # Training datasets
    train:
      admed_anoni:
        enabled: true
        max_samples: 10000  # optional limit
      admed_human:
        enabled: true

    # Test datasets
    test:
      admed_anoni:
        enabled: true
      admed_human:
        enabled: true

Usage:
    from data.experiment_config import ExperimentConfig

    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    train_data = config.load_train_data()
    test_data = config.load_test_data()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, concatenate_datasets


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    enabled: bool = True
    max_samples: int | None = None  # None = use all

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        limit = f", max={self.max_samples}" if self.max_samples else ""
        return f"{self.name}({status}{limit})"


@dataclass
class ModelConfig:
    """Model-specific settings."""

    type: str = "qwen3-asr"
    max_duration_seconds: float = 30.0


@dataclass
class EvaluationConfig:
    """How training scripts build the eval / tensorized 'test' split."""

    # If True: slice a dev set from the training concat by unique text (legacy).
    # If False (default): use enabled `test:` datasets from config (held-out, aligns with benchmarks).
    inner_dev_from_train: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    model: ModelConfig
    train_datasets: dict[str, DatasetConfig]
    test_datasets: dict[str, DatasetConfig]
    evaluation: EvaluationConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Parse model config
        model_data = data.get("model", {})
        model = ModelConfig(
            type=model_data.get("type", "qwen3-asr"),
            max_duration_seconds=model_data.get("max_duration_seconds", 30.0),
        )

        # Parse train datasets (skip reserved / non-dataset keys)
        train_datasets = {}
        for name, cfg in data.get("train", {}).items():
            if name in ("evaluation", "defaults"):
                continue
            if not isinstance(cfg, dict):
                continue
            train_datasets[name] = DatasetConfig(
                name=name,
                enabled=cfg.get("enabled", True),
                max_samples=cfg.get("max_samples"),
            )

        # Parse test datasets
        test_datasets = {}
        for name, cfg in data.get("test", {}).items():
            if cfg is None:
                cfg = {}
            test_datasets[name] = DatasetConfig(
                name=name,
                enabled=cfg.get("enabled", True),
                max_samples=cfg.get("max_samples"),
            )

        eval_data = data.get("evaluation", {}) or {}
        evaluation = EvaluationConfig(
            inner_dev_from_train=eval_data.get("inner_dev_from_train", False),
        )

        return cls(
            name=data.get("name", "unnamed"),
            model=model,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            evaluation=evaluation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model": {
                "type": self.model.type,
                "max_duration_seconds": self.model.max_duration_seconds,
            },
            "train": {
                name: {"enabled": cfg.enabled, "max_samples": cfg.max_samples}
                for name, cfg in self.train_datasets.items()
            },
            "test": {
                name: {"enabled": cfg.enabled, "max_samples": cfg.max_samples}
                for name, cfg in self.test_datasets.items()
            },
            "evaluation": {
                "inner_dev_from_train": self.evaluation.inner_dev_from_train,
            },
        }

    def get_enabled_train_datasets(self) -> list[DatasetConfig]:
        """Get list of enabled training datasets."""
        return [cfg for cfg in self.train_datasets.values() if cfg.enabled]

    def get_enabled_test_datasets(self) -> list[DatasetConfig]:
        """Get list of enabled test datasets."""
        return [cfg for cfg in self.test_datasets.values() if cfg.enabled]

    def load_train_data(self, use_local_cache: bool = True) -> Dataset:
        """Load and concatenate all enabled training datasets.

        Args:
            use_local_cache: If True, try loading from local cache first

        Returns:
            Concatenated training dataset
        """
        from .load_data import load_single_dataset

        datasets = []
        enabled = self.get_enabled_train_datasets()

        print(f"Loading {len(enabled)} training datasets...")

        for cfg in enabled:
            ds = load_single_dataset(
                cfg.name,
                split="train",
                max_samples=cfg.max_samples,
                use_local=use_local_cache,
            )
            print(f"  {cfg.name}: {len(ds)} samples")
            datasets.append(ds)

        if not datasets:
            raise ValueError("No training datasets enabled!")

        combined = concatenate_datasets(datasets)
        print(f"Total training samples: {len(combined)}")

        return combined

    def load_test_data(self, use_local_cache: bool = True) -> dict[str, Dataset]:
        """Load all enabled test datasets.

        Args:
            use_local_cache: If True, try loading from local cache first

        Returns:
            Dict mapping dataset name to test Dataset
        """
        from .load_data import load_single_dataset

        test_data = {}
        enabled = self.get_enabled_test_datasets()

        print(f"Loading {len(enabled)} test datasets...")

        for cfg in enabled:
            ds = load_single_dataset(
                cfg.name,
                split="test",
                max_samples=cfg.max_samples,
                use_local=use_local_cache,
            )
            print(f"  {cfg.name}: {len(ds)} samples")
            test_data[cfg.name] = ds

        return test_data

    def summary(self) -> str:
        """Return a summary of the configuration."""
        lines = [
            f"Experiment: {self.name}",
            f"Model: {self.model.type} (max {self.model.max_duration_seconds}s)",
            "",
            "Training datasets:",
        ]
        for cfg in self.train_datasets.values():
            status = "✓" if cfg.enabled else "✗"
            limit = f" (max {cfg.max_samples})" if cfg.max_samples else ""
            lines.append(f"  {status} {cfg.name}{limit}")

        lines.append("")
        lines.append("Test datasets:")
        for cfg in self.test_datasets.values():
            status = "✓" if cfg.enabled else "✗"
            limit = f" (max {cfg.max_samples})" if cfg.max_samples else ""
            lines.append(f"  {status} {cfg.name}{limit}")

        lines.append("")
        lines.append("Evaluation:")
        lines.append(
            f"  inner_dev_from_train: {self.evaluation.inner_dev_from_train} "
            "(if True, dev split is carved from train by unique text)"
        )

        return "\n".join(lines)


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        name="default",
        model=ModelConfig(),
        train_datasets={
            "admed_anoni": DatasetConfig(name="admed_anoni", enabled=True),
            "admed_human": DatasetConfig(name="admed_human", enabled=True),
        },
        test_datasets={
            "admed_anoni": DatasetConfig(name="admed_anoni", enabled=True),
            "admed_human": DatasetConfig(name="admed_human", enabled=True),
        },
        evaluation=EvaluationConfig(),
    )
