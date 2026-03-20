# Data pipeline

This repo treats ASR training data in three explicit stages. Keeping them separate makes it clear what to extend when adding a dataset or a new model head.

## 1. Build (clean splits)

**Goal:** Fixed train/test partitions with no text leakage, optional quality filtering, optional push to the Hugging Face Hub.

- **Code:** [`data/create_clean_splits.py`](../data/create_clean_splits.py), [`data/quality_filter.py`](../data/quality_filter.py)
- **Metadata:** Producer fields in [`data/dataset_catalog.py`](../data/dataset_catalog.py) (raw HF paths, `needs_quality_filter`, caps, shared-text groups)
- **CLI:** `uv run python -m data.create_clean_splits --datasets ...`

Shared-text datasets (e.g. admed_anoni + admed_human) are split jointly so the same transcript cannot appear in both train and test.

## 2. Load (Arrow datasets)

**Goal:** Read **clean** data from local disk (`DATASET_CACHE_DIR` / `CLEAN_SPLITS_DIR`) or from published “clean” Hub datasets; normalize column names and ASR text.

- **Code:** [`data/load_data.py`](../data/load_data.py), [`data/experiment_config.py`](../data/experiment_config.py), [`data/text_normalize.py`](../data/text_normalize.py)
- **Metadata:** Consumer fields in [`data/dataset_catalog.py`](../data/dataset_catalog.py) (clean Hub path, config name, `local_name`)

Experiment YAML lists which sources are enabled for `train:` and `test:`.

## 3. Tensorize (model inputs + cache)

**Goal:** Convert `audio` + `text` rows into model-specific tensors, then cache a `DatasetDict` (`train` / `test`) for repeat runs.

- **Code:** [`data/training_dataset.py`](../data/training_dataset.py)
- **Callers:** [`train.py`](../train.py) (batched `datasets.map` for Whisper / GLM-ASR), [`train_qwen3_asr.py`](../train_qwen3_asr.py) (chunked loop to avoid `map` hangs with the Qwen processor)

Cache directories include a fingerprint of the experiment YAML and model id so changes invalidate stale caches.

## Evaluation split during training

Training scripts build the tensorized **test** split in one of two ways (see `evaluation:` in `configs/experiment.yaml`):

- **`inner_dev_from_train: false` (default):** Concatenate enabled **`test:`** datasets — aligns with benchmark-style held-out data.
- **`inner_dev_from_train: true`:** Legacy behaviour — hold out a dev slice from the **training** concat using unique text keys.

If `inner_dev_from_train` is false but no test datasets are enabled, scripts fall back to the inner-dev behaviour and print a warning.

## Adding a new corpus

1. Add a [`DatasetCatalogEntry`](../data/dataset_catalog.py) (producer + consumer fields).
2. If it shares transcripts with another corpus, add the group to `SHARED_TEXT_GROUPS`.
3. Regenerate clean splits, then train with the new source enabled in YAML.
