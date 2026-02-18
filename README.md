# Eskulap-a: Polish Medical ASR

Fine-tuning ASR models (Whisper, Qwen3-ASR) for Polish medical domain.

## Setup

```bash
# Install dependencies
uv sync

# Configure paths in .env (optional)
cp .env.example .env
```

## Data

Training and test data are pre-processed and stored locally. The pipeline loads from clean splits with no text overlap between train and test sets.

### Available Datasets

| Dataset | Description | Train | Test |
|---------|-------------|-------|------|
| admed_anoni | Medical reports - SALT anonymized audio | ~15k | ~1.5k |
| admed_human | Medical reports - human recordings | ~6k | ~1.5k |
| youtube | YouTube medical content | ~7k | ~800 |
| gemini | Gemini-generated medical transcriptions | ~2k | ~200 |
| bigos | General domain Polish speech | ~20k | ~2k |

Run `python -m data.create_clean_splits --datasets <names>` to create clean splits for each dataset.

### Loading Data

```python
from data import load_train_data, load_test_data

# Load admed datasets
train = load_train_data(admed_anoni=True, admed_human=True)

# Load all available training data
train = load_train_data(admed_anoni=True, admed_human=True, youtube=True, gemini=True)

# Load with sample limits
train = load_train_data(admed_anoni=1000, admed_human=500, youtube=500)

# Load from config file
train = load_train_data(config_path="configs/experiment.yaml")

# Load test data (only admed datasets have test splits)
test = load_test_data(admed_anoni=True, admed_human=True)
```

### Creating/Regenerating Splits

To create clean train/test splits from raw HuggingFace data:

```bash
# Create splits for admed datasets (default)
uv run python -m data.create_clean_splits

# Create splits for specific datasets
uv run python -m data.create_clean_splits --datasets youtube gemini bigos

# Create all datasets
uv run python -m data.create_clean_splits --datasets admed_anoni admed_human youtube gemini bigos

# Push to HuggingFace
uv run python -m data.create_clean_splits --datasets admed_anoni admed_human --push --repo-id lion-ai/admed_voice_clean
```

## Configuration

Experiment configs are YAML files in `configs/`:

```yaml
# configs/experiment.yaml
name: "medical_asr_v1"

model:
  type: "qwen3-asr"
  max_duration_seconds: 30.0

train:
  admed_anoni:
    enabled: true
  admed_human:
    enabled: true

test:
  admed_anoni:
    enabled: true
  admed_human:
    enabled: true
```

You can limit samples per dataset:

```yaml
train:
  admed_anoni:
    enabled: true
    max_samples: 5000
  admed_human:
    enabled: true
    max_samples: 2000
```

## Training

### Qwen3-ASR

```bash
uv run python train_qwen3_asr.py
```

### Whisper / GLM-ASR

```bash
uv run python train.py
```

## Benchmarking

Evaluate models on test sets:

```bash
# Default model (Qwen3 + LoRA)
uv run python benchmark.py

# Use config file
uv run python benchmark.py --config configs/experiment.yaml

# Use specific model
uv run python benchmark.py --model Qwen/Qwen3-ASR-1.7B

# Use a LoRA adapter (auto-detected)
uv run python benchmark.py --model models/my-lora-checkpoint

# Use Whisper
uv run python benchmark.py --model-type whisper --model openai/whisper-large-v3-turbo

# Specify test sets directly
uv run python benchmark.py --test-sets admed_anoni admed_human

# Save results
uv run python benchmark.py --save

# With LM rescoring (Whisper + Bielik n-best rescoring)
uv run python benchmark.py --model-type whisper --model openai/whisper-large-v3-turbo \
    --rescore --test-sets admed_human

# Custom rescoring parameters
uv run python benchmark.py --model-type whisper --model openai/whisper-large-v3-turbo \
    --rescore --lm-model speakleash/Bielik-1.5B-v3 --lm-weight 0.35 \
    --num-beams-rescore 8 --lm-batch-size 8 --test-sets admed_human
```

### LM Rescoring

The `--rescore` flag enables n-best rescoring for Whisper models. It generates multiple hypotheses via beam search, scores each with a causal language model, and picks the best by combined ASR + LM score.

| Option | Default | Description |
|--------|---------|-------------|
| `--rescore` | off | Enable LM rescoring |
| `--lm-model` | `speakleash/Bielik-1.5B-v3` | Causal LM for rescoring |
| `--lm-weight` | `0.35` | LM interpolation weight (0 = ASR only, 1 = LM only) |
| `--num-beams-rescore` | `8` | Beam count for n-best generation |
| `--lm-batch-size` | `8` | Batch size for LM scoring |

### Output

```
======================================================================
BENCHMARK RESULTS
======================================================================
Model: Qwen/Qwen3-ASR-1.7B+checkpoint-7350

Test Set                  WER          CER          Samples
----------------------------------------------------------------------
admed_anoni               0.1109       0.0548       1500
admed_human               0.1234       0.0612       1500
----------------------------------------------------------------------
Average (weighted)        0.1172       0.0580       3000
======================================================================
```

## Project Structure

```
data/
├── __init__.py           # Module exports
├── load_data.py          # Main data loading API
├── experiment_config.py  # YAML config loader
├── create_clean_splits.py # Script to regenerate splits
└── quality_filter.py     # Text/audio deduplication

configs/
└── experiment.yaml       # Training/test configuration

train.py                  # Whisper/GLM-ASR training
train_qwen3_asr.py        # Qwen3-ASR training
benchmark.py              # Model evaluation
app.py                    # Gradio demo app
```

## Gradio App

```bash
uv run python app.py
```

Launches a web interface for testing transcription.
