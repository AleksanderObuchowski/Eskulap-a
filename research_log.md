# Research Log

## 2026-02-12 - LLM-based rescoring for Whisper ASR

### Research question
Does external LLM rescoring improve Whisper transcription quality in Polish medical ASR, compared with Whisper's own 1-best output?

### Method under test
We tested **n-best hypothesis rescoring** with a log-linear objective:

`score(h) = (1 - lambda) * avg_logp_asr(h|x) + lambda * avg_logp_lm(h) + word_bonus * len_words(h)`

Where:
- `avg_logp_asr(h|x)` is the normalized Whisper score for hypothesis `h`
- `avg_logp_lm(h)` is the normalized Bielik LM likelihood of `h`
- `lambda` controls ASR vs LM influence

Models:
- ASR: `openai/whisper-large-v3-turbo`
- LM for rescoring: `speakleash/Bielik-1.5B-v3`

### Experimental protocol
- Dataset slice: `admed_anoni` + `admed_human`
- Candidate pool: 80 samples per dataset
- Evaluated subset: 6 short utterances per dataset (12 total)
- Decoding on CPU
- Primary rescoring setting: `lambda=0.35`, `word_bonus=0.0`
- N-best generation:
  - Beam request: `num_beams=8`, `num_return_sequences=8`
  - If beam collapsed to one unique transcript, fallback generation used stochastic sampling (`temperature=0.9`, `top_p=0.95`, `top_k=50`)

### Findings
1. **Rescoring requires candidate diversity**.
   When Whisper produced only one unique hypothesis, rescoring had no effect.
2. **With fallback diversity enabled, rescoring changed outputs and improved metrics**.
   - Fallback sampling used on 11/12 samples
   - Changed predictions: 2/12
   - Baseline: `WER=0.183908`, `CER=0.048105`
   - Rescored: `WER=0.160920`, `CER=0.042274`
   - Relative improvement: `WER -12.5%`, `CER -12.12%`
3. **Observed improvement is directional, not yet conclusive** due to small sample size.

### General interpretation
The method appears promising **only when n-best alternatives are meaningfully diverse**. In practical terms: LLM rescoring is not a free gain on top of deterministic beam output; it depends on upstream decoding producing competing candidates.

### Threats to validity
- Very small evaluation set (12 utterances)
- Short-utterance bias in sample selection
- Single LM weight reported here (`lambda=0.35`) for the final test
- CPU-only run may limit practical decoding settings compared to production GPU runs

### Recommended next experiments
1. Increase evaluation size (e.g., 100-500 samples) with fixed random seed.
2. Run a controlled sweep over `lambda`, decoding diversity settings, and optional word bonus.
3. Report confidence intervals or bootstrap uncertainty for WER/CER deltas.
4. Separate analysis by dataset/domain (`admed_anoni` vs `admed_human`) and utterance length.

### Artifact
- Detailed run output: `benchmark_results/rescoring_subset_cpu_with_fallback.json`

---

## 2026-02-11 → 2026-02-13 - LoRA fine-tuning iterations (v1–v4)

### Goal
Fine-tune Whisper large-v3-turbo with LoRA on Polish medical ASR data (5 datasets: admed_anoni, admed_human, youtube, gemini, bigos) and systematically improve WER.

### Iteration history

#### Baseline (no LoRA, pre-normalization)
- Benchmark: `multi_test_openai_whisper-large-v3-turbo_20260211_123139.json`
- Avg WER (unweighted): 24.36%

#### LoRA v1 — initial attempt
- Benchmark: `multi_test_..._20260211_125842.json`
- Config: LR=1e-4, r=64, alpha=64, q_proj/v_proj only, 5 epochs, no regularization
- Avg WER: 24.54% — **worse than baseline**
- Root cause: LR too high, LoRA rank too high (overfitting), no regularization

#### Text normalization applied (2026-02-12)
- Created `data/text_normalize.py` — pipeline to align all datasets with Whisper's output style (capitalization, trailing period, Unicode punctuation, markdown removal, parenthetical removal)
- Applied to both train labels and benchmark evaluation (predictions + references)
- Baseline WER after normalization: 20.64% (unweighted) — **3.72pp free improvement**
- Benchmark: `multi_test_openai_whisper-large-v3-turbo_20260212_125528.json`

#### LoRA v2 — post-normalization, LR reduced
- Benchmark: `multi_test_..._20260212_131222.json`
- Config: LR=5e-5, r=64, alpha=64, q_proj/v_proj, 5 epochs
- Improved on most sets but gemini catastrophically regressed (+11.25pp WER)
- Root cause: gemini `max_duration_seconds: None` allowed >30s audio into training. Whisper truncates at 30s, so the model learned to generate text without acoustic evidence → hallucination loops.

#### Gemini fix (2026-02-12)
- Fixed `create_clean_splits.py`: removed `None` override for gemini max_duration
- Fixed `load_raw_dataset()`: gemini had audio in `file_name` column, not `audio`
- Gemini after fix: 1301 train / 144 test (removed 686 samples >30s)

#### LoRA v3 — gemini fixed
- Benchmark: `multi_test_..._20260212_230033.json`
- Config: same as v2 (LR=5e-5, r=64, alpha=64, q_proj/v_proj, 5 epochs)
- Gemini: 26.79% → 8.69% (hallucination loops gone)
- admed still ~22% — not satisfactory

#### LoRA v4 — regularization + reduced rank
- Benchmark: `multi_test_..._20260213_094314.json`
- Config changes vs v3:
  - LORA_R: 64 → 16
  - LORA_ALPHA: 64 → 32 (effective scaling = alpha/r = 2.0)
  - LoRA targets: q_proj/v_proj → q_proj/v_proj/k_proj/o_proj/fc1/fc2
  - NUM_TRAIN_EPOCHS: 5 → 3
  - WEIGHT_DECAY: 0 → 0.01
  - LABEL_SMOOTHING: 0 → 0.1
  - Added decoder_input_ids to collator (required for label_smoothing compatibility)
  - Added repetition_penalty=1.2 + no_repeat_ngram_size=3 to benchmark generation
- Trainable params: ~7.1M (v3) → 1.8M (v4)

| Test Set | Baseline | v3 | v4 | v4 vs Baseline |
|---|---|---|---|---|
| admed_anoni | 24.43% | 22.28% | 22.02% | -2.41pp |
| admed_human | 21.20% | 22.34% | 17.47% | -3.73pp |
| youtube | 15.06% | 13.85% | 18.18% | +3.12pp |
| gemini | 26.79% | 8.69% | 13.07% | -13.72pp |
| bigos | 15.72% | 11.04% | 12.38% | -3.34pp |
| **Avg (weighted)** | **20.73%** | **18.20%** | **17.48%** | **-3.25pp** |

### v4 detailed error analysis (2026-02-13)

#### YouTube regression root cause
- 77% of youtube samples got worse in v4, only 11.5% improved
- Caused entirely by `repetition_penalty=1.2` and `no_repeat_ngram_size=3` in benchmark generation config
- `no_repeat_ngram_size=3` blocks legitimate repeated 3-grams in lecture content (e.g. listing patterns "Mogą być to... Mogą być to...")
- `repetition_penalty=1.2` distorts all token probabilities globally — 248/255 changed predictions had no repeated words at all
- Only 1 sample actually benefited (one repetition loop fixed), at a cost of +706 errors across 201 samples
- **Action**: remove both parameters from benchmark.py

#### admed_anoni error breakdown (22.02% WER)
Informed by the ADMEDVOICE dataset paper (Nature Scientific Data, Czyżewski et al. 2025):

- **37.6% of samples are perfect** (WER=0). Errors are concentrated in the remaining 62.4%.
- **~4pp is punctuation** (comma↔period, parentheses, em-dashes). WER without punctuation: ~18%.
- **~2pp is text normalization ambiguity** — arguably correct alternative transcriptions:
  - Roman numerals: `V` → `piątej` (9x)
  - Unit expansions: `mg/dl` → `mg na decylitr` (8x)
  - Number-unit spacing: `500ml` → `500 ml` (7x)
  - Dimension format: `14-14-41 mm` → `14x14x41 mm` (3x)
  - The paper itself notes this issue: "fully expanding metric unit abbreviations led to erroneous reading"
- **~14pp is genuine ASR errors**:
  - Medical term garbling: `zastawki`→`zestawki`, `tętniczego`→`tynczego`, `lidokainę`→`do kainy`
  - Short emergency commands: `Analiza rytmu.`→`Anna Rizalewton`, `i-GEL` never correct, `ROSC`→`Rozk.`
  - Abbreviation confusion: `USG`→`UESG`/`USB`
  - Cyrillic hallucination: `zastawki`→`заставки` (4 occurrences)
- **Errors are audio-dependent**: 209/269 repeated references produce inconsistent predictions across different audio recordings of the same text
- **SALT anonymization** changes speaker characteristics and spectral content, adding acoustic difficulty
- **Catastrophic forgetting is low** (6.1%): 18/297 baseline-perfect samples broken, mostly minor consonant corruption (`Przerwa`→`Pzerwa`)
- **Referrals are the LoRA's strongest domain**: 18.2% → 8.5% WER, 11:1 improve/regress ratio
- **LoRA regresses on long sentences** (16+ words): more regressions than improvements, likely autoregressive drift

#### Dataset paper reference results (Whisper-medium, full fine-tune, not LoRA)
| Model | Anonymized WER (95% CI) |
|---|---|
| Whisper pretrained | 26.6% [25.4, 28.0] |
| Fine-tuned Anoni only | 15.9% [14.8, 17.1] |
| Fine-tuned N+A+S | 17.2% [16.1, 18.4] |
| Fine-tuned N+A+S+CV+Fl | 8.6% [8.0, 9.3] |

Our 22% with decoder-only LoRA is ~6pp behind their full fine-tune on anonymized-only, but they used Whisper-medium (769M) with full parameter updates.

### Pending improvements
1. Remove repetition_penalty and no_repeat_ngram_size from benchmark.py
2. Expand text normalization to handle Roman numerals, unit expansions, dimension formats
3. Reduce max_text_duplicates for admed from 4 to 2
4. Consider adding CommonVoice/Fleurs data (paper shows N+A+S+CV+Fl achieves 8.6% WER)
