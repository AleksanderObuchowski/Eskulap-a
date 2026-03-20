# Whisper + Bielik LLM Rescoring

This repository includes `eval/llm_rescore.py`, a minimal n-best rescoring pipeline for ASR (`whisper_llm_rescore_demo.py` is a thin CLI wrapper).

## Method

For a given audio input `x`, Whisper generates an n-best list of transcriptions `h`.

Each hypothesis gets:
- `avg_logp_asr(h | x)`: average token log-probability from Whisper beam search (`compute_transition_scores`)
- `avg_logp_lm(h)`: average token log-probability under Bielik v3 1.5B

Final score:

`score(h) = (1 - lambda) * avg_logp_asr(h | x) + lambda * avg_logp_lm(h) + word_bonus * num_words(h)`

This is the standard log-linear rescoring idea used in ASR beam search with external LMs.

## Why This Is Proper Rescoring

- It uses Whisper n-best hypotheses (not only 1-best text post-editing).
- It scores all candidates with a causal LM likelihood.
- It combines ASR and LM scores with explicit interpolation and optional insertion/length term.

## Sources

- Hugging Face generation docs (`sequences_scores`, `compute_transition_scores`):
  https://huggingface.co/docs/transformers/main/main_classes/text_generation
- ESPnet beam-search score composition (AM/decoder + LM + length bonus):
  https://espnet.github.io/espnet/guide/espnet/nets/BeamSearch.html
- ProGRes paper (LLM rescoring for ASR):
  https://ar5iv.org/html/2401.07370
- Bielik v3 1.5B model:
  https://huggingface.co/speakleash/Bielik-1.5B-v3

## Quick Run

```bash
python -m eval.llm_rescore \
  --audio-path path/to/audio.wav \
  --num-beams 8 \
  --num-return-sequences 8 \
  --num-beam-groups 4 \
  --diversity-penalty 0.4 \
  --fallback-sampling \
  --lm-weight 0.35 \
  --output-json benchmark_results/rescoring_demo.json
```

Tune `--lm-weight` (and optionally `--word-bonus`) on a dev set for best WER.

If your environment has no GPU, add `--device cpu` (slower).

If n-best still collapses to one text, increase diversity:
- higher `--num-beams` / `--num-return-sequences`
- enable diverse beam search with `--num-beam-groups` + `--diversity-penalty`
- use `--fallback-sampling` and tune `--sampling-temperature`, `--sampling-top-p`
