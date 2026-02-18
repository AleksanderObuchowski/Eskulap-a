#!/usr/bin/env python3
"""Demo: Whisper n-best rescoring with Bielik v3 1.5B.

Implements a standard log-linear n-best rescoring pipeline:
1. Generate n-best hypotheses with Whisper beam search.
2. Score each hypothesis with an external LM (Bielik-1.5B-v3).
3. Combine Whisper and LM scores with an interpolation weight.

Scoring objective (length-normalized):
    score(h) = (1 - lambda) * avg_logp_asr(h | x) + lambda * avg_logp_lm(h)
               + word_bonus * num_words(h)

References:
- Hugging Face generation API (`sequences_scores`, `compute_transition_scores`)
  https://huggingface.co/docs/transformers/main/main_classes/text_generation
- ESPnet beam-search scoring (ASR + LM + length bonus)
  https://espnet.github.io/espnet/guide/espnet/nets/BeamSearch.html
- ProGRes (LLM rescoring for ASR n-best hypotheses)
  https://ar5iv.org/html/2401.07370

Example:
    python whisper_llm_rescore_demo.py \
        --audio-path sample.wav \
        --num-beams 8 \
        --num-return-sequences 8 \
        --lm-weight 0.35
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_BIELIK_MODEL = "speakleash/Bielik-1.5B-v3"
TARGET_SAMPLING_RATE = 16_000
_FALLBACK_WARNING_SHOWN = False
_GROUP_BEAM_WARNING_SHOWN = False


@dataclass
class Hypothesis:
    asr_rank: int
    text: str
    asr_logprob_sum: float
    asr_logprob_avg: float
    asr_token_count: int
    lm_logprob_sum: float
    lm_logprob_avg: float
    lm_token_count: int
    combined_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper n-best rescoring with Bielik v3 1.5B"
    )
    parser.add_argument("--audio-path", required=True, help="Path to input audio file")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--bielik-model", default=DEFAULT_BIELIK_MODEL)
    parser.add_argument(
        "--language",
        default="polish",
        help="Whisper language for forced decoder prompt (use 'none' to disable)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument(
        "--num-beam-groups",
        type=int,
        default=1,
        help=">1 enables diverse beam search (must divide --num-beams).",
    )
    parser.add_argument(
        "--diversity-penalty",
        type=float,
        default=0.0,
        help="Diverse beam search penalty (typically 0.1-1.0).",
    )
    parser.add_argument("--num-return-sequences", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lm-weight", type=float, default=0.35)
    parser.add_argument(
        "--word-bonus",
        type=float,
        default=0.0,
        help="Optional insertion/length term: word_bonus * num_words",
    )
    parser.add_argument("--lm-batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate text hypotheses from Whisper n-best",
    )
    parser.add_argument(
        "--fallback-sampling",
        action="store_true",
        help="If beam n-best collapses to 1 unique text, retry with sampling.",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling fallback.",
    )
    parser.add_argument(
        "--sampling-top-p",
        type=float,
        default=0.95,
        help="Top-p for sampling fallback.",
    )
    parser.add_argument(
        "--sampling-top-k",
        type=int,
        default=50,
        help="Top-k for sampling fallback.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save full rescoring details as JSON",
    )
    return parser.parse_args()


def _resample_linear(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    new_length = int(round(audio.shape[0] * target_sr / original_sr))
    old_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=max(1, new_length), endpoint=True)
    resampled = np.interp(new_positions, old_positions, audio)
    return resampled.astype(np.float32, copy=False)


def load_audio_mono_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    if sr != TARGET_SAMPLING_RATE:
        audio = _resample_linear(audio, sr, TARGET_SAMPLING_RATE)
    return audio


def load_whisper(model_id: str, device: str) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    processor = WhisperProcessor.from_pretrained(model_id)
    whisper_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=whisper_dtype,
    ).to(device)
    model.eval()
    return processor, model


def get_whisper_nbest(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: np.ndarray,
    device: str,
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
    language: str | None,
    do_sample: bool = False,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_top_k: int = 50,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    global _FALLBACK_WARNING_SHOWN
    global _GROUP_BEAM_WARNING_SHOWN
    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    generate_kwargs = {
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = sampling_temperature
        generate_kwargs["top_p"] = sampling_top_p
        generate_kwargs["top_k"] = sampling_top_k
    elif num_beam_groups > 1:
        generate_kwargs["num_beam_groups"] = num_beam_groups
        generate_kwargs["diversity_penalty"] = diversity_penalty

    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language.lower(), task="transcribe"
        )
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.no_grad():
        try:
            outputs = model.generate(input_features, **generate_kwargs)
        except ValueError as exc:
            if num_beam_groups > 1 and "Group Beam Search requires" in str(exc):
                if not _GROUP_BEAM_WARNING_SHOWN:
                    print(
                        "Warning: diverse/group beam search unavailable in this "
                        "transformers setup; falling back to standard beam search."
                    )
                    _GROUP_BEAM_WARNING_SHOWN = True
                fallback_kwargs = dict(generate_kwargs)
                fallback_kwargs.pop("num_beam_groups", None)
                fallback_kwargs.pop("diversity_penalty", None)
                outputs = model.generate(input_features, **fallback_kwargs)
            else:
                raise

    try:
        transition_scores = model.compute_transition_scores(
            sequences=outputs.sequences,
            scores=outputs.scores,
            beam_indices=getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        ).cpu()
        asr_token_counts = (transition_scores < 0).sum(dim=1).to(torch.int64)
        asr_logprob_sum = transition_scores.sum(dim=1)
        asr_logprob_avg = asr_logprob_sum / asr_token_counts.clamp_min(1)
    except RuntimeError as exc:
        if (
            hasattr(outputs, "sequences_scores")
            and outputs.sequences_scores is not None
        ):
            # Fallback for Whisper/transformers combinations where
            # compute_transition_scores can fail on beam indices.
            if not _FALLBACK_WARNING_SHOWN:
                print(
                    "Warning: compute_transition_scores failed; falling back to "
                    "generation `sequences_scores`."
                )
                _FALLBACK_WARNING_SHOWN = True
            asr_logprob_avg = outputs.sequences_scores.cpu().to(torch.float32)
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is None:
                asr_token_counts = torch.full(
                    (outputs.sequences.shape[0],),
                    fill_value=outputs.sequences.shape[1],
                    dtype=torch.int64,
                )
            else:
                asr_token_counts = (
                    outputs.sequences.ne(pad_id).sum(dim=1).to(torch.int64).cpu()
                )
            asr_token_counts = asr_token_counts.clamp_min(1)
            asr_logprob_sum = asr_logprob_avg * asr_token_counts.to(torch.float32)
        elif hasattr(outputs, "scores") and outputs.scores:
            # Sampling fallback: score generated suffix from per-step logits.
            step_scores = torch.stack(outputs.scores, dim=1).float().cpu()
            gen_len = step_scores.shape[1]
            generated_tokens = outputs.sequences[:, -gen_len:].cpu()
            step_log_probs = F.log_softmax(step_scores, dim=-1)
            token_log_probs = torch.gather(
                step_log_probs, dim=2, index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)
            asr_logprob_sum = token_log_probs.sum(dim=1)
            asr_token_counts = torch.full(
                (generated_tokens.shape[0],), fill_value=gen_len, dtype=torch.int64
            )
            asr_logprob_avg = asr_logprob_sum / asr_token_counts.clamp_min(1)
        else:
            raise RuntimeError(
                "Whisper n-best scoring failed: no transition/sequence/sampling scores."
            ) from exc

    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    return (
        texts,
        asr_logprob_sum.numpy().astype(np.float32),
        asr_logprob_avg.numpy().astype(np.float32),
        asr_token_counts.numpy().astype(np.int64),
    )


def deduplicate_by_text(
    texts: list[str],
    asr_sum: np.ndarray,
    asr_avg: np.ndarray,
    asr_count: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    keep_indices: list[int] = []
    seen: dict[str, int] = {}

    for idx, text in enumerate(texts):
        key = " ".join(text.strip().split())
        if key in seen:
            prev = seen[key]
            if asr_avg[idx] > asr_avg[prev]:
                seen[key] = idx
        else:
            seen[key] = idx

    keep_indices = sorted(seen.values(), key=lambda i: asr_avg[i], reverse=True)
    return (
        [texts[i] for i in keep_indices],
        asr_sum[keep_indices],
        asr_avg[keep_indices],
        asr_count[keep_indices],
    )


def load_bielik(model_id: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_id} has neither pad_token nor eos_token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    lm_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=lm_dtype,
    ).to(device)
    model.eval()
    return tokenizer, model


def score_with_causal_lm(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lm_sums: list[float] = []
    lm_avgs: list[float] = []
    lm_counts: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(
            token_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        gathered = gathered * shift_mask

        seq_sum = gathered.sum(dim=1)
        seq_count = shift_mask.sum(dim=1).clamp_min(1.0)
        seq_avg = seq_sum / seq_count

        lm_sums.extend(seq_sum.cpu().tolist())
        lm_avgs.extend(seq_avg.cpu().tolist())
        lm_counts.extend(seq_count.cpu().to(torch.int64).tolist())

    return (
        np.array(lm_sums, dtype=np.float32),
        np.array(lm_avgs, dtype=np.float32),
        np.array(lm_counts, dtype=np.int64),
    )


def build_hypotheses(
    texts: list[str],
    asr_sum: np.ndarray,
    asr_avg: np.ndarray,
    asr_count: np.ndarray,
    lm_sum: np.ndarray,
    lm_avg: np.ndarray,
    lm_count: np.ndarray,
    lm_weight: float,
    word_bonus: float,
) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []
    asr_count = np.maximum(1, asr_count.astype(np.int64))

    for idx, text in enumerate(texts):
        words = max(1, len(text.split()))
        combined = (
            (1.0 - lm_weight) * float(asr_avg[idx])
            + lm_weight * float(lm_avg[idx])
            + word_bonus * words
        )
        hypotheses.append(
            Hypothesis(
                asr_rank=idx + 1,
                text=text.strip(),
                asr_logprob_sum=float(asr_sum[idx]),
                asr_logprob_avg=float(asr_avg[idx]),
                asr_token_count=int(asr_count[idx]),
                lm_logprob_sum=float(lm_sum[idx]),
                lm_logprob_avg=float(lm_avg[idx]),
                lm_token_count=int(lm_count[idx]),
                combined_score=combined,
            )
        )
    return hypotheses


def print_summary(hypotheses: list[Hypothesis]) -> None:
    best_asr = max(hypotheses, key=lambda h: h.asr_logprob_avg)
    best_rescored = max(hypotheses, key=lambda h: h.combined_score)

    print("\nBaseline Whisper best (ASR score):")
    print(best_asr.text)

    print("\nBest after LLM rescoring:")
    print(best_rescored.text)

    print("\nTop hypotheses by combined score:")
    ranked = sorted(hypotheses, key=lambda h: h.combined_score, reverse=True)
    for rank, hyp in enumerate(ranked, start=1):
        print(
            f"[{rank:02d}] combined={hyp.combined_score:.4f} "
            f"asr_avg={hyp.asr_logprob_avg:.4f} "
            f"lm_avg={hyp.lm_logprob_avg:.4f} | {hyp.text}"
        )


def main() -> None:
    args = parse_args()

    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences must be <= --num-beams for beam search")
    if args.num_beam_groups < 1:
        raise ValueError("--num-beam-groups must be >= 1")
    if args.num_beam_groups > args.num_beams:
        raise ValueError("--num-beam-groups must be <= --num-beams")
    if args.num_beams % args.num_beam_groups != 0:
        raise ValueError("--num-beams must be divisible by --num-beam-groups")
    if not (0.0 <= args.lm_weight <= 1.0):
        raise ValueError("--lm-weight must be in [0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA device, but CUDA is not available. "
            "Use --device cpu or run on a CUDA-enabled machine."
        )

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    language = None if args.language.strip().lower() == "none" else args.language.strip()
    device = args.device

    print(f"Loading audio: {audio_path}")
    audio = load_audio_mono_16k(str(audio_path))
    duration_seconds = audio.shape[0] / TARGET_SAMPLING_RATE
    print(f"Audio length: {duration_seconds:.2f} s")
    if duration_seconds > 30:
        print(
            "Warning: this demo runs single-window Whisper decoding. "
            "For long-form audio (>30s), segment audio first."
        )

    print(f"\nLoading Whisper model: {args.whisper_model}")
    whisper_processor, whisper_model = load_whisper(args.whisper_model, device)

    print("Generating Whisper n-best hypotheses...")
    texts, asr_sum, asr_avg, asr_count = get_whisper_nbest(
        model=whisper_model,
        processor=whisper_processor,
        audio=audio,
        device=device,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        language=language,
        do_sample=False,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
    )

    if not args.allow_duplicates:
        texts, asr_sum, asr_avg, asr_count = deduplicate_by_text(
            texts, asr_sum, asr_avg, asr_count
        )

    if len(texts) < 2 and args.fallback_sampling:
        print(
            "Beam n-best collapsed to <2 unique hypotheses. "
            "Retrying with sampling fallback..."
        )
        sample_count = max(2, args.num_return_sequences)
        texts, asr_sum, asr_avg, asr_count = get_whisper_nbest(
            model=whisper_model,
            processor=whisper_processor,
            audio=audio,
            device=device,
            num_beams=1,
            num_return_sequences=sample_count,
            max_new_tokens=args.max_new_tokens,
            language=language,
            do_sample=True,
            sampling_temperature=args.sampling_temperature,
            sampling_top_p=args.sampling_top_p,
            sampling_top_k=args.sampling_top_k,
        )
        if not args.allow_duplicates:
            texts, asr_sum, asr_avg, asr_count = deduplicate_by_text(
                texts, asr_sum, asr_avg, asr_count
            )

    if len(texts) < 2:
        raise RuntimeError(
            "Need at least 2 distinct hypotheses for rescoring. "
            "Try increasing --num-beams and --num-return-sequences."
        )

    # Free Whisper memory before loading Bielik (important on single-GPU setups).
    del whisper_model
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"\nLoading Bielik model: {args.bielik_model}")
    bielik_tokenizer, bielik_model = load_bielik(args.bielik_model, device)

    print("Scoring hypotheses with Bielik...")
    lm_sum, lm_avg, lm_count = score_with_causal_lm(
        texts=texts,
        tokenizer=bielik_tokenizer,
        model=bielik_model,
        device=device,
        batch_size=args.lm_batch_size,
    )

    hypotheses = build_hypotheses(
        texts=texts,
        asr_sum=asr_sum,
        asr_avg=asr_avg,
        asr_count=asr_count,
        lm_sum=lm_sum,
        lm_avg=lm_avg,
        lm_count=lm_count,
        lm_weight=args.lm_weight,
        word_bonus=args.word_bonus,
    )

    print_summary(hypotheses)

    if args.output_json:
        payload = {
            "audio_path": str(audio_path),
            "whisper_model": args.whisper_model,
            "bielik_model": args.bielik_model,
            "lm_weight": args.lm_weight,
            "word_bonus": args.word_bonus,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "language": language,
            "results": [asdict(h) for h in hypotheses],
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nSaved details to: {output_path}")


if __name__ == "__main__":
    main()
