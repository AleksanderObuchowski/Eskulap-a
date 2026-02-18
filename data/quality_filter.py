"""Quality filtering module for ADMED datasets.

Filters out short phrases, exact duplicates (text and audio), and near-duplicate texts
to improve training data quality and reduce overfitting.
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from datasketch import MinHash, MinHashLSH


@dataclass
class FilterConfig:
    """Configuration for dataset quality filtering."""

    # Minimum text length filter
    min_words: int = 5

    # Text deduplication
    deduplicate_text: bool = True
    # Maximum copies to keep per unique text (1 = full dedup, 2+ = allow some duplicates)
    max_text_duplicates: int = 10

    # Audio deduplication
    deduplicate_audio: bool = True
    # Maximum copies to keep per unique audio
    max_audio_duplicates: int = 1

    # Similarity-based filtering
    use_similarity_filter: bool = True
    similarity_threshold: float = 0.75
    num_perm: int = 128  # MinHash permutations

    # Name of the text column in the dataset
    text_column: str = "text"

    def __str__(self) -> str:
        return (
            f"FilterConfig(min_words={self.min_words}, "
            f"deduplicate_text={self.deduplicate_text}, max_text_dup={self.max_text_duplicates}, "
            f"deduplicate_audio={self.deduplicate_audio}, max_audio_dup={self.max_audio_duplicates}, "
            f"similarity_threshold={self.similarity_threshold})"
        )


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def get_shingles(text: str, k: int = 3) -> set[str]:
    """Get character k-grams (shingles) from text."""
    text = normalize_text(text)
    if len(text) < k:
        return set()
    return set(text[i : i + k] for i in range(len(text) - k + 1))


def create_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    """Create MinHash signature from shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m


def compute_audio_hash(audio_data: dict) -> str | None:
    """
    Compute a hash fingerprint for audio data.

    Uses a combination of audio statistics and sampled values to create
    a robust hash that identifies duplicate audio.

    Args:
        audio_data: Dict with 'array' (numpy array) and 'sampling_rate'

    Returns:
        Hex string hash or None if audio is invalid
    """
    try:
        if audio_data is None:
            return None

        array = audio_data.get("array")
        if array is None or len(array) == 0:
            return None

        # Convert to numpy if needed
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Create fingerprint from audio characteristics:
        # 1. Length (quantized to reduce sensitivity to minor differences)
        length_quantized = len(array) // 100 * 100

        # 2. Statistical features
        mean_val = float(np.mean(array))
        std_val = float(np.std(array))
        max_val = float(np.max(np.abs(array)))

        # 3. Sample some values at fixed positions for more uniqueness
        sample_positions = np.linspace(
            0, len(array) - 1, min(50, len(array)), dtype=int
        )
        samples = array[sample_positions]

        # Combine into a fingerprint string
        fingerprint = f"{length_quantized}:{mean_val:.6f}:{std_val:.6f}:{max_val:.6f}"
        fingerprint += ":" + ",".join(f"{s:.4f}" for s in samples)

        # Hash the fingerprint
        return hashlib.md5(fingerprint.encode()).hexdigest()

    except Exception:
        return None


def filter_dataset(dataset, config: FilterConfig | None = None, **kwargs):
    """
    Apply quality filters to dataset.

    Args:
        dataset: HuggingFace dataset with 'text' and optionally 'audio' columns
        config: FilterConfig object with all settings
        **kwargs: Override config values (min_words, deduplicate_text,
                  deduplicate_audio, similarity_threshold, etc.)

    Returns:
        Filtered dataset with only high-quality samples
    """
    # Build config from defaults and overrides
    if config is None:
        config = FilterConfig()

    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        # Handle legacy parameter names
        elif key == "deduplicate":
            config.deduplicate_text = value

    text_col = config.text_column
    if text_col not in dataset.column_names:
        # Auto-detect text column
        for candidate in ("text", "sentence", "transcription"):
            if candidate in dataset.column_names:
                text_col = candidate
                break
        else:
            raise ValueError(
                f"No text column found in dataset. "
                f"Columns: {dataset.column_names}. "
                f"Set text_column in FilterConfig."
            )
    texts = dataset[text_col]
    original_count = len(texts)

    if original_count == 0:
        print("  Empty dataset, skipping filters")
        return dataset

    valid_indices = list(range(original_count))

    print(f"  Config: {config}")

    # Step 1: Filter by minimum length
    if config.min_words > 0:
        valid_indices = [
            i for i in valid_indices if len(texts[i].split()) >= config.min_words
        ]
        print(
            f"  After min length filter ({config.min_words}+ words): "
            f"{len(valid_indices)}/{original_count} "
            f"({len(valid_indices)/original_count*100:.1f}%)"
        )

    # Step 2: Remove exact text duplicates (keeping up to max_text_duplicates)
    if config.deduplicate_text:
        text_counts: dict[str, int] = {}
        dedup_indices = []
        for i in valid_indices:
            norm = normalize_text(texts[i])
            current_count = text_counts.get(norm, 0)
            if current_count < config.max_text_duplicates:
                text_counts[norm] = current_count + 1
                dedup_indices.append(i)
        valid_indices = dedup_indices
        print(
            f"  After text deduplication (max {config.max_text_duplicates} per text): "
            f"{len(valid_indices)}/{original_count} "
            f"({len(valid_indices)/original_count*100:.1f}%)"
        )

    # Step 3: Remove exact audio duplicates (keeping up to max_audio_duplicates)
    if config.deduplicate_audio and "audio" in dataset.column_names:
        print("  Computing audio fingerprints...")
        audio_counts: dict[str, int] = {}
        dedup_indices = []

        for i in valid_indices:
            audio_data = dataset[i]["audio"]
            audio_hash = compute_audio_hash(audio_data)

            if audio_hash is None:
                # Keep samples with invalid audio (they'll be filtered later by duration)
                dedup_indices.append(i)
            else:
                current_count = audio_counts.get(audio_hash, 0)
                if current_count < config.max_audio_duplicates:
                    audio_counts[audio_hash] = current_count + 1
                    dedup_indices.append(i)

        valid_indices = dedup_indices
        print(
            f"  After audio deduplication (max {config.max_audio_duplicates} per audio): "
            f"{len(valid_indices)}/{original_count} "
            f"({len(valid_indices)/original_count*100:.1f}%)"
        )

    # Step 4: Remove near-duplicates using MinHash LSH
    if (
        config.use_similarity_filter
        and config.similarity_threshold < 1.0
        and len(valid_indices) > 0
    ):
        lsh = MinHashLSH(
            threshold=config.similarity_threshold, num_perm=config.num_perm
        )
        minhashes = {}
        indices_to_keep = []

        # Build MinHash signatures
        for i in valid_indices:
            shingles = get_shingles(texts[i])
            if len(shingles) >= 3:
                m = create_minhash(shingles, config.num_perm)
                minhashes[i] = m

        # Cluster similar texts
        clusters = defaultdict(list)
        for i in valid_indices:
            if i in minhashes:
                result = lsh.query(minhashes[i])
                if result:
                    # Join to existing cluster (use first match as cluster ID)
                    cluster_id = result[0]
                    clusters[cluster_id].append(i)
                else:
                    # No similar text found - start new cluster
                    lsh.insert(i, minhashes[i])
                    clusters[i].append(i)
            else:
                # Very short text after normalization - keep it
                indices_to_keep.append(i)

        # Keep longest text from each cluster (more informative)
        for cluster_id, members in clusters.items():
            best = max(members, key=lambda idx: len(texts[idx]))
            indices_to_keep.append(best)

        valid_indices = indices_to_keep
        print(
            f"  After similarity filtering (threshold={config.similarity_threshold}): "
            f"{len(valid_indices)}/{original_count} "
            f"({len(valid_indices)/original_count*100:.1f}%)"
        )

    # Sort indices to maintain original order
    valid_indices.sort()

    print(
        f"  Final: kept {len(valid_indices)}/{original_count} samples "
        f"({len(valid_indices)/original_count*100:.1f}%)"
    )

    return dataset.select(valid_indices)


def get_filter_stats(dataset, config: FilterConfig | None = None):
    """
    Get statistics about what would be filtered without actually filtering.

    Returns dict with counts for each filter stage.
    """
    if config is None:
        config = FilterConfig()

    text_col = config.text_column
    if text_col not in dataset.column_names:
        for candidate in ("text", "sentence", "transcription"):
            if candidate in dataset.column_names:
                text_col = candidate
                break
    texts = dataset[text_col]
    original_count = len(texts)

    # Count by length
    length_counts = defaultdict(int)
    for t in texts:
        wc = len(t.split())
        length_counts[wc] += 1

    short_count = sum(c for wc, c in length_counts.items() if wc < config.min_words)

    # Count exact text duplicates
    from collections import Counter

    normalized = [normalize_text(t) for t in texts]
    text_counts = Counter(normalized)
    text_duplicate_count = sum(c - 1 for c in text_counts.values() if c > 1)
    unique_text_duplicated = sum(1 for c in text_counts.values() if c > 1)

    stats = {
        "original_count": original_count,
        "short_phrases": short_count,
        "short_phrases_pct": short_count / original_count * 100,
        "text_duplicates": text_duplicate_count,
        "unique_texts_with_duplicates": unique_text_duplicated,
        "length_distribution": dict(sorted(length_counts.items())),
    }

    # Count audio duplicates if available
    if "audio" in dataset.column_names:
        audio_hashes = []
        for i in range(len(dataset)):
            h = compute_audio_hash(dataset[i]["audio"])
            if h:
                audio_hashes.append(h)

        audio_counts = Counter(audio_hashes)
        audio_duplicate_count = sum(c - 1 for c in audio_counts.values() if c > 1)
        stats["audio_duplicates"] = audio_duplicate_count

    return stats
