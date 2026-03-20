"""Text normalization for Polish medical ASR datasets.

Normalizes text to match Whisper's natural Polish output style:
capitalized first letter, proper punctuation, clean spacing, no formatting artifacts.

Usage:
    from data.text_normalize import normalize_text_for_asr

    text = normalize_text_for_asr("Tak.Jeśli **bold**\nnowa linia")
    # -> "Tak. Jeśli bold nowa linia."
"""

import re


def _guard(text: str | None) -> str:
    """Handle None/empty, strip whitespace."""
    if text is None:
        return ""
    return text.strip()


def _remove_markdown(text: str) -> str:
    """Remove markdown formatting: **bold**, *italic*, bullet markers."""
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic: *text* or _text_ (but not inside words like e_mail)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Bullet markers at start of line
    text = re.sub(r"^[\-\•\*]\s+", "", text, flags=re.MULTILINE)
    # Numbered list markers at start of line (e.g. "1. ", "2) ")
    text = re.sub(r"^\d+[\.\)]\s+", "", text, flags=re.MULTILINE)
    return text


def _flatten_newlines(text: str) -> str:
    """Replace newlines with spaces."""
    return text.replace("\n", " ").replace("\r", " ")


def _fix_missing_space_after_punctuation(text: str) -> str:
    """Fix missing space after punctuation, preserving decimals.

    "Tak.Jeśli" -> "Tak. Jeśli"
    But preserves: "1.5", "1,5", abbreviations like "np." before lowercase
    """
    # Add space after . ! ? : ; when followed by an uppercase letter or opening paren
    # This avoids breaking decimals (digit.digit) and abbreviations
    text = re.sub(r"([.!?])([A-ZĄĆĘŁŃÓŚŹŻ])", r"\1 \2", text)
    # Also fix comma followed by uppercase (but not digit,digit)
    text = re.sub(r",([A-ZĄĆĘŁŃÓŚŹŻ])", r", \1", text)
    # Fix colon/semicolon followed by a letter
    text = re.sub(r"([;:])([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])", r"\1 \2", text)
    return text


def _normalize_unicode_punctuation(text: str) -> str:
    """Normalize Unicode punctuation to ASCII equivalents."""
    # Em-dash / en-dash -> hyphen
    text = text.replace("\u2013", "-")  # en-dash
    text = text.replace("\u2014", "-")  # em-dash
    # Smart quotes -> ASCII
    text = text.replace("\u201c", '"')  # left double
    text = text.replace("\u201d", '"')  # right double
    text = text.replace("\u201e", '"')  # low double (Polish opening)
    text = text.replace("\u2018", "'")  # left single
    text = text.replace("\u2019", "'")  # right single
    # Ellipsis -> three dots
    text = text.replace("\u2026", "...")
    # NBSP -> space
    text = text.replace("\u00a0", " ")
    # Remove zero-width chars
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text


def _remove_long_parentheticals(text: str) -> str:
    """Remove parenthetical notes longer than 50 chars (editor annotations).

    Keeps short medical ones like (TIRADS 3), (USG), etc.
    """
    def _replace(m):
        content = m.group(1)
        if len(content) > 50:
            return ""
        return m.group(0)

    text = re.sub(r"\(([^)]*)\)", _replace, text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into single space, strip."""
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _capitalize_first_letter(text: str) -> str:
    """Uppercase only the first alphabetic character; rest untouched.

    Preserves medical abbreviations like USG, EKG, etc.
    """
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def _ensure_trailing_period(text: str) -> str:
    """Add trailing period if text has 2+ words and doesn't end with sentence punctuation."""
    if not text:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    if text[-1] not in ".!?":
        text += "."
    return text


def normalize_text_for_asr(text: str | None) -> str:
    """Normalize text for ASR training and evaluation.

    Pipeline:
    1. Guard (None/empty)
    2. Remove markdown
    3. Flatten newlines
    4. Fix missing space after punctuation
    5. Normalize Unicode punctuation
    6. Remove long parenthetical notes
    7. Collapse whitespace
    8. Capitalize first letter
    9. Ensure trailing period

    Args:
        text: Input text (can be None)

    Returns:
        Normalized text string (empty string for None/empty input)
    """
    text = _guard(text)
    if not text:
        return ""
    text = _remove_markdown(text)
    text = _flatten_newlines(text)
    text = _fix_missing_space_after_punctuation(text)
    text = _normalize_unicode_punctuation(text)
    text = _remove_long_parentheticals(text)
    text = _collapse_whitespace(text)
    if not text:
        return ""
    text = _capitalize_first_letter(text)
    text = _ensure_trailing_period(text)
    return text
