import json
import re
from typing import Dict, Tuple

CTC_BLANK_ID = 0

# Only lowercase letters and space are valid for fingerspelling.
ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz ")


def normalize_phrase(phrase: str) -> str:
    """Lowercase and strip anything that is not a-z or space."""
    text = str(phrase).lower()
    text = re.sub(r"[^a-z ]", "", text)
    # Collapse multiple spaces and strip edges.
    return re.sub(r" +", " ", text).strip()


def encode_phrase(phrase: str, char_to_idx: Dict[str, int]) -> list:
    """Encode a raw phrase: normalize first, then map to indices."""
    clean = normalize_phrase(phrase)
    return [char_to_idx[c] for c in clean if c in char_to_idx]


def build_ctc_vocab(vocab_json_path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Load Kaggle vocab JSON and keep only lowercase letters + space.

    Returns (char_to_idx, idx_to_char, blank_id).
    Index 0 is always reserved for CTC blank.
    """
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        raw = {k: int(v) for k, v in json.load(f).items()}

    # Filter to allowed chars only, then re-index starting at 1 (0 = blank).
    allowed_sorted = sorted(k for k in raw if k in ALLOWED_CHARS)
    char_to_idx = {c: i + 1 for i, c in enumerate(allowed_sorted)}

    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, CTC_BLANK_ID
