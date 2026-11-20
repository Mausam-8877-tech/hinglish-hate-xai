"""Member 1: data loading and preprocessing.

Other tasks only need ONE import:

    from xhate.data import load_split
    train = load_split("hasoc", "train")      # or "hatexplain"; "train" / "val" / "test"

The torch Dataset lives in ``xhate.data.dataset`` and is imported separately,
so that this package works without torch installed.
"""

# Label constants and the raw -> binary mapping.
from .labels import HOF, LABEL_NAMES, NOT, to_binary
# Functions to read the processed data (what other members use).
from .loader import load_all, load_dataframe, load_split, read_jsonl, write_jsonl
# Text cleaning functions.
from .preprocess import clean_text, normalize_tokens

# Names exported by "from xhate.data import *".
__all__ = [
    "HOF", "NOT", "LABEL_NAMES", "to_binary",
    "load_split", "load_all", "load_dataframe", "read_jsonl", "write_jsonl",
    "clean_text", "normalize_tokens",
]
