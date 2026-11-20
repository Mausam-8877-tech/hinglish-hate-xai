"""Read and write the processed data. THIS is what Members 2, 3 and 4 import.

Typical use (from the repo root):

    from xhate.data import load_split, load_dataframe

    train = load_split("hasoc", "train")        # list of dicts
    df    = load_dataframe("hatexplain", "test") # pandas DataFrame

Every record has the same keys (see data/README.md for the full description):
    id, dataset, source, lang, split, text, raw_text, tokens, label,
    has_rationale, rationale, rationale_soft

Q&A
---
Q: Why JSONL (one JSON object per line) and not CSV?
A: ``tokens`` and ``rationale`` are lists. JSON stores lists natively; CSV would
   force us to encode them as strings and parse them back in every task.

Q: Where does the loader look for files?
A: ``<processed_dir>/<dataset>/<split>.jsonl``. ``processed_dir`` defaults to
   ``data/processed`` relative to the repo root, so it works from notebooks/ too.
"""

# json: to encode / decode one record per line.
import json
# Path: OS-independent file paths (works on Windows and Linux).
from pathlib import Path

# Repo root = two folders above this file (xhate/data/loader.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Default location of the processed data.
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
# The datasets that prepare_data.py can produce.
DATASETS = ("hasoc", "hatexplain")
# The splits every dataset has.
SPLITS = ("train", "val", "test")


def detect_encoding(path):
    """Guess a text file's encoding from its first bytes (its "BOM").

    Q: Why is this needed?
    A: On Windows PowerShell 5.1, ``command > file`` saves the file as UTF-16, so a
       downloaded dataset.json starts with the bytes FF FE and plain UTF-8 reading
       crashes with "can't decode byte 0xff". (This happened to us.)
    """
    # Read only the first 4 bytes.
    with Path(path).open("rb") as f:
        head = f.read(4)
    # FF FE or FE FF = UTF-16 (Python's "utf-16" codec reads the BOM to pick the byte order).
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    # EF BB BF = UTF-8 with a BOM (Notepad / Excel often write this).
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # No BOM: assume normal UTF-8.
    return "utf-8"


def write_jsonl(records, path):
    """Write a list of dicts to ``path``, one JSON object per line (UTF-8)."""
    # Turn the path into a Path object (accepts str or Path).
    path = Path(path)
    # Create the parent folder if it does not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file for writing as UTF-8 (needed for Devanagari text).
    with path.open("w", encoding="utf-8") as f:
        # Write the records one by one.
        for rec in records:
            # ensure_ascii=False keeps Hindi readable instead of \uXXXX escapes.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Read a JSONL file and return a list of dicts."""
    # Open the file as UTF-8.
    with Path(path).open("r", encoding="utf-8") as f:
        # Parse every non-empty line as JSON.
        return [json.loads(line) for line in f if line.strip()]


def split_path(dataset, split, processed_dir=None):
    """Return the path of ``<processed_dir>/<dataset>/<split>.jsonl``."""
    # Use the default folder unless the caller gave one.
    base = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    # Build and return the full path.
    return base / dataset / f"{split}.jsonl"


def load_split(dataset, split, processed_dir=None):
    """Load one split of one dataset as a list of dicts.

    Args:
        dataset: "hasoc" or "hatexplain".
        split: "train", "val" or "test".
        processed_dir: folder written by scripts/prepare_data.py (optional).

    Q: What if the file is missing?
    A: We raise FileNotFoundError with the command that creates it, so a new team
       member knows what to run.
    """
    # Reject unknown dataset names early, with a helpful message.
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")  # stop with a clear message
    # Reject unknown split names early.
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")  # stop with a clear message
    # Work out where the file should be.
    path = split_path(dataset, split, processed_dir)
    # If it does not exist, explain how to create it.
    if not path.exists():
        raise FileNotFoundError(  # the message names the command to run
            f"{path} not found. Run:  python scripts/prepare_data.py --datasets {dataset}"
        )
    # Read and return the records.
    return read_jsonl(path)


def load_all(dataset, processed_dir=None):
    """Load train/val/test of one dataset as ``{"train": [...], "val": [...], "test": [...]}``."""
    # One dict entry per split.
    return {split: load_split(dataset, split, processed_dir) for split in SPLITS}


def load_dataframe(dataset, split, processed_dir=None):
    """Same as ``load_split`` but returns a pandas DataFrame (handy for EDA and sklearn)."""
    # Import pandas here so that importing xhate.data does not require pandas.
    import pandas as pd
    # Build the DataFrame from the list of dicts.
    return pd.DataFrame(load_split(dataset, split, processed_dir))
