"""HASOC (Hindi / Hinglish) loader.

HASOC needs registration (https://hasocfire.github.io), so every member downloads
it into ``data/raw/hasoc/`` themselves. Releases differ in file type and column
names, so this loader AUTO-DETECTS them:

    file types : .csv, .tsv, .xlsx, .xls  (searched recursively, sub-folders allowed)
    text column: text / tweet / tweet_text / post / content / sentence
    label col. : task_1 / task1 / task_a / subtask_a / label / class  (HOF vs NOT)
    id column  : text_id / tweet_id / _id / id / hasoc_id  (generated if missing)
    split      : "test" in the file name -> test, otherwise train

Known layouts this covers: HASOC 2019 Hindi (tsv), 2020 Hindi (xlsx/csv),
2021 Hindi (csv). Files without a recognisable text + label column are skipped
with a warning (e.g. the conversational ICHCL json release).

Q&A
---
Q: Why decide train/test from the file name?
A: HASOC ships its official test set as a separate file whose name contains
   "test". Keeping it as our test set makes our numbers comparable with the
   HASOC papers. Validation is carved out of train later (prepare_data.py).

Q: What about duplicate posts?
A: Exact duplicates (same cleaned text) inside a split are removed, and any
   training post that also appears in the test set is removed from training, so
   the test score is not inflated by leakage.
"""

# re: to find a year like "2019" in the file path.
import re
# warnings: to tell the user about files we could not use.
import warnings
# Path: OS-independent paths.
from pathlib import Path

# pandas: reads csv / tsv / xlsx.
import pandas as pd

# Binary label scheme.
from .labels import to_binary
# Detects UTF-8 / UTF-8-with-BOM / UTF-16 files from their first bytes.
from .loader import detect_encoding
# Raw string -> clean tokens.
from .preprocess import clean_text

# File extensions we try to read.
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
# Possible names of the text column, in order of preference.
TEXT_COLUMNS = ["text", "tweet", "tweet_text", "post", "content", "sentence"]
# Possible names of the task-A (HOF/NOT) label column, in order of preference.
LABEL_COLUMNS = ["task_1", "task1", "task_a", "subtask_a", "label", "class"]
# Possible names of the id column, in order of preference.
ID_COLUMNS = ["text_id", "tweet_id", "_id", "id", "hasoc_id"]


def _find_column(columns, candidates):
    """Return the real column name matching the first candidate (case-insensitive), or None."""
    # Map lower-cased, stripped names to the real names.
    lookup = {str(c).strip().lower(): c for c in columns}
    # Try each candidate in order of preference.
    for cand in candidates:
        # Return the first one that exists.
        if cand in lookup:
            return lookup[cand]  # the real column name, with its original spelling
    # No candidate matched.
    return None


def _read_table(path):
    """Read one csv / tsv / xlsx file into a DataFrame (all columns as strings)."""
    # File extension, lower-cased (".TSV" -> ".tsv").
    ext = path.suffix.lower()
    # Excel files.
    if ext in {".xlsx", ".xls"}:
        # dtype=str keeps long tweet ids from turning into floats.
        return pd.read_excel(path, dtype=str)
    # Tab-separated for .tsv, comma-separated for .csv.
    sep = "\t" if ext == ".tsv" else ","
    # Try the encoding detected from the file's first bytes, then Latin-1 as a last resort.
    # (Detecting the BOM matters: reading a UTF-8-with-BOM file as plain UTF-8 "works" but
    # renames the first column to "﻿text_id", so the id column would not be found.)
    for encoding in (detect_encoding(path), "latin-1"):
        try:
            # Read the file with this encoding.
            return pd.read_csv(path, sep=sep, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            # Wrong encoding: try the next one.
            continue
    # Unreachable in practice (latin-1 accepts every byte), but keeps linters happy.
    raise ValueError(f"Could not decode {path}")


def _guess_split(path):
    """'test' if the file name mentions test, otherwise 'train'."""
    # Look only at the file name, lower-cased.
    return "test" if "test" in path.name.lower() else "train"


def _guess_source(path, raw_dir):
    """A short label for where a row came from, e.g. 'hasoc2019/hindi_dataset.tsv'."""
    # Path relative to data/raw/hasoc/, with forward slashes.
    rel = path.relative_to(raw_dir).as_posix()
    # Look for a year such as 2019 / 2020 / 2021 anywhere in the path.
    year = re.search(r"20\d\d", rel)
    # Prefix the year if we found one.
    return f"hasoc{year.group(0)}/{path.name}" if year else f"hasoc/{rel}"


def read_hasoc_file(path, raw_dir):
    """Read ONE HASOC file and return a list of processed records (or [] if unusable)."""
    # Load the table.
    df = _read_table(path)
    # Find the text column.
    text_col = _find_column(df.columns, TEXT_COLUMNS)
    # Find the HOF/NOT label column.
    label_col = _find_column(df.columns, LABEL_COLUMNS)
    # Find the id column (optional).
    id_col = _find_column(df.columns, ID_COLUMNS)
    # Without text and label we cannot use the file.
    if text_col is None or label_col is None:
        # Tell the user which columns we saw.
        warnings.warn(f"Skipping {path}: no text/label column found in {list(df.columns)}")
        # Nothing usable.
        return []
    # Split from the file name.
    split = _guess_split(path)
    # Source label from the path.
    source = _guess_source(path, raw_dir)
    # Collected records.
    records = []
    # Go through every row of the table.
    for row_idx, row in df.iterrows():
        # The original post text.
        raw_text = row[text_col]
        # The original label string.
        raw_label = row[label_col]
        # Skip rows without text or without a label.
        if pd.isna(raw_text) or pd.isna(raw_label):
            continue  # go to the next row
        # Clean and tokenise the text.
        tokens = clean_text(raw_text)
        # Skip posts that become empty after cleaning (e.g. the text was only punctuation).
        if not tokens:
            continue  # nothing left after cleaning: skip the row
        # Use the file's id if it has one, otherwise build "<source>#<row number>".
        post_id = str(row[id_col]) if id_col is not None and not pd.isna(row[id_col]) else f"{source}#{row_idx}"
        # Build the record in the shared processed format.
        records.append({
            "id": post_id,                    # original or generated id
            "dataset": "hasoc",               # which dataset
            "source": source,                 # which year / file
            "lang": "hi-en",                  # Hindi / Hinglish (code-mixed)
            "split": split,                   # train or test (val is made later)
            "text": " ".join(tokens),         # cleaned text
            "raw_text": str(raw_text),        # original text
            "tokens": tokens,                 # cleaned tokens
            "label": to_binary(raw_label),    # 0 = NOT, 1 = HOF
            "has_rationale": False,           # HASOC has no word rationales
            "rationale": None,                # no rationale available
            "rationale_soft": None,           # no rationale available
        })
    # Return everything we could read from this file.
    return records


def _deduplicate(records):
    """Drop records whose cleaned text was already seen (keeps the first one)."""
    # Cleaned texts we have already kept.
    seen = set()
    # Records we keep.
    kept = []
    # Go through the records in order.
    for rec in records:
        # Skip if this exact cleaned text was already kept.
        if rec["text"] in seen:
            continue  # duplicate: do not keep it
        # Remember the text.
        seen.add(rec["text"])
        # Keep the record.
        kept.append(rec)
    # Return the de-duplicated list.
    return kept


def load_hasoc(raw_dir):
    """Read every HASOC file under ``raw_dir`` and return ``({"train": [...], "test": [...]}, stats)``.

    The "val" split is NOT created here; prepare_data.py carves it out of train
    (stratified), because that needs a random seed and a size.
    """
    # Folder with the HASOC files.
    raw_dir = Path(raw_dir)
    # All supported files, searched recursively, in a stable (sorted) order.
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    # Records per split.
    splits = {"train": [], "test": []}
    # Statistics for the summary.
    stats = {"files_used": [], "files_skipped": []}
    # Read every file.
    for path in files:
        # Read the records from this file.
        records = read_hasoc_file(path, raw_dir)
        # Remember whether it was usable.
        (stats["files_used"] if records else stats["files_skipped"]).append(path.name)
        # Add every record to the split it belongs to.
        for rec in records:
            splits[rec["split"]].append(rec)  # rec["split"] is "train" or "test"
    # Count before de-duplication.
    before = {s: len(r) for s, r in splits.items()}
    # Remove duplicates inside each split.
    splits = {s: _deduplicate(r) for s, r in splits.items()}
    # Cleaned texts present in the test set.
    test_texts = {rec["text"] for rec in splits["test"]}
    # Remove training posts that also appear in test (leakage).
    splits["train"] = [rec for rec in splits["train"] if rec["text"] not in test_texts]
    # How many duplicates / leaked posts were removed per split.
    stats["removed_duplicates"] = {s: before[s] - len(splits[s]) for s in splits}
    # Return the records and the statistics.
    return splits, stats
