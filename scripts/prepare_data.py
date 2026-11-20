"""Convert raw datasets into data/processed/.

Usage (from the repo root):

    python scripts/prepare_data.py                         # both datasets
    python scripts/prepare_data.py --datasets hatexplain   # only HateXplain
    python scripts/prepare_data.py --val-size 0.1 --seed 42

Output:
    data/processed/hasoc/{train,val,test}.jsonl
    data/processed/hatexplain/{train,val,test}.jsonl
    data/processed/stats.json      (counts and class balance, for the report)

A dataset whose raw folder is empty is skipped with a message, so you can run the
script before you have HASOC access.

Q&A
---
Q: How is the HASOC validation set made?
A: A stratified ``--val-size`` fraction (default 10%) of HASOC train, with a fixed
   ``--seed``, so every member gets exactly the same split. HASOC's official test
   file stays the test set.

Q: And HateXplain?
A: It already has official train/val/test ids (post_id_divisions.json); we use them.

Q: What if no HASOC file has "test" in its name?
A: We print a warning and also carve a stratified test set out of train, so the
   other tasks still get all three splits.
"""

# argparse: command-line options.
import argparse
# json: to write stats.json.
import json
# sys: to change the import path and the console encoding.
import sys
# Counter: class-balance counts.
from collections import Counter
# Path: OS-independent paths.
from pathlib import Path

# Repo root (this file is repo/scripts/prepare_data.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Make "import xhate" work when the script is run directly.
sys.path.insert(0, str(REPO_ROOT))

# Stratified splitting from scikit-learn.
from sklearn.model_selection import train_test_split  # noqa: E402

# Our dataset readers and the JSONL writer.
from xhate.data.hasoc import load_hasoc  # noqa: E402
from xhate.data.hatexplain import load_hatexplain  # noqa: E402
from xhate.data.loader import write_jsonl  # noqa: E402


def parse_args():
    """Define and read the command-line options."""
    # The argument parser, with the module docstring as help text.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Which datasets to prepare.
    parser.add_argument("--datasets", nargs="+", default=["hasoc", "hatexplain"],
                        choices=["hasoc", "hatexplain"], help="datasets to prepare")
    # Where the raw data lives.
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw",
                        help="folder containing hasoc/ and hatexplain/")
    # Where to write the processed data.
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed",
                        help="output folder")
    # Fraction of HASOC train used for validation.
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="fraction of HASOC train used as validation")
    # Random seed for the split (same seed = same split for everyone).
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Return the parsed options.
    return parser.parse_args()


def stratified_split(records, size, seed):
    """Split ``records`` into (rest, held_out) keeping the HOF/NOT ratio the same."""
    # The labels, used for stratification.
    labels = [r["label"] for r in records]
    # sklearn does the stratified random split.
    rest, held_out = train_test_split(records, test_size=size, random_state=seed, stratify=labels)
    # Return both parts.
    return rest, held_out


def summarize(splits):
    """Count posts and class balance per split (for printing and stats.json)."""
    # One entry per split.
    summary = {}
    # Go through the splits.
    for split, records in splits.items():
        # Count NOT (0) and HOF (1).
        counts = Counter(r["label"] for r in records)
        # Store totals and per-class counts.
        summary[split] = {"n": len(records), "NOT": counts.get(0, 0), "HOF": counts.get(1, 0),
                          "with_rationale": sum(1 for r in records if r["has_rationale"])}
    # Return the summary.
    return summary


def prepare_hasoc(args):
    """Build HASOC train/val/test. Returns (splits, stats) or None if no raw data."""
    # Raw HASOC folder.
    raw = args.raw_dir / "hasoc"
    # Skip if the folder does not exist or has no files except .gitkeep.
    if not raw.exists() or not any(p.is_file() and p.name != ".gitkeep" for p in raw.rglob("*")):
        # Tell the user what to do.
        print(f"[hasoc] no files in {raw} - skipping (download HASOC from hasocfire.github.io)")
        # Nothing produced.
        return None
    # Read and clean all HASOC files.
    splits, stats = load_hasoc(raw)
    # Stop if nothing usable was found.
    if not splits["train"]:
        # Zip files that were never extracted are the most common reason.
        zips = sorted(p.name for p in raw.rglob("*.zip"))
        # Explain why.
        print(f"[hasoc] no usable training rows found in {raw} - skipping")
        # Point at the zips, if any.
        if zips:
            print(f"[hasoc] found zip file(s) {zips}: extract them first (7-Zip; if it asks for "  # how to fix it
                  f"a password, use the one from the HASOC registration), then run this script again")
        # Nothing produced.
        return None
    # No official test file -> carve one out of train.
    if not splits["test"]:
        # Warn: results are then not comparable with HASOC papers.
        print("[hasoc] WARNING: no file with 'test' in its name; carving a test split from train")
        # Stratified split of train into train + test.
        splits["train"], splits["test"] = stratified_split(splits["train"], args.val_size, args.seed)
    # Carve the validation set out of train (stratified).
    splits["train"], splits["val"] = stratified_split(splits["train"], args.val_size, args.seed)
    # Write the correct split name into each record.
    for split, records in splits.items():
        for rec in records:  # every record of this split
            rec["split"] = split  # train / val / test
    # Return the splits in a fixed order, plus stats.
    return {s: splits[s] for s in ("train", "val", "test")}, stats


def prepare_hatexplain(args):
    """Build HateXplain train/val/test. Returns (splits, stats) or None if no raw data."""
    # Raw HateXplain folder.
    raw = args.raw_dir / "hatexplain"
    # Both raw files must exist.
    if not (raw / "dataset.json").exists() or not (raw / "post_id_divisions.json").exists():
        # Tell the user what to do.
        print(f"[hatexplain] dataset.json / post_id_divisions.json missing in {raw} - skipping")
        # Browsers save a second download as "dataset (1).json": list the .json files found.
        found = sorted(p.name for p in raw.glob("*.json")) if raw.exists() else []
        # Only print the hint if there is something to rename.
        if found:
            print(f"[hatexplain] .json files found: {found} - rename them to exactly "  # rename hint
                  f"'dataset.json' and 'post_id_divisions.json'")
        # Nothing produced.
        return None
    # Read, vote and aggregate rationales.
    return load_hatexplain(raw)


def main():
    """Run the preparation for every requested dataset."""
    # Print Hindi / emoji safely on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # avoid UnicodeEncodeError on cp1252 consoles
    # Read the command-line options.
    args = parse_args()
    # Which function prepares which dataset.
    preparers = {"hasoc": prepare_hasoc, "hatexplain": prepare_hatexplain}
    # Stats for every dataset, saved to stats.json.
    all_stats = {}
    # Prepare each requested dataset.
    for name in args.datasets:
        # Run the preparer.
        result = preparers[name](args)
        # Skip datasets with no raw data.
        if result is None:
            continue  # go to the next dataset
        # Unpack the splits and the stats.
        splits, stats = result
        # Write one JSONL file per split.
        for split, records in splits.items():
            # Output path, e.g. data/processed/hasoc/train.jsonl.
            path = args.out_dir / name / f"{split}.jsonl"
            # Write it.
            write_jsonl(records, path)
        # Count posts and classes.
        summary = summarize(splits)
        # Keep the stats for stats.json.
        all_stats[name] = {"splits": summary, "details": stats}
        # Print a short summary.
        print(f"[{name}] wrote {args.out_dir / name}")
        # Print the details (dropped posts, skipped files, ...).
        print(f"[{name}] details: {stats}")
        # One line per split.
        for split, s in summary.items():
            print(f"    {split:5s}: n={s['n']:6d}  NOT={s['NOT']:6d}  HOF={s['HOF']:6d}  "  # size and class counts
                  f"with_rationale={s['with_rationale']}")
    # Save stats.json if anything was produced.
    if all_stats:
        # Make sure the folder exists.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Path of the stats file.
        stats_path = args.out_dir / "stats.json"
        # Keep the stats of datasets NOT prepared in this run (e.g. only --datasets hasoc).
        old = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        # New stats replace old ones for the datasets prepared now.
        old.update(all_stats)
        # Write the merged stats file.
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(old, f, indent=2)  # indent=2 keeps it human-readable
    else:
        # Nothing was prepared at all.
        print("Nothing was prepared. Put the raw files in data/raw/ (see data/README.md).")


# Run main() only when executed as a script (not when imported).
if __name__ == "__main__":
    main()  # start the script
