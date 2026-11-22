"""Sample Hinglish posts for our own rationale annotation, and collect the result (Member 4).

Step 1, make the sheets (one per member), from the repo root:

    python scripts/sample_for_annotation.py                  # 250 HOF test posts -> 4 sheets
    # -> data/annotations/sheets/member1.csv ... member4.csv   (git-ignored: they contain HASOC text)

Step 2, every member opens THEIR sheet in Excel / Google Sheets / LibreOffice and fills the
"rationale" column with the NUMBERS of the words that make the post hateful/offensive,
e.g. "2 3" (see data/annotations/README.md for the guidelines). Save as CSV (UTF-8).

Step 3, collect all sheets into the committed file:

    python scripts/sample_for_annotation.py --collect
    # -> data/annotations/hasoc_rationales.jsonl   (ids + 0/1 masks only, NO text: commit this)

Q&A
---
Q: Why only HOF posts, and why from the TEST set?
A: A rationale explains why a post is hateful/offensive; NOT posts have no such words (same
   rule as HateXplain and Member 3's loss). The explanations are evaluated on the test set, so
   that is where we need human rationales. Gold labels are used, not a model's predictions, so
   the sample does not favour any model.

Q: Why a random sample with a fixed seed?
A: Random = no hand-picking of easy posts. Fixed seed = every member who runs this gets the
   same 250 posts and the same split into sheets.

Q: Why word NUMBERS instead of asking people to retype words?
A: Retyping Hinglish / Devanagari invites spelling differences ("nhi" vs "nahi"), and the same
   word can appear twice in a post. A number points at exactly one position of Member 1's
   tokens, which is the format every metric uses.

Q: Why are the sheets git-ignored but the collected file committed?
A: The sheets contain the post text, and the HASOC licence forbids redistributing it. The
   collected file only has post ids and 0/1 masks: our own work, safe to share.

Q: Why CSV with "utf-8-sig"?
A: Excel on Windows only recognises a CSV as UTF-8 if it starts with a BOM; otherwise
   Devanagari shows up as garbage. "utf-8-sig" writes that BOM.

Q: How is ~250 split over 4 members?
A: Round-robin (post 1 -> member 1, post 2 -> member 2, ...), so the sheets are 63 / 63 / 62 / 62.
"""

# argparse: command-line options.
import argparse
# csv: read / write the sheets.
import csv
# json: write the collected annotations.
import json
# random: seeded sample.
import random
# re: split the "rationale" cell into numbers.
import re
# sys: import path and console encoding.
import sys
# Path: OS-independent paths.
from pathlib import Path

# Repo root (this file is repo/scripts/sample_for_annotation.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Make "import xhate" work when the script is run directly.
sys.path.insert(0, str(REPO_ROOT))

# Member 1's loader and the label id of HOF.
from xhate.data import HOF, load_split  # noqa: E402
# Where the committed annotations go.
from xhate.evaluation.annotations import DEFAULT_ANNOTATIONS  # noqa: E402

# Folder of the per-member sheets (git-ignored).
DEFAULT_SHEETS = REPO_ROOT / "data" / "annotations" / "sheets"
# Columns of a sheet, in order.
COLUMNS = ["id", "text", "numbered_words", "rationale", "comment"]


def parse_args(argv=None):
    """Define and read the command-line options (argv=None -> the real command line)."""
    # Parser with this docstring as --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mode: collect filled sheets instead of making new ones.
    parser.add_argument("--collect", action="store_true", help="collect filled sheets into the jsonl file")
    # Sampling options.
    parser.add_argument("--n", type=int, default=250, help="number of HOF posts to sample")
    parser.add_argument("--members", type=int, default=4, help="number of sheets (one per member)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Folders / files.
    parser.add_argument("--sheets-dir", type=Path, default=DEFAULT_SHEETS, help="folder of the sheets")
    parser.add_argument("--out", type=Path, default=DEFAULT_ANNOTATIONS, help="collected annotations (jsonl)")
    parser.add_argument("--processed-dir", type=Path, default=None, help="processed data (default data/processed)")
    # Do not overwrite filled sheets by accident.
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    # Read the options.
    return parser.parse_args(argv)


def numbered(tokens):
    """ "0=tum | 1=log | 2=gaddar" : every word with its number, for the annotator."""
    # Number each word and join with a visible separator.
    return " | ".join(f"{i}={tok}" for i, tok in enumerate(tokens))


def make_sheets(args):
    """Sample HOF test posts and write one CSV sheet per member. Returns the sheet paths."""
    # Never overwrite sheets that people may already have filled in.
    existing = sorted(args.sheets_dir.glob("member*.csv")) if args.sheets_dir.exists() else []
    if existing and not args.force:
        raise SystemExit(f"sheets already exist in {args.sheets_dir} ({len(existing)} files); "
                         "use --collect to read them, or --force to overwrite")
    # HASOC test posts whose GOLD label is HOF (see Q&A).
    hof = [rec for rec in load_split("hasoc", "test", args.processed_dir) if rec["label"] == HOF]
    # Seeded random sample (all posts if there are fewer than --n).
    sample = random.Random(args.seed).sample(hof, min(args.n, len(hof)))
    # Make the folder.
    args.sheets_dir.mkdir(parents=True, exist_ok=True)
    # Written files.
    paths = []
    # One sheet per member.
    for m in range(args.members):
        # Round-robin share of this member (see Q&A).
        mine = sample[m:: args.members]
        # member1.csv, member2.csv, ...
        path = args.sheets_dir / f"member{m + 1}.csv"
        # utf-8-sig so Excel shows Devanagari correctly (see Q&A); newline="" as the csv module wants.
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            # One row per post.
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for rec in mine:
                writer.writerow({"id": rec["id"], "text": rec["text"], "numbered_words": numbered(rec["tokens"]),
                                 "rationale": "", "comment": ""})  # the two columns the annotator fills
        # Remember the file.
        paths.append(path)
        print(f"[annotate] {path}: {len(mine)} posts")
    # Done.
    return paths


def parse_rationale(cell, n_tokens):
    """ "2 3" / "2,3" / "2-4" -> 0/1 mask of length n_tokens. Raises ValueError for bad numbers.

    "2-4" is a range (words 2, 3 and 4), handy for multi-word insults.
    """
    # Start with no word marked.
    mask = [0] * n_tokens
    # Pieces separated by spaces, commas or semicolons.
    for piece in re.split(r"[\s,;]+", (cell or "").strip()):
        # Skip empty pieces (e.g. an empty cell).
        if not piece:
            continue
        # A range "a-b" or a single number "a".
        start, _, end = piece.partition("-")
        # int() raises ValueError for anything that is not a number.
        first, last = int(start), int(end) if end else int(start)
        # Every number must point at a real word.
        if not 0 <= first <= last < n_tokens:
            raise ValueError(f"word number {piece!r} is outside 0..{n_tokens - 1}")
        # Mark the word(s).
        for i in range(first, last + 1):
            mask[i] = 1
    # The finished mask.
    return mask


def collect(args):
    """Read every filled sheet and write the committed jsonl. Returns the collected records."""
    # Current processed posts, by id, to get each post's token count.
    by_id = {rec["id"]: rec for rec in load_split("hasoc", "test", args.processed_dir)}
    # All sheets, in a stable order.
    sheets = sorted(args.sheets_dir.glob("member*.csv"))
    # Nothing to collect.
    if not sheets:
        raise SystemExit(f"no sheets found in {args.sheets_dir}; run without --collect first")
    # Collected records and problems found.
    collected, problems = [], []
    # Read each member's sheet.
    for path in sheets:
        # "member2" from "member2.csv".
        annotator = path.stem
        # utf-8-sig also reads files without a BOM (e.g. saved from Google Sheets).
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                # The processed post this row refers to.
                rec = by_id.get(row["id"])
                # Unknown id (processed data changed?): report and skip.
                if rec is None:
                    problems.append(f"{path.name}: id {row['id']} not in the HASOC test set")
                    continue
                # Not annotated yet: skip silently (it is simply not done).
                if not (row.get("rationale") or "").strip():
                    continue
                # Turn the numbers into a mask; collect mistakes instead of stopping at the first.
                try:
                    mask = parse_rationale(row["rationale"], len(rec["tokens"]))
                except ValueError as err:
                    problems.append(f"{path.name}: id {row['id']}: {err}")
                    continue
                # Ids + masks only, NO text (licence, see Q&A).
                collected.append({"id": rec["id"], "n_tokens": len(rec["tokens"]), "rationale": mask,
                                  "annotator": annotator})
    # Show every problem so the annotators can fix their sheets.
    for problem in problems:
        print(f"[annotate] PROBLEM {problem}")
    # Write the committed file, one record per line.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in collected:
            f.write(json.dumps(rec) + "\n")
    # Summary.
    print(f"[annotate] wrote {len(collected)} annotated posts to {args.out} ({len(problems)} problems)")
    return collected


def main(argv=None):
    """Make the sheets, or collect them with --collect."""
    # Print Hindi safely on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Read the options.
    args = parse_args(argv)
    # Pick the mode.
    return collect(args) if args.collect else make_sheets(args)


# Run main() only when executed as a script (not when imported by a test).
if __name__ == "__main__":
    main()
