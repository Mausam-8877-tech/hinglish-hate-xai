"""Our own Hinglish rationale annotations (Member 4).

HASOC has labels but no rationales. To measure plausibility on CODE-MIXED text we annotate
~250 HOF posts of the HASOC test set ourselves (scripts/sample_for_annotation.py).

The committed file ``data/annotations/hasoc_rationales.jsonl`` has one line per post:

    {"id": "hasoc_en_123", "n_tokens": 9, "rationale": [0, 0, 1, 1, 0, 0, 0, 0, 0],
     "annotator": "member2"}

    from xhate.evaluation.annotations import load_annotations, attach_annotations
    gold = load_annotations()                       # {post id: record}
    attach_annotations(test_records, gold)          # adds rationale / has_rationale to HASOC posts

Q&A
---
Q: Why does the file contain NO post text?
A: The HASOC licence does not allow redistributing the data, and data/annotations/ is committed
   to GitHub. Ids + 0/1 masks are our own work; the text is re-joined from each member's own
   processed HASOC copy by id.

Q: Why store ``n_tokens``?
A: The mask is per token of Member 1's cleaned post. If preprocessing ever changes (a new
   spelling rule, a different emoji name), the token count may change and the mask would point
   at the wrong words. We check ``n_tokens`` against the current tokens and refuse to use a
   mask that no longer fits, instead of silently mis-aligning it.
"""

# json: one JSON object per line.
import json
# warnings: report annotations that no longer fit their post.
import warnings
# Path: OS-independent paths.
from pathlib import Path

# Repo root (this file is repo/xhate/evaluation/annotations.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Where the committed annotations live.
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "annotations" / "hasoc_rationales.jsonl"


def load_annotations(path=None):
    """Read the annotations file into ``{post id: record}`` (empty dict if it does not exist)."""
    # Default location unless a path is given.
    path = Path(path) if path else DEFAULT_ANNOTATIONS
    # Not annotated yet: nothing to load.
    if not path.exists():
        return {}
    # One record per non-empty line, keyed by the post id.
    with path.open("r", encoding="utf-8") as f:
        return {rec["id"]: rec for rec in (json.loads(line) for line in f if line.strip())}


def attach_annotations(records, annotations):
    """Copy human rationales onto matching processed records (in place). Returns how many matched.

    Sets ``rationale`` (0/1 list), ``has_rationale`` (True if a word is marked) and
    ``annotator`` on every record whose id is annotated and whose token count still fits.
    """
    # How many records received an annotation.
    matched = 0
    # Go through the processed posts.
    for rec in records:
        # The annotation for this post, if any.
        ann = annotations.get(rec["id"])
        # Not annotated: leave the record as it is.
        if ann is None:
            continue
        # Token count changed since annotation: the mask would point at the wrong words (see Q&A).
        if ann["n_tokens"] != len(rec["tokens"]) or len(ann["rationale"]) != len(rec["tokens"]):
            warnings.warn(f"annotation for {rec['id']} has {ann['n_tokens']} tokens but the post now has "
                          f"{len(rec['tokens'])}: skipped (was the preprocessing changed?)")
            continue  # do not use a misaligned mask
        # Copy the human mask.
        rec["rationale"] = [int(x) for x in ann["rationale"]]
        # A post with no marked word has no rationale to compare against.
        rec["has_rationale"] = sum(rec["rationale"]) > 0
        # Who annotated it (for agreement analysis later).
        rec["annotator"] = ann.get("annotator")
        # Count it.
        matched += 1
    # Tell the caller how many posts got a rationale.
    return matched
