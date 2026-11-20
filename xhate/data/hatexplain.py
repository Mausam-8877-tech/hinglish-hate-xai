"""HateXplain (English, with rationales) loader.

Raw files (download into data/raw/hatexplain/):
    dataset.json            {post_id: {"annotators": [...], "rationales": [...], "post_tokens": [...]}}
    post_id_divisions.json  {"train": [post_id, ...], "val": [...], "test": [...]}

What this module does:
    1. Majority vote over the 3 annotators' labels (hatespeech / offensive / normal).
       Posts where all 3 annotators disagree are DROPPED (as in the HateXplain paper).
    2. Map the majority label to our binary scheme (NOT / HOF), see labels.py.
    3. Aggregate the annotators' word rationales into one per-token target:
           rationale_soft[i] = fraction of rationale lists that marked token i
           rationale[i]      = 1 if rationale_soft[i] >= 0.5 else 0
       NOT posts get all-zero rationales.
    4. Normalise the tokens 1-to-1 (the token count NEVER changes).

Q&A
---
Q: Why do some posts have only 2 rationale lists when there are 3 annotators?
A: In HateXplain, only annotators who labelled a post hateful/offensive marked
   rationale words. So the soft score is the fraction of THOSE lists, which is
   also how the HateXplain authors build their ground-truth attention.

Q: What if a rationale list has a different length from the tokens?
A: That happens for 1 post in the release. We skip that broken list (and count
   it), rather than guessing how to align it.

Q: What is ``has_rationale``?
A: True when the post is HOF and at least one valid rationale list exists.
   Member 3 should only apply the rationale loss where it is True.
"""

# json: the raw files are JSON.
import json
# Counter: counts labels for the majority vote.
from collections import Counter
# Path: OS-independent paths.
from pathlib import Path

# Our binary label scheme.
from .labels import HOF, NOT, to_binary
# Detects UTF-8 / UTF-8-with-BOM / UTF-16 files (Windows PowerShell saves UTF-16).
from .loader import detect_encoding
# 1-to-1 token normalisation (keeps rationale alignment).
from .preprocess import normalize_tokens

# Names of the two raw files.
DATASET_FILE = "dataset.json"
# File with the official train / val / test post ids.
DIVISIONS_FILE = "post_id_divisions.json"


def majority_label(annotator_labels):
    """Return the label chosen by most annotators, or None if there is no majority.

    Example: ["offensive", "offensive", "normal"] -> "offensive"
             ["offensive", "hatespeech", "normal"] -> None
    """
    # Count how often each label was chosen.
    counts = Counter(annotator_labels)
    # Take the most frequent label and its count.
    label, count = counts.most_common(1)[0]
    # A majority means MORE than half of the annotators agree.
    return label if count > len(annotator_labels) / 2 else None


def aggregate_rationales(rationale_lists, n_tokens):
    """Combine several 0/1 rationale lists into (hard, soft, n_used).

    Args:
        rationale_lists: list of lists, each a 0/1 mask over the tokens.
        n_tokens: number of tokens in the post.

    Returns:
        hard:   list of 0/1 ints (1 where at least half the lists marked the token)
        soft:   list of floats in [0, 1] (fraction of lists that marked the token)
        n_used: how many lists were valid (right length) and used
    """
    # Keep only lists whose length matches the number of tokens.
    valid = [r for r in rationale_lists if len(r) == n_tokens]
    # With no valid list, every token gets 0.
    if not valid:
        return [0] * n_tokens, [0.0] * n_tokens, 0  # all-zero hard and soft targets, 0 lists used
    # For every token position i, average the marks across the valid lists.
    soft = [sum(int(r[i]) for r in valid) / len(valid) for i in range(n_tokens)]
    # Round to 4 decimals to keep the files small (e.g. 0.6667).
    soft = [round(s, 4) for s in soft]
    # Hard target: 1 if at least half the lists marked it.
    hard = [1 if s >= 0.5 else 0 for s in soft]
    # Return all three values.
    return hard, soft, len(valid)


def process_post(post_id, post, split):
    """Turn ONE raw HateXplain post into our processed record (or None if dropped).

    Returns:
        (record or None, info dict)  where info says why a post was dropped and
        how many rationale lists were skipped.
    """
    # Bookkeeping for the summary printed by prepare_data.py.
    info = {"dropped_no_majority": 0, "bad_rationale_lists": 0}
    # The 3 annotator labels, e.g. ["normal", "offensive", "offensive"].
    labels = [a["label"] for a in post["annotators"]]
    # Majority vote on the ORIGINAL 3 classes.
    majority = majority_label(labels)
    # No majority -> drop the post.
    if majority is None:
        # Record the reason.
        info["dropped_no_majority"] = 1
        # Nothing to return.
        return None, info
    # Map the majority label to 0 (NOT) or 1 (HOF).
    label = to_binary(majority)
    # The original tokens, exactly as provided by HateXplain.
    raw_tokens = post["post_tokens"]
    # Normalise the tokens 1-to-1 (length preserved).
    tokens = normalize_tokens(raw_tokens)
    # Number of tokens (same for raw and normalised).
    n = len(tokens)
    # All rationale lists provided for this post (may be empty).
    rationale_lists = post.get("rationales", [])
    # Count lists that will be skipped because of a length mismatch.
    info["bad_rationale_lists"] = sum(1 for r in rationale_lists if len(r) != n)
    # HOF posts: aggregate the rationales.
    if label == HOF:
        # Combine the lists into hard and soft targets.
        hard, soft, n_used = aggregate_rationales(rationale_lists, n)
    else:
        # NOT posts: no word is responsible for hate, so all zeros.
        hard, soft, n_used = [0] * n, [0.0] * n, 0
    # Build the record in the shared processed format.
    record = {
        "id": post_id,                              # HateXplain post id
        "dataset": "hatexplain",                    # which dataset this came from
        "source": "hatexplain",                     # raw file / release
        "lang": "en",                               # language of the post
        "split": split,                             # train / val / test
        "text": " ".join(tokens),                   # cleaned text
        "raw_text": " ".join(raw_tokens),           # original text
        "tokens": tokens,                           # cleaned tokens
        "label": label,                             # 0 = NOT, 1 = HOF
        "has_rationale": label == HOF and n_used > 0,  # True if rationale is real
        "rationale": hard,                          # 0/1 per token
        "rationale_soft": soft,                     # fraction per token
    }
    # Return the record and the bookkeeping info.
    return record, info


def load_hatexplain(raw_dir):
    """Read the raw HateXplain files and return ``({split: [records]}, stats)``.

    The official ``post_id_divisions.json`` decides the train / val / test split.
    """
    # Folder containing the two raw files.
    raw_dir = Path(raw_dir)
    # Open and parse dataset.json, in whatever encoding it was saved.
    with (raw_dir / DATASET_FILE).open(encoding=detect_encoding(raw_dir / DATASET_FILE)) as f:
        # {post_id: post}
        posts = json.load(f)
    # Open and parse post_id_divisions.json, in whatever encoding it was saved.
    with (raw_dir / DIVISIONS_FILE).open(encoding=detect_encoding(raw_dir / DIVISIONS_FILE)) as f:
        # {"train": [...], "val": [...], "test": [...]}
        divisions = json.load(f)
    # Output: one list of records per split.
    splits = {"train": [], "val": [], "test": []}
    # Running totals for the summary.
    stats = {"raw_posts": len(posts), "dropped_no_majority": 0,
             "bad_rationale_lists": 0, "missing_ids": 0}
    # Go through the official splits.
    for split, ids in divisions.items():
        # Go through every post id listed for this split.
        for post_id in ids:
            # Some ids could be missing from dataset.json; count and skip them.
            if post_id not in posts:
                stats["missing_ids"] += 1  # count the missing id
                continue  # and skip it
            # Process the post.
            record, info = process_post(post_id, posts[post_id], split)
            # Add up the bookkeeping counters.
            for key, value in info.items():
                stats[key] += value  # add this post's counts to the totals
            # Keep the record unless it was dropped.
            if record is not None:
                splits[split].append(record)  # add it to its official split
    # All post ids that appear in some official split.
    listed = {pid for ids in divisions.values() for pid in ids}
    # Posts the HateXplain authors left out of every split (mostly no-majority posts).
    stats["not_in_divisions"] = sum(1 for pid in posts if pid not in listed)
    # Return the split records and the statistics.
    return splits, stats
