"""Explanation metrics (Member 4): FAITHFULNESS and PLAUSIBILITY.

Two different questions about an explanation (a score per word):

    FAITHFULNESS  "Are the highlighted words really what the MODEL used?"     (no humans needed)
    PLAUSIBILITY  "Are the highlighted words what a HUMAN would highlight?"  (needs rationales)

An explanation can be plausible but unfaithful (it highlights the slur, but the model actually
decided from "<user>"), or faithful but implausible (the model really relies on an odd word).
Our research question is about FAITHFULNESS on Hinglish; plausibility is measured too.

Faithfulness: ERASER (DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP
Models", ACL 2020). Let p(x) be the model's probability of its PREDICTED class on post x,
and r_k the top-k% words of the explanation.

    comprehensiveness_k = p(x) - p(x without r_k)    HIGHER is better
        Delete the important words. If they really were the reason, the model's confidence
        should DROP a lot.
    sufficiency_k       = p(x) - p(only r_k)         LOWER is better
        Keep ONLY the important words. If they are enough, the confidence should stay
        about the same (difference near 0).

Both are averaged over the bins k in {1, 5, 10, 20, 50}% (ERASER's "AOPC" style aggregation),
so the score does not depend on one arbitrary k.

Plausibility (also from ERASER / HateXplain), against the human hard rationale:
    token F1  top-k words of the explanation vs the human-marked words (discrete)
    AUPRC     area under the precision-recall curve of the raw scores (no k needed)

Q&A
---
Q: Why delete words instead of replacing them with <unk> / [MASK]?
A: ERASER deletes. Replacing with a special token keeps the length and position, but the model
   never saw many [MASK]s during fine-tuning, so the input becomes unnatural in another way.
   Deletion is the standard; we use it everywhere (LIME too, mask_string="") so all methods
   are measured with the same perturbation.

Q: Why measure against the PREDICTED class and not the gold label?
A: Faithfulness is about the MODEL's reasoning. If the model is wrong, a faithful explanation
   shows why it was wrong. The gold label plays no role in faithfulness.

Q: How many words is "top 1%" of a 20-word tweet?
A: k = ceil(fraction * n), but at least 1 word. So 1% and 5% are both 1 word, 10% is 2 words,
   20% is 4 words and 50% is 10 words.

Q: How are ties in the scores broken?
A: By position (a stable sort): the earlier word wins. Deterministic and reproducible.

Q: Can sufficiency be negative?
A: Yes: if the model is MORE confident on the kept words alone than on the full post
   (the other words were confusing it). Negative = even better than "sufficient".

Q: For token F1, how many words count as the model's rationale?
A: Humans mark a variable number of words. Our choice: k = the AVERAGE number of human-marked
   words over the evaluated posts (rounded, at least 1). Every post gets the same k, so the
   metric never peeks at a post's own gold count. AUPRC needs no k at all, which is why we
   report both.

Q: Why is token F1 averaged per post (macro) and not pooled over all words?
A: Pooling would let long posts dominate. Averaging per post gives every post the same weight
   (ERASER does the same).

Q: Which posts enter the plausibility scores?
A: Only posts with a human rationale that marks at least one word: HateXplain HOF test posts
   (Member 1's ``has_rationale``) and our own annotated Hinglish HOF posts (data/annotations/).
"""

# math: ceil for the top-k size.
import math

# numpy: score arrays.
import numpy as np

# ERASER's bins: the top 1, 5, 10, 20 and 50 percent of the words.
ERASER_BINS = (0.01, 0.05, 0.10, 0.20, 0.50)


# =============================================================================
# Shared helper
# =============================================================================
def top_k_indices(scores, fraction=None, k=None):
    """Positions of the top-k scores (highest first; ties -> earlier position).

    Give either ``fraction`` (k = ceil(fraction * n), at least 1) or ``k`` directly.
    """
    # Number of words in the post.
    n = len(scores)
    # An empty post has no top words.
    if n == 0:
        return np.zeros(0, dtype=int)  # empty index array
    # k from the fraction (see Q&A), or the given k; never more than n, never less than 1.
    k = math.ceil(fraction * n) if fraction is not None else k
    k = max(1, min(int(k), n))
    # Sort by score descending; "stable" keeps the original order for ties (see Q&A).
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    # The first k positions.
    return order[:k]


# =============================================================================
# Faithfulness (ERASER)
# =============================================================================
def faithfulness(predictor, token_lists, score_lists, preds=None, bins=ERASER_BINS):
    """ERASER comprehensiveness and sufficiency for one explanation method.

    Args:
        predictor: an ``xhate.explain.predictor.Predictor`` (anything with predict_proba).
        token_lists: the posts, as lists of words.
        score_lists: one score array per post, HIGH = supports the predicted class
            (use ``xhate.explain.scores_for_prediction`` for LIME).
        preds: predicted classes (computed here if None).
        bins: fractions of words to delete / keep.

    Returns:
        dict with
          "comprehensiveness", "sufficiency": means over posts of the bin-averaged scores
          "per_bin": {fraction: {"comprehensiveness": .., "sufficiency": ..}}
          "per_post": list of {"comprehensiveness": .., "sufficiency": ..} (bin-averaged)
          "n_posts": number of posts
    """
    # Number of posts.
    n_posts = len(token_lists)
    # Nothing to measure.
    if n_posts == 0:
        return {"comprehensiveness": None, "sufficiency": None, "per_bin": {}, "per_post": [], "n_posts": 0}
    # Probabilities on the FULL posts (one model pass).
    full = predictor.predict_proba(token_lists)
    # Predicted class per post, unless the caller already has it.
    preds = full.argmax(axis=1) if preds is None else np.asarray(preds, dtype=int)
    # p(x): probability of the predicted class on the full post.
    p_full = full[np.arange(n_posts), preds]
    # All perturbed posts go into ONE big list, so the model runs in large batches (fast).
    removed_posts, kept_posts = [], []
    # For every post and every bin, build "post minus top words" and "only top words".
    for tokens, scores in zip(token_lists, score_lists):
        # A wrong-length score array would silently explain the wrong words: check it.
        if len(scores) != len(tokens):
            raise ValueError(f"{len(scores)} scores for {len(tokens)} tokens")
        # One perturbation pair per bin.
        for fraction in bins:
            # The top-k positions for this bin, as a set for fast lookup.
            top = set(top_k_indices(scores, fraction=fraction).tolist())
            # Comprehensiveness input: delete the top words (keep the rest, in order).
            removed_posts.append([tok for i, tok in enumerate(tokens) if i not in top])
            # Sufficiency input: keep ONLY the top words (in their original order).
            kept_posts.append([tok for i, tok in enumerate(tokens) if i in top])
    # Probabilities of every perturbed post: shape (n_posts * n_bins, 2).
    p_removed_all = predictor.predict_proba(removed_posts)
    p_kept_all = predictor.predict_proba(kept_posts)
    # Number of bins.
    n_bins = len(bins)
    # Row index of (post i, bin b) in the big lists is i * n_bins + b.
    rows = np.arange(n_posts * n_bins)
    # The predicted class of each row's post (repeated n_bins times).
    row_pred = np.repeat(preds, n_bins)
    # Probability of the predicted class after deleting / keeping, reshaped to (posts, bins).
    p_removed = p_removed_all[rows, row_pred].reshape(n_posts, n_bins)
    p_kept = p_kept_all[rows, row_pred].reshape(n_posts, n_bins)
    # comprehensiveness_k = p(x) - p(x without r_k), per post and bin.
    comp = p_full[:, None] - p_removed
    # sufficiency_k = p(x) - p(only r_k), per post and bin.
    suff = p_full[:, None] - p_kept
    # Everything as plain Python floats (JSON cannot store numpy types).
    return {
        "comprehensiveness": float(comp.mean()),                       # mean over posts AND bins
        "sufficiency": float(suff.mean()),                             # mean over posts AND bins
        "per_bin": {str(fraction): {"comprehensiveness": float(comp[:, b].mean()),  # one bin
                                    "sufficiency": float(suff[:, b].mean())}
                    for b, fraction in enumerate(bins)},
        "per_post": [{"comprehensiveness": float(comp[i].mean()),       # bin-averaged, post i
                      "sufficiency": float(suff[i].mean())} for i in range(n_posts)],
        "n_posts": n_posts,                                            # how many posts
    }


# =============================================================================
# Plausibility (agreement with human rationales)
# =============================================================================
def token_f1(predicted, gold):
    """F1 between two 0/1 masks over the same words (1 = rationale word).

    Example: predicted [1, 1, 0, 0], gold [0, 1, 1, 0] -> P = 1/2, R = 1/2, F1 = 0.5
    """
    # As boolean arrays.
    predicted, gold = np.asarray(predicted, dtype=bool), np.asarray(gold, dtype=bool)
    # True positives: words marked by both.
    tp = int((predicted & gold).sum())
    # No overlap at all: F1 = 0 (also avoids 0/0).
    if tp == 0:
        return 0.0
    # Precision: share of predicted words that humans also marked.
    precision = tp / predicted.sum()
    # Recall: share of human words that the explanation found.
    recall = tp / gold.sum()
    # Harmonic mean.
    return float(2 * precision * recall / (precision + recall))


def plausibility(score_lists, gold_lists, k=None):
    """Token F1 (top-k) and AUPRC of explanations against human hard rationales.

    Args:
        score_lists: one score array per post (HIGH = important; HOF direction for LIME).
        gold_lists: one 0/1 human rationale per post (same length as its scores).
        k: words taken from each explanation for token F1. None = the average number of
           human-marked words over these posts (see the module Q&A).

    Returns:
        dict with "token_f1", "auprc" (means over posts), "k" and "n_posts".
        Posts whose rationale marks no word are skipped (nothing to agree with).
    """
    # Imported here so importing this module does not require scikit-learn.
    from sklearn.metrics import average_precision_score
    # Keep only posts with at least one human-marked word.
    pairs = [(np.asarray(s, dtype=np.float64), np.asarray(g, dtype=int))
             for s, g in zip(score_lists, gold_lists) if g is not None and sum(g) > 0]
    # Nothing to compare.
    if not pairs:
        return {"token_f1": None, "auprc": None, "k": None, "n_posts": 0}
    # Every pair must cover the same words.
    for scores, gold in pairs:
        if len(scores) != len(gold):
            raise ValueError(f"{len(scores)} scores for a rationale of {len(gold)} tokens")
    # k = average human rationale size (rounded, at least 1) unless given.
    if k is None:
        k = max(1, round(float(np.mean([gold.sum() for _, gold in pairs]))))
    # Per-post token F1 and AUPRC.
    f1s, auprcs = [], []
    for scores, gold in pairs:
        # 0/1 mask of the explanation's top-k words.
        predicted = np.zeros(len(scores), dtype=int)
        predicted[top_k_indices(scores, k=k)] = 1
        # Discrete agreement.
        f1s.append(token_f1(predicted, gold))
        # Ranking agreement: average precision = area under the precision-recall curve.
        auprcs.append(float(average_precision_score(gold, scores)))
    # Means over posts (macro, see Q&A).
    return {"token_f1": float(np.mean(f1s)), "auprc": float(np.mean(auprcs)), "k": int(k), "n_posts": len(pairs)}
