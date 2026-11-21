"""Loss functions.

Member 2: class weights for the imbalanced HOF / NOT data.
Member 3: the rationale-supervision loss (``rationale_loss``, ``rationale_supervised_loss_fn``).

Q&A
---
Q: Why class weights?
A: HASOC is imbalanced (more posts of one class than the other). With a plain
   loss, a model can score well on accuracy by mostly predicting the bigger
   class. Weighting each class by the inverse of its frequency makes a mistake
   on the rare class cost more, which helps macro-F1 (the metric we select on).

Q: What formula do we use?
A: The "balanced" formula from scikit-learn:  weight_c = N / (C * count_c),
   where N = number of training posts and C = number of classes.
   Example: 600 NOT + 400 HOF -> NOT: 1000/(2*600) = 0.83, HOF: 1000/(2*400) = 1.25.
   A perfectly balanced dataset gets weight 1.0 for every class.
"""

# Counter: counts posts per class.
from collections import Counter

# torch: tensors.
import torch


def class_weights(labels, num_classes=2):
    """Return a float tensor of "balanced" class weights, one per class.

    Args:
        labels: list of int labels of the TRAINING posts.
        num_classes: number of classes (2 for NOT / HOF).

    Q: What if a class does not appear at all?
    A: It gets weight 0 (instead of dividing by zero); it cannot occur in the
       training labels anyway, so its weight is never used.
    """
    # How many training posts per class.
    counts = Counter(int(label) for label in labels)
    # Total number of posts.
    total = sum(counts.values())
    # N / (C * count_c) for every class (0 for a missing class).
    weights = [total / (num_classes * counts[c]) if counts[c] else 0.0 for c in range(num_classes)]
    # Return as a float tensor (what nn.CrossEntropyLoss expects).
    return torch.tensor(weights, dtype=torch.float)


# =============================================================================
# Member 3: rationale supervision
# =============================================================================
#
#     loss = CE(logits, label) + lambda * rationale_loss(cls_attention, rationale)
#
# Q&A
# ---
# Q: What is a rationale?
# A: The words a human annotator marked as the REASON a post is hateful/offensive.
#    HateXplain has them for its English HOF posts; HASOC has none.
#
# Q: What does the rationale loss do?
# A: It pushes the model's [CLS] attention (which tokens the classifier "looks at")
#    towards the human rationale. The hope: the model predicts HOF for the RIGHT reasons,
#    and its attention becomes a more faithful explanation, also on Hinglish posts.
#
# Q: How exactly is it computed? (HateXplain, Mathew et al. 2021)
# A: For one post with sub-word positions i (only real word positions, rationale_mask = 1):
#      target_i = r_i / sum_j r_j            human rationale as a probability distribution
#      attn_i   = a_i / sum_j a_j            model attention, re-normalised over the same positions
#      loss     = - sum_i target_i * log(attn_i)          (cross-entropy between distributions)
#    It is smallest when attn == target. The batch loss is the mean over posts that have a
#    rationale; posts without one contribute nothing.
#
# Q: Why re-normalise the attention over the word positions?
# A: [CLS] also attends to itself, to [SEP] and (in training) its row is scaled by attention
#    dropout. Those positions have no human target, so we compare the two distributions only
#    where both are defined.
#
# Q: Why cross-entropy and not MSE or KL divergence?
# A: Cross-entropy is what HateXplain used, which keeps our results comparable. KL(target||attn)
#    differs from it only by the entropy of the target, a constant, so it gives the same
#    gradients. MSE on probabilities gives weak gradients when values are small (most
#    attention weights are ~0.01), so it learns slowly.
#
# Q: Why apply it ONLY to posts with a rationale?
# A: A NOT post has no "responsible words" (all zeros -> target undefined: 0/0), and HASOC
#    posts were never annotated. Treating "no annotation" as "no word matters" would teach
#    the model something false. The issue asks for exactly this.
#
# Q: What does lambda control?
# A: The trade-off between classifying correctly (CE) and attending like a human (rationale).
#    lambda = 0 -> plain CE (our "more data only" control run); large lambda (10) -> attention
#    dominates and accuracy may drop. That is why we sweep 0.01, 0.1, 1, 10.
#
# Q: Why do we need eps?
# A: log(0) = -inf. Adding a tiny eps (1e-8) avoids NaN losses when an attention weight is 0.


def rationale_loss(cls_attention, rationale, rationale_mask, eps=1e-8):
    """Cross-entropy between the human rationale and the model's [CLS] attention.

    Args:
        cls_attention: (B, L) attention of [CLS] over the sub-words (from the model).
        rationale: (B, L) float rationale per sub-word (soft in [0, 1] or hard 0/1).
        rationale_mask: (B, L) 1 where a rationale target exists, 0 elsewhere
            (special tokens, padding, and every position of a post without rationale).
        eps: small number that keeps divisions and log() finite.

    Returns:
        Scalar tensor: mean loss over posts that have a rationale (0 if the batch has none).
    """
    # 0/1 mask as float, so it can multiply tensors.
    mask = rationale_mask.float()
    # Human rationale restricted to valid positions.
    target = rationale * mask
    # Total rationale mass per post: (B,).
    target_sum = target.sum(dim=-1)
    # A post counts only if some word is marked (a rationale that is all 0 has no distribution).
    has_rationale = target_sum > 0
    # No post in this batch has a rationale (e.g. a pure HASOC batch): contribute 0.
    if not has_rationale.any():
        # "* 0.0" keeps the result attached to the graph (and on the right device) so that
        # adding it to the CE loss and calling backward() still works.
        return cls_attention.sum() * 0.0
    # Normalise the target to sum to 1 per post (clamp avoids 0/0 for the posts we drop).
    target = target / target_sum.clamp_min(eps).unsqueeze(-1)
    # Model attention on valid positions only.
    attn = cls_attention * mask
    # Re-normalise the attention to sum to 1 over those positions (see Q&A).
    attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
    # Cross-entropy per post: -sum_i target_i * log(attn_i)  -> (B,).
    per_post = -(target * torch.log(attn + eps)).sum(dim=-1)
    # Mean over posts with a rationale only.
    return per_post[has_rationale].mean()


def rationale_supervised_loss_fn(model, lam=1.0, weight=None, ce_loss_fn=None):
    """Return ``loss_fn(logits, batch)`` = CE + lam * rationale_loss, for Member 2's ``fit``.

    Args:
        model: the TransformerClassifier being trained. Its ``last_cls_attention``
            (filled by the forward pass that produced ``logits``) is read here.
        lam: lambda, the weight of the rationale term (0 = plain cross-entropy).
        weight: class weights for the cross-entropy (or None).
        ce_loss_fn: the cross-entropy part; default = ``trainer.default_loss_fn(weight)``.

    Q: Why does the loss function need the model?
    A: ``fit`` only passes the logits and the batch to the loss. The attention lives on the
       model (see transformer.py), so the closure keeps a reference to it.
    """
    # Imported here to avoid a circular import (trainer.py is also imported by train.py).
    from .trainer import default_loss_fn
    # Classification part: (class-weighted) cross-entropy on batch["label"].
    ce_loss_fn = ce_loss_fn or default_loss_fn(weight)

    def loss_fn(logits, batch):
        """CE on the labels + lam * rationale loss on the attention of the same forward pass."""
        # Cross-entropy for every post.
        loss = ce_loss_fn(logits, batch)
        # Skip the extra work when lambda is 0 (the control run): identical to plain CE.
        if lam:
            # Rationale term (0 for batches without any rationale).
            loss = loss + lam * rationale_loss(model.last_cls_attention,
                                               batch["rationale"], batch["rationale_mask"])
        # Scalar loss for backward().
        return loss

    # Return the closure.
    return loss_fn
