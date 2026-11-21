"""Loss functions.

Member 2 (this file so far): class weights for the imbalanced HOF / NOT data.
Member 3 adds the rationale-supervision loss here.

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
