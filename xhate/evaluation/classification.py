"""Classification metrics (accuracy, macro-F1, ...), Member 4.

    from xhate.evaluation.classification import compute_metrics
    m = compute_metrics(labels, preds)    # accuracy, macro_f1, per_class, confusion_matrix

Q&A
---
Q: Why does this file only re-export Member 2's function?
A: Member 2 wrote ``compute_metrics`` in xhate/training/trainer.py because early stopping
   needed it before this task started (their docstring says Member 4 may take it over). Every
   reported number so far (README tables, 21 transformer runs) comes from that function.
   Writing a second, slightly different implementation here would risk two "macro-F1"s in
   the report. Re-exporting keeps ONE implementation and gives it its natural home.

Q: What does it compute?
A: accuracy, macro-F1 (the mean of the F1 of NOT and HOF: the official HASOC metric),
   precision / recall / F1 / support per class, and the confusion matrix
   [[NOT->NOT, NOT->HOF], [HOF->NOT, HOF->HOF]] (rows = gold, columns = predicted).
"""

# The single implementation used everywhere in the project (see Q&A).
from xhate.training.trainer import compute_metrics

# Names exported by "from xhate.evaluation.classification import *".
__all__ = ["compute_metrics"]
