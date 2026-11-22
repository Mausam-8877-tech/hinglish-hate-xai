"""Member 4: word-level explanations.

    predictor.py       any trained model -> predict_proba(list of token lists)
    attention.py       [CLS] attention (MuRIL / XLM-R) and BiLSTM attention, per word
    lime_explainer.py  LIME weight per word position (works for every model, also the CNN)
    visualize.py       HTML report that highlights the important words

One entry point for every method:

    from xhate.explain import explain_posts
    scores = explain_posts(predictor, token_lists, "attention")   # or "lime", "random"

Modules that need torch / lime are imported inside the functions, so ``import xhate.explain``
works on a data-only setup (same rule as xhate.data and xhate.models).

Q&A
---
Q: Why is there a "random" method?
A: It is the SANITY BASELINE. Random scores pick random words. A real explanation method must
   beat it on faithfulness (higher comprehensiveness, lower sufficiency) and on plausibility;
   if it does not, the explanation carries no information about the model. Reporting the
   random row makes every other number in the table interpretable.

Q: What does "oriented towards the prediction" mean (``scores_for_prediction``)?
A: ERASER asks: "do the words the explanation calls important support the PREDICTED class?"
   Attention and random scores have no sign, so they are used as they are. LIME scores are
   stored in the HOF direction (positive = pushes towards HOF). If the model predicted NOT,
   the words that support that prediction are the ones with NEGATIVE HOF weight, so we flip
   the sign before picking the top-k words.
"""

# numpy: random scores.
import numpy as np

# The explanation methods this package offers.
METHODS = ("attention", "lime", "random")


def random_scores(token_lists, seed=42):
    """Uniform random score per word: the sanity baseline (see Q&A)."""
    # A seeded generator, so the "random" row is the same in every run.
    rng = np.random.default_rng(seed)
    # One array of random numbers in [0, 1) per post.
    return [rng.random(len(tokens)) for tokens in token_lists]


def explain_posts(predictor, token_lists, method, lime_samples=500, seed=42):
    """Word scores of one method for every post (list of numpy arrays, one per post).

    Args:
        predictor: an ``xhate.explain.predictor.Predictor``.
        token_lists: the posts, as lists of words.
        method: "attention", "lime" or "random".
        lime_samples: perturbed copies per post (LIME only).
        seed: random seed (LIME perturbations / random scores).
    """
    # [CLS] attention or BiLSTM attention.
    if method == "attention":
        # Imported here: needs torch.
        from .attention import attention_scores
        return attention_scores(predictor, token_lists)
    # LIME (works for every model).
    if method == "lime":
        # Imported here: needs the lime package.
        from .lime_explainer import lime_scores
        return lime_scores(predictor, token_lists, num_samples=lime_samples, seed=seed)
    # The random sanity baseline.
    if method == "random":
        return random_scores(token_lists, seed=seed)
    # Anything else is a typo: fail loudly.
    raise ValueError(f"unknown explanation method {method!r}; expected one of {METHODS}")


def scores_for_prediction(scores, method, pred):
    """Orient one post's scores so that HIGH = supports the predicted class (see Q&A).

    Args:
        scores: numpy array, one score per word.
        method: the method that produced them.
        pred: predicted class of the post (0 = NOT, 1 = HOF).
    """
    # LIME is signed in the HOF direction: flip it for a NOT prediction.
    if method == "lime" and int(pred) == 0:
        return -np.asarray(scores, dtype=np.float64)
    # Attention and random scores have no direction.
    return np.asarray(scores, dtype=np.float64)


# Names exported by "from xhate.explain import *".
__all__ = ["METHODS", "explain_posts", "random_scores", "scores_for_prediction"]
