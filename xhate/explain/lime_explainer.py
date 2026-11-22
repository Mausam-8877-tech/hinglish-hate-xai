"""LIME word importance (Member 4): one score per word POSITION.

    from xhate.explain.lime_explainer import lime_scores
    scores = lime_scores(predictor, [["tum", "log", "gaddar", "ho"]], num_samples=500)
    # -> [array([0.02, -0.01, 0.61, 0.00])]   > 0 pushes towards HOF, < 0 towards NOT

How LIME works (Ribeiro et al., "Why Should I Trust You?", KDD 2016):
    1. Make ``num_samples`` perturbed copies of the post by deleting random subsets of words.
    2. Ask the model for P(HOF) on every copy.
    3. Weight each copy by how similar it is to the original (fewer deleted words = closer).
    4. Fit a weighted LINEAR model:  P(HOF) ~ b + sum_i w_i * [word i is present].
    5. w_i is word i's importance: how much having word i raises P(HOF) near this post.

Q&A
---
Q: Why ``split_expression=r"\\s+"`` and ``bow=False``?
A: * LIME's default splitter is the regex ``\\W+``. Python's ``\\W`` treats Hindi vowel signs
     (matras) as non-letters, so "हरामी" would be explained as letters "ह", "र", "म" (see
     data/README.md). Our text is ``" ".join(tokens)``, so splitting on spaces gives back
     exactly Member 1's tokens.
   * bow=True would treat every occurrence of a word as ONE feature ("bag of words").
     bow=False gives one feature per POSITION, so the scores line up 1-to-1 with the tokens
     and with the human rationales (which are per position).

Q: Why ``mask_string=""``?
A: With bow=False LIME replaces a "removed" word with the placeholder "UNKWORDZ" by default.
   That fake word would be fed to the model (an unknown word / odd sub-words). With "" the word
   is really DELETED, which is also how our ERASER faithfulness metrics remove words. Same
   kind of perturbation in both = a fair comparison.

Q: Why explain the HOF class (label 1) only?
A: With 2 classes P(NOT) = 1 - P(HOF), so LIME's weights for NOT are just the negatives of the
   weights for HOF. We store the HOF direction: positive = "evidence for hate". For the
   faithfulness metrics the scores are flipped when the model predicted NOT (see
   ``scores_for_prediction`` in xhate/explain/__init__.py).

Q: How many samples?
A: More samples = a more stable linear fit, but every sample is one model call. LIME's default
   is 5000. Tweets are short (~20 words), so a few hundred to 1000 samples already give stable
   rankings; we default to 500 and make it a command-line option (--lime-samples).

Q: Why ``random_state``?
A: The perturbations are random. A fixed seed makes the explanations reproducible.

Q: What is the ``kernel_width``?
A: The similarity weight is exp(-d^2 / width^2), where d is the cosine distance between the
   copy and the original. We keep LIME's default (25).
"""

# numpy: one score array per post.
import numpy as np

# Label names for LIME's output ("NOT", "HOF").
from xhate.data.labels import HOF, LABEL_NAMES


def lime_scores(predictor, token_lists, num_samples=500, seed=42, label=HOF, progress_every=50):
    """LIME weight per word position for every post (list of numpy arrays).

    Args:
        predictor: an ``xhate.explain.predictor.Predictor`` (any model: CNN, BiLSTM, transformer).
        token_lists: the posts, as lists of words (no word may contain a space).
        num_samples: perturbed copies per post (see Q&A).
        seed: random seed for the perturbations.
        label: class whose probability is explained (1 = HOF, see Q&A).
        progress_every: print a progress line every this many posts (0 = silent).

    Q: Why print progress?
    A: LIME is by far the slowest step (num_samples model calls per post); for a transformer on
       1,314 posts it takes many minutes. Without progress lines it looks frozen.
    """
    # Imported here so that importing xhate.explain does not require the lime package.
    from lime.lime_text import LimeTextExplainer
    # time: elapsed time and the estimate of the time left.
    import time
    # Start of the LIME run.
    start = time.time()

    # One explainer for all posts (see the Q&A for every option).
    explainer = LimeTextExplainer(class_names=LABEL_NAMES,   # "NOT", "HOF"
                                  split_expression=r"\s+",   # words = Member 1's tokens
                                  bow=False,                 # one feature per position
                                  mask_string="",            # really delete removed words
                                  random_state=seed)         # reproducible perturbations

    def classifier_fn(texts):
        """LIME sends strings; split them on whitespace and ask the model."""
        # str.split() drops the empty pieces left by deleted words ("tum  ho" -> ["tum", "ho"]).
        return predictor.predict_proba([text.split() for text in texts])

    # One array per post.
    out = []
    # Explain the posts one by one (LIME fits one small linear model per post).
    for i, tokens in enumerate(token_lists):
        # Progress line every `progress_every` posts: done / total, elapsed, estimated time left.
        if progress_every and i and i % progress_every == 0:
            elapsed = time.time() - start                            # seconds so far
            left = elapsed / i * (len(token_lists) - i)              # same speed for the rest
            print(f"[lime] {i}/{len(token_lists)} posts | {elapsed / 60:.1f} min elapsed | "
                  f"~{left / 60:.1f} min left", flush=True)          # flush: show it immediately
        # A post with no words has nothing to explain.
        if len(tokens) == 0:
            out.append(np.zeros(0, dtype=np.float64))  # empty score array
            continue  # next post
        # A space (or tab, newline) inside a token, or an empty token, would make LIME see a
        # different number of words and break the alignment with the tokens.
        if any(not tok or any(ch.isspace() for ch in tok) for tok in tokens):
            raise ValueError(f"token with a space breaks the word alignment: {tokens}")
        # LIME needs the post as one string.
        text = " ".join(tokens)
        # Fit the local linear model; ask for a weight for EVERY word (num_features = n words).
        exp = explainer.explain_instance(text, classifier_fn, labels=(label,),
                                         num_features=len(tokens), num_samples=num_samples)
        # One score per position, 0 for words LIME did not return (none, with num_features = n).
        scores = np.zeros(len(tokens), dtype=np.float64)
        # as_map()[label] = list of (feature id, weight); with bow=False, feature id = word position.
        for position, weight in exp.as_map()[label]:
            scores[position] = weight  # this word's weight
        # Keep this post's scores.
        out.append(scores)
    # Done.
    return out
