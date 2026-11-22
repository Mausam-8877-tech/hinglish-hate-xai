"""Attention-based word importance (Member 4).

    from xhate.explain.attention import attention_scores
    scores = attention_scores(predictor, [["tum", "log", "gaddar", "ho"]])
    # -> [array([0.10, 0.05, 0.80, 0.05])]   one weight per WORD, summing to 1

Where the weights come from:
    * MuRIL / XLM-R: the [CLS] attention of the supervised layer (last layer by default),
      averaged over the heads, from Member 3's ``TransformerClassifier.get_attention``.
      It is per SUB-WORD, so we add it back up into words (``subword_to_word``).
    * BiLSTM: the additive-attention weights of Member 2's model (already one per word).
    * CNN: has no attention, so there is no attention explanation (use LIME instead).

Q&A
---
Q: What does the [CLS] attention mean?
A: The classifier only reads the final [CLS] vector. In the chosen layer, [CLS] builds its new
   vector as a weighted average of every token's vector; the weights are softmax(q_cls . k_j / sqrt(d)).
   A high weight on a word = [CLS] "took a lot" from that word. That is the intuition for using
   it as an explanation, and it is also what the rationale loss trains (Member 3).

Q: How are sub-words turned back into words? Why SUM and not mean?
A: "gaddaron" may become "gadd", "##aron". Attention is a probability mass, so the word's
   share is the SUM of its pieces' shares. With the mean, a word split into 5 pieces would look
   5x less important than the same mass on a one-piece word, which is an artefact of the
   tokenizer, not of the model. Summing is also the inverse of Member 1's mapping (every
   sub-word got its word's rationale).

Q: Why re-normalise over the words?
A: [CLS] also attends to itself and to [SEP] (often a LOT: [SEP] is a known "no-op" sink in
   BERT). Those are not words, so we drop them and rescale the word weights to sum to 1.
   Same idea as the re-normalisation in Member 3's rationale loss.

Q: What about words cut off by truncation?
A: They never reached the model, so they get weight 0.

Q: Is attention a FAITHFUL explanation?
A: Not necessarily! "Attention is not Explanation" (Jain & Wallace, 2019) found attention can
   disagree with gradient-based importance, and that other attention maps can give the same
   prediction. "Attention is not not Explanation" (Wiegreffe & Pinter, 2019) replied that it
   depends on the model and task. That is exactly why we MEASURE faithfulness (ERASER
   comprehensiveness / sufficiency) instead of trusting the weights.

Q: Why only one layer and not all 12?
A: The rationale loss supervises ONE layer (``attention_layer``, last by default). Explaining
   the same layer is what lets us test whether supervision changed the explanation.
   (Combining layers, e.g. "attention rollout", is a possible extension.)
"""

# numpy: one score array per post.
import numpy as np
# torch: no_grad.
import torch


def subword_to_word(weights, word_index, n_words):
    """Add sub-word weights into word weights and re-normalise them to sum to 1.

    Args:
        weights: 1-D sequence, one weight per model position (sub-word or word).
        word_index: same length; the word each position belongs to, or None (specials, padding).
        n_words: number of words in the post (words cut by truncation get 0).

    Returns:
        numpy array of length ``n_words``.

    Example:
        weights    = [0.3, 0.1, 0.2, 0.2, 0.2]       ([CLS] you are tr ##ash)
        word_index = [None, 0, 1, 2, 2]
        -> raw [0.1, 0.2, 0.4] -> normalised [0.143, 0.286, 0.571]
    """
    # One bucket per word, starting at 0.
    scores = np.zeros(n_words, dtype=np.float64)
    # Walk over the positions together with their word.
    for weight, wid in zip(weights, word_index):
        # Specials ([CLS], [SEP]) and padding belong to no word: skip them (see Q&A).
        if wid is None or wid >= n_words:
            continue  # not a word of this post
        # Add this piece's attention to its word (SUM, see Q&A).
        scores[wid] += float(weight)
    # Total attention that landed on words.
    total = scores.sum()
    # Re-normalise so the words share 100% (unless nothing landed on words at all).
    return scores / total if total > 0 else scores


@torch.no_grad()  # explanation only: no gradients
def attention_scores(predictor, token_lists):
    """One attention weight per word for every post (list of numpy arrays).

    Args:
        predictor: an ``xhate.explain.predictor.Predictor`` (transformer or BiLSTM).
        token_lists: the posts, as lists of words.

    Raises:
        ValueError for a model without attention (the CNN).
    """
    # The underlying network.
    model = predictor.model
    # Transformers expose get_attention; the BiLSTM exposes return_attention=True.
    is_transformer = predictor.kind == "transformer"
    # The CNN (or a BiLSTM trained with use_attention: false) has no attention to show.
    if not is_transformer and not getattr(model, "use_attention", False):
        raise ValueError("this model has no attention weights (CNN or BiLSTM without attention); "
                         "use LIME instead")
    # One array per post, filled batch by batch.
    out = [None] * len(token_lists)
    # Same batching and encoding as predict_proba.
    for start, input_ids, attention_mask, word_index in predictor.batches(token_lists):
        # Transformer: [CLS] attention (B, L) in eval mode, from Member 3's model.
        if is_transformer:
            weights = model.get_attention(input_ids, attention_mask)
        # BiLSTM: the attention weights (B, L), returned next to the logits.
        else:
            _, weights = model(input_ids, attention_mask, return_attention=True)
        # Back to numpy on the CPU.
        weights = weights.float().cpu().numpy()
        # Turn every row of the batch into word scores.
        for row in range(weights.shape[0]):
            # Position of this post in the full list.
            i = start + row
            # Model positions -> words (sum + re-normalise).
            out[i] = subword_to_word(weights[row], word_index[row], len(token_lists[i]))
    # Done.
    return out
