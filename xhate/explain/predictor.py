"""Wrap ANY trained model as ``predict_proba(list of token lists)`` (Member 4).

Every explainer and every metric in this task only needs one question answered:
"what probability does the model give to NOT / HOF for this list of words?"
``Predictor`` answers it the same way for the CNN, the BiLSTM, MuRIL and XLM-R:

    from xhate.runs import load_run
    from xhate.explain.predictor import Predictor

    pred = Predictor.from_run(load_run("results/baseline_bilstm"))
    pred.predict_proba([["tum", "log", "gaddar", "ho"], ["aaj", "mausam", "accha", "hai"]])
    # -> numpy array of shape (2, 2): [[P(NOT), P(HOF)], [P(NOT), P(HOF)]]

Q&A
---
Q: Why does the input have to be a list of TOKEN lists, not raw strings?
A: Explanations give one score per WORD POSITION, and they must line up with Member 1's
   ``tokens`` (and with the human rationales, which are per token). Re-splitting text
   ourselves could cut Hindi words at their vowel signs (the ``\\w`` matra bug described in
   data/README.md). Working on token lists keeps everything aligned.

Q: How does one class handle two very different kinds of model?
A: Only the ENCODING differs:
   * baseline:    words -> ids with the run's ``Vocab`` (unknown words -> <unk>), cut to
                  ``max_length`` words, right-padded with <pad> (id 0);
   * transformer: words -> sub-words with the HuggingFace tokenizer
                  (``is_split_into_words=True``), cut to ``max_length`` sub-words, padded with
                  the tokenizer's OWN pad id (0 for MuRIL, 1 for XLM-R).
   After that both models are called the same way: ``model(input_ids, attention_mask)``.

Q: What is ``word_index`` in ``encode``?
A: For every model position, the index of the WORD it came from (None for [CLS], [SEP] and
   padding). Baselines: position i = word i. Transformers: ``word_ids()`` of the fast tokenizer
   (the word "trash" -> "tr", "##ash" gives [.., 2, 2, ..]). The attention explainer uses it to
   add sub-word attention back up into word attention.

Q: What happens with an EMPTY post (all words erased)?
A: The faithfulness metrics erase words, so a post can become empty. A baseline gets a single
   <unk> (an LSTM cannot run on length 0); a transformer gets just [CLS] [SEP]. Either way the
   model still returns a probability: its "no evidence" guess.

Q: Why batches, and why ``torch.no_grad()``?
A: LIME and the faithfulness metrics call the model thousands of times. Batching makes this
   fast; no_grad skips the gradient bookkeeping we never need here (faster, less memory).

Q: Why softmax?
A: The models return logits (unnormalised scores). softmax turns them into probabilities that
   are >= 0 and sum to 1, which is what LIME and the ERASER metrics expect.

Q: Why half precision (fp16) on a GPU?
A: LIME needs hundreds of thousands of forward passes for a transformer. GPUs such as the Colab
   T4 compute fp16 much faster than fp32. ``torch.autocast`` runs the matrix multiplications in
   fp16 and keeps sensitive operations (softmax, LayerNorm) in fp32, so probabilities change only
   in the 3rd-4th decimal. Every run is explained the same way, so comparisons stay fair.
   On the CPU nothing changes (fp32).
"""

# nullcontext: a "do nothing" with-block (used on the CPU instead of autocast).
from contextlib import nullcontext

# numpy: the output format LIME and scikit-learn expect.
import numpy as np
# torch: tensors and no_grad.
import torch


class Predictor:
    """A trained model + its encoding, exposed as ``predict_proba(token_lists)``."""

    def __init__(self, model, kind, vocab=None, tokenizer=None, max_length=128, device="cpu", batch_size=32):
        """Store the model and how to encode posts for it.

        Args:
            model: a CNN / BiLSTM / TransformerClassifier (any ``model(ids, mask) -> logits``).
            kind: "baseline" (needs ``vocab``) or "transformer" (needs ``tokenizer``).
            vocab: the run's ``Vocab`` (baselines).
            tokenizer: the run's HuggingFace *fast* tokenizer (transformers; needed for word_ids()).
            max_length: words (baselines) or sub-words (transformers) kept per post.
            device: where the model lives ("cpu" / "cuda" / torch.device).
            batch_size: posts per forward pass.
        """
        # A wrong kind would only fail much later, with a confusing error: check now.
        if kind not in ("baseline", "transformer"):
            raise ValueError(f"kind must be 'baseline' or 'transformer', got {kind!r}")
        # Each kind needs its own encoder.
        if kind == "baseline" and vocab is None:
            raise ValueError("a baseline Predictor needs the run's vocab")  # clear message
        if kind == "transformer" and tokenizer is None:
            raise ValueError("a transformer Predictor needs the run's tokenizer")  # clear message
        # Keep everything.
        self.model = model                      # the network
        self.kind = kind                        # how to encode
        self.vocab = vocab                      # word -> id (baselines)
        self.tokenizer = tokenizer              # word -> sub-words (transformers)
        self.max_length = max_length            # truncation length
        self.device = torch.device(device)      # accept "cpu" strings too
        self.batch_size = batch_size            # posts per forward pass
        # Explanations are always computed in eval mode (dropout off).
        self.model.to(self.device).eval()

    @classmethod
    def from_run(cls, run, batch_size=32):
        """Build a Predictor from ``xhate.runs.load_run(...)``'s result."""
        # Every field comes straight from the loaded run.
        return cls(run.model, run.kind, vocab=run.vocab, tokenizer=run.tokenizer,
                   max_length=run.max_length, device=run.device or "cpu", batch_size=batch_size)

    # ------------------------------------------------------------------ encoding
    def encode(self, token_lists):
        """Turn a batch of token lists into model tensors.

        Returns:
            (input_ids (B, L), attention_mask (B, L), word_index)
            where ``word_index[b][t]`` is the word index of position t of post b, or None.
        """
        # Baselines: our own word vocabulary.
        if self.kind == "baseline":
            # Imported here so a transformer-only setup does not import the baseline code.
            from xhate.models.vocab import PAD_ID, UNK_ID
            # Word ids per post, cut to max_length; an empty post becomes one <unk> (see Q&A).
            ids = [self.vocab.encode(list(toks)[: self.max_length]) or [UNK_ID] for toks in token_lists]
            # Longest post in this batch.
            width = max(len(seq) for seq in ids)
            # Batch filled with <pad>; mask filled with 0.
            input_ids = torch.full((len(ids), width), PAD_ID, dtype=torch.long)
            attention_mask = torch.zeros((len(ids), width), dtype=torch.long)
            # Copy each post into its row (right-padding, as in collate_words).
            for row, seq in enumerate(ids):
                input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)  # the word ids
                attention_mask[row, : len(seq)] = 1                              # real positions
            # Position t = word t, but only for real words (an empty post's <unk> is no word).
            word_index = [[t if t < len(toks) and t < self.max_length else None for t in range(width)]
                          for toks in token_lists]
            # Done.
            return input_ids, attention_mask, word_index
        # Transformers: the pre-trained tokenizer, on the ALREADY split words.
        enc = self.tokenizer([list(toks) for toks in token_lists],  # list of word lists
                             is_split_into_words=True,             # do not re-split our words
                             truncation=True,                      # cut long posts
                             max_length=self.max_length,           # ... to this many sub-words
                             padding=True,                         # pad to the longest in the batch
                             return_tensors="pt")                  # PyTorch tensors
        # word_ids(b): word index per sub-word of post b (None for specials and padding).
        word_index = [enc.word_ids(b) for b in range(len(token_lists))]
        # The tokenizer pads with its own pad id (0 for MuRIL, 1 for XLM-R).
        return enc["input_ids"], enc["attention_mask"], word_index

    def batches(self, token_lists):
        """Yield (start index, input_ids, attention_mask, word_index) per batch, on the device."""
        # Walk over the posts in steps of batch_size.
        for start in range(0, len(token_lists), self.batch_size):
            # Encode this slice of posts.
            input_ids, attention_mask, word_index = self.encode(token_lists[start: start + self.batch_size])
            # Move the tensors to the model's device and hand them out.
            yield start, input_ids.to(self.device), attention_mask.to(self.device), word_index

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()  # no gradients needed for explanations (see Q&A)
    def predict_proba(self, token_lists):
        """Class probabilities, shape (N, 2): column 0 = P(NOT), column 1 = P(HOF)."""
        # Nothing to do for an empty list (LIME never sends one, but be safe).
        if len(token_lists) == 0:
            return np.zeros((0, 2), dtype=np.float64)  # 0 rows, 2 columns
        # Probabilities of every batch.
        chunks = []
        # On a GPU, run the model in half precision (fp16): 2-3x faster for LIME (see Q&A).
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        # One forward pass per batch.
        for _, input_ids, attention_mask, _ in self.batches(token_lists):
            # Logits (B, 2), in fp16 on a GPU; softmax in float32 for exact probabilities.
            with amp:
                logits = self.model(input_ids, attention_mask)
            probs = logits.float().softmax(dim=-1)
            # Back to the CPU as numpy (float64: what LIME's regression expects).
            chunks.append(probs.double().cpu().numpy())
        # Stack the batches into one (N, 2) array.
        return np.concatenate(chunks, axis=0)

    def predict(self, token_lists):
        """Predicted class (0 = NOT, 1 = HOF) per post, plus the full probabilities."""
        # Probabilities for every post.
        probs = self.predict_proba(token_lists)
        # The class with the larger probability.
        return probs.argmax(axis=1), probs

    def __call__(self, token_lists):
        """``predictor(token_lists)`` is the same as ``predictor.predict_proba(token_lists)``."""
        # Handy when a library wants a plain function.
        return self.predict_proba(token_lists)
