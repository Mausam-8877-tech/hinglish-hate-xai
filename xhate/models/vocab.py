"""Word vocabulary + word-level Dataset for the CNN / BiLSTM baselines (Member 2).

The transformers (Member 3) use a HuggingFace tokenizer. Our baselines learn
their OWN word embeddings, so they need their own word -> id table.

    from torch.utils.data import DataLoader
    from xhate.data import load_split
    from xhate.models.vocab import Vocab, WordDataset, collate_words

    train = load_split("hasoc", "train")
    vocab = Vocab.build([r["tokens"] for r in train], min_freq=2)
    ds = WordDataset(train, vocab, max_length=100)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_words)

Each batch is a dict with the SAME key names as Member 1's transformer batch,
so one training loop works for every model:
    input_ids       (B, L)  word ids, padded with PAD_ID
    attention_mask  (B, L)  1 for real words, 0 for padding
    label           (B,)    0 = NOT, 1 = HOF

Q&A
---
Q: What are <pad> and <unk>?
A: <pad> (id 0) fills short posts so every post in a batch has the same length;
   its embedding is fixed to zeros and it is masked out. <unk> (id 1) stands for
   any word that is not in the vocabulary (rare words, or words only seen in test).

Q: Why build the vocabulary from the TRAIN split only?
A: Words that appear only in val/test must look "unknown" to the model, exactly
   as they would for real new posts. Building from test would leak test data.

Q: What does min_freq do?
A: Words seen fewer than ``min_freq`` times in train become <unk>. A word seen
   once gives the model almost nothing to learn from, and dropping such words
   makes the model handle unknown words at test time.

Q: Why use the ``tokens`` list and not split ``text`` ourselves?
A: Member 1 already tokenised the posts correctly. Python's ``\\w`` regex does not
   match Hindi vowel signs (matras), so a re-split would cut Hindi words into
   letters (see data/README.md).

Q: Why truncate to max_length?
A: A few very long posts would make every batch they are in very slow. Hate is
   almost always visible in the first 100 words of a tweet.
"""

# Counter: counts how often each word appears in train.
from collections import Counter
# Path: OS-independent file paths.
from pathlib import Path

# torch: tensors.
import torch
# Dataset: base class that DataLoader can iterate over.
from torch.utils.data import Dataset

# json helpers (UTF-8, Hindi stays readable).
from xhate.utils import load_json, save_json

# Special token for padding. Its id MUST be 0 (the models assume it).
PAD = "<pad>"
# Special token for unknown words.
UNK = "<unk>"
# Id of <pad>.
PAD_ID = 0
# Id of <unk>.
UNK_ID = 1


class Vocab:
    """A two-way table between words and integer ids."""

    def __init__(self, itos):
        """Create a vocabulary from a list of words (index = id).

        Args:
            itos: "id to string" list. Must start with <pad>, <unk>.
        """
        # The special tokens must be at fixed positions.
        if itos[:2] != [PAD, UNK]:
            raise ValueError(f"vocabulary must start with {[PAD, UNK]}, got {itos[:2]}")  # clear message
        # id -> word.
        self.itos = list(itos)
        # word -> id (the reverse table).
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=1, max_size=None):
        """Build the vocabulary from tokenised TRAINING posts.

        Args:
            token_lists: list of token lists (e.g. ``[r["tokens"] for r in train]``).
            min_freq: drop words seen fewer times than this.
            max_size: keep at most this many words (most frequent first), plus the 2 specials.

        Q: How are ties in frequency ordered?
        A: Alphabetically, so the same data always gives the same ids on every machine.
        """
        # Count every word in every post.
        counts = Counter(tok for tokens in token_lists for tok in tokens)
        # The special tokens are added by hand, never counted.
        for special in (PAD, UNK):
            counts.pop(special, None)  # Member 1 may emit "<unk>" for empty HateXplain tokens
        # Keep words that are frequent enough.
        words = [w for w, c in counts.items() if c >= min_freq]
        # Most frequent first; ties broken alphabetically (deterministic).
        words.sort(key=lambda w: (-counts[w], w))
        # Cut to max_size if a limit was given.
        if max_size is not None:
            words = words[:max_size]  # keep the most frequent ones
        # Specials first, then the words.
        return cls([PAD, UNK] + words)

    def __len__(self):
        """Number of entries, INCLUDING <pad> and <unk> (= rows of the embedding table)."""
        # One entry per id.
        return len(self.itos)

    def encode(self, tokens):
        """Turn a list of words into a list of ids (unknown words -> UNK_ID)."""
        # Look every word up; .get returns UNK_ID for words we never saw.
        return [self.stoi.get(tok, UNK_ID) for tok in tokens]

    def decode(self, ids):
        """Turn a list of ids back into words (handy for printing attention)."""
        # Index into the id -> word list.
        return [self.itos[i] for i in ids]

    def save(self, path):
        """Save the vocabulary as JSON (needed to reload a trained model)."""
        # Only the list is needed; the reverse table is rebuilt on load.
        save_json({"itos": self.itos}, path)

    @classmethod
    def load(cls, path):
        """Load a vocabulary saved by ``save``."""
        # Rebuild from the saved list.
        return cls(load_json(Path(path))["itos"])


class WordDataset(Dataset):
    """Turn processed records (from ``load_split``) into word-id tensors."""

    def __init__(self, records, vocab, max_length=100):
        """Store the records and settings.

        Args:
            records: list of dicts from ``xhate.data.load_split``.
            vocab: a ``Vocab`` built from the training split.
            max_length: posts longer than this (in words) are cut.
        """
        # Keep the records.
        self.records = records
        # Keep the vocabulary.
        self.vocab = vocab
        # Keep the maximum length.
        self.max_length = max_length

    def __len__(self):
        """Number of posts."""
        # One item per record.
        return len(self.records)

    def __getitem__(self, idx):
        """Return the ids and the label of post number ``idx``."""
        # The processed record.
        rec = self.records[idx]
        # Word ids, cut to max_length.
        ids = self.vocab.encode(rec["tokens"][: self.max_length])
        # An empty post would break the LSTM (length 0); give it one <unk> instead.
        if not ids:
            ids = [UNK_ID]  # cannot happen with Member 1's data, but costs nothing
        # Return tensors.
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),         # (len,)
            "label": torch.tensor(rec["label"], dtype=torch.long),   # ()
        }


def collate_words(items):
    """Pad a list of items from ``WordDataset`` into one batch.

    Q: Why right-padding?
    A: The real words then always start at position 0, which is what
       ``pack_padded_sequence`` (LSTM) and our CNN length mask expect.
    """
    # Longest post in this batch.
    max_len = max(item["input_ids"].size(0) for item in items)
    # Empty batch tensors, filled with <pad>.
    input_ids = torch.full((len(items), max_len), PAD_ID, dtype=torch.long)
    # Mask: 0 everywhere at first.
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    # Copy each post into its row.
    for row, item in enumerate(items):
        n = item["input_ids"].size(0)  # real length of this post
        input_ids[row, :n] = item["input_ids"]  # the word ids
        attention_mask[row, :n] = 1  # mark the real words
    # Return the batch.
    return {
        "input_ids": input_ids,                                   # (B, L)
        "attention_mask": attention_mask,                         # (B, L)
        "label": torch.stack([item["label"] for item in items]),  # (B,)
    }
