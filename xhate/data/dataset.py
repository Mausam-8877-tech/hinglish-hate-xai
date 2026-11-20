"""PyTorch Dataset and collate function for transformer models (MuRIL / XLM-R).

This turns processed records (from ``xhate.data.load_split``) into tensors and
aligns the WORD-level rationale with the model's SUB-WORD tokens.

    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from xhate.data import load_split
    from xhate.data.dataset import HateSpeechDataset, collate_batch

    tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    ds = HateSpeechDataset(load_split("hatexplain", "train"), tok, max_length=128)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_batch)

Each item is a dict of tensors:
    input_ids       (seq_len,)  sub-word ids
    attention_mask  (seq_len,)  1 for real tokens, 0 for padding
    label           ()          0 = NOT, 1 = HOF
    rationale       (seq_len,)  float in [0, 1] per sub-word (soft target)
    rationale_mask  (seq_len,)  1 where the rationale should be used in the loss

Q&A
---
Q: How is a word's rationale given to its sub-words?
A: Every sub-word of a word gets that word's rationale value. Special tokens
   ([CLS], [SEP], padding) get rationale_mask = 0, so they never enter the loss.

Q: Why is this file not imported by ``xhate/data/__init__.py``?
A: It needs torch. Keeping it separate means Members who only need the data
   (EDA, LIME, sklearn baselines) can ``import xhate.data`` without torch.

Q: Why the soft rationale and not the hard 0/1?
A: Soft targets carry annotator agreement. Member 3 can switch with
   ``use_soft_rationale=False``.
"""

# torch: tensors and the Dataset base class.
import torch
# Dataset: base class that DataLoader knows how to iterate.
from torch.utils.data import Dataset


class HateSpeechDataset(Dataset):
    """Wrap processed records + a HuggingFace *fast* tokenizer."""

    def __init__(self, records, tokenizer, max_length=128, use_soft_rationale=True):
        """Store the records and settings.

        Args:
            records: list of dicts from ``load_split``.
            tokenizer: a HuggingFace fast tokenizer (needed for ``word_ids()``).
            max_length: posts longer than this (in sub-words) are truncated.
            use_soft_rationale: True -> ``rationale_soft``, False -> hard 0/1 ``rationale``.
        """
        # Keep the list of records.
        self.records = records
        # Keep the tokenizer.
        self.tokenizer = tokenizer
        # Keep the maximum sequence length.
        self.max_length = max_length
        # Keep which rationale field to use.
        self.rationale_key = "rationale_soft" if use_soft_rationale else "rationale"

    def __len__(self):
        """Number of posts."""
        # One item per record.
        return len(self.records)

    def __getitem__(self, idx):
        """Return the tensors for post number ``idx``."""
        # The processed record.
        rec = self.records[idx]
        # Tokenise the already-split words; truncate long posts.
        enc = self.tokenizer(rec["tokens"], is_split_into_words=True,
                             truncation=True, max_length=self.max_length)
        # For each sub-word, the index of the word it came from (None for special tokens).
        word_ids = enc.word_ids()
        # The word-level rationale, or None if this post has none.
        word_rationale = rec.get(self.rationale_key) if rec.get("has_rationale") else None
        # Sub-word rationale values (filled below).
        rationale = []
        # 1 where the rationale should count in the loss, else 0.
        rationale_mask = []
        # Walk over the sub-words.
        for wid in word_ids:
            # Special token, or a post without rationale: value 0, masked out.
            if wid is None or word_rationale is None:
                rationale.append(0.0)  # no target for this sub-word
                rationale_mask.append(0)  # excluded from the rationale loss
            else:
                # Copy the word's rationale to this sub-word, and use it in the loss.
                rationale.append(float(word_rationale[wid]))
                rationale_mask.append(1)  # included in the rationale loss
        # Return everything as tensors.
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),            # sub-word ids
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),  # 1 = real token
            "label": torch.tensor(rec["label"], dtype=torch.long),                    # class id
            "rationale": torch.tensor(rationale, dtype=torch.float),                 # target per sub-word
            "rationale_mask": torch.tensor(rationale_mask, dtype=torch.long),        # where to use it
        }


def collate_batch(items, pad_token_id=0):
    """Pad a list of items (from ``HateSpeechDataset``) into one batch of equal length.

    Q: Is pad_token_id = 0 correct for every model?
    A: It is for MuRIL/BERT. For XLM-R the pad id is 1; pass
       ``functools.partial(collate_batch, pad_token_id=tokenizer.pad_token_id)``.
    """
    # Longest sequence in this batch.
    max_len = max(item["input_ids"].size(0) for item in items)

    def pad(tensor, value):
        """Right-pad a 1-D tensor with ``value`` up to ``max_len``."""
        # How many positions are missing.
        missing = max_len - tensor.size(0)
        # Append ``missing`` copies of ``value``.
        return torch.cat([tensor, tensor.new_full((missing,), value)])

    # Stack every field into a batch tensor.
    return {
        "input_ids": torch.stack([pad(i["input_ids"], pad_token_id) for i in items]),   # (B, L)
        "attention_mask": torch.stack([pad(i["attention_mask"], 0) for i in items]),   # (B, L)
        "label": torch.stack([i["label"] for i in items]),                             # (B,)
        "rationale": torch.stack([pad(i["rationale"], 0.0) for i in items]),           # (B, L)
        "rationale_mask": torch.stack([pad(i["rationale_mask"], 0) for i in items]),   # (B, L)
    }
