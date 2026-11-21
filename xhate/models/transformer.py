"""Member 3: MuRIL / XLM-R classifier that can also return its [CLS] attention.

    from xhate.models.transformer import TransformerClassifier, build_transformer

    model = TransformerClassifier("google/muril-base-cased")     # or "xlm-roberta-base"
    logits = model(batch["input_ids"], batch["attention_mask"])  # (B, 2)
    att = model.last_cls_attention                               # (B, L), set by every forward
    att = model.get_attention(input_ids, attention_mask)         # same, for Member 4 (no gradients)

Architecture (what happens inside ``forward``):

    sub-word ids ─► pre-trained encoder (12 transformer layers) ─► hidden vector per token
                                                                     │
                       [CLS] hidden vector (position 0) ─► dropout ─► Linear(hidden, 2) ─► logits
    attention of layer ``attention_layer``: row 0 ([CLS] looking at every token),
    averaged over the heads ─► ``last_cls_attention`` (used by the rationale loss and by Member 4)

The interface is the one Member 2's ``fit`` expects: ``model(input_ids, attention_mask)``
returns logits of shape (B, C). Nothing in the training loop had to change for this model.

Q&A
---
Q: What is MuRIL? What is XLM-R?
A: Both are BERT-style transformer ENCODERS pre-trained with masked-language-modelling
   (hide 15% of the tokens, predict them back) on huge unlabelled text.
   * MuRIL (Google, 2021): 17 Indian languages + English, AND transliterated text
     (Hindi written in Latin letters). This makes it a natural fit for Hinglish.
     Vocabulary: WordPiece, ~197k entries. Pad id 0.
   * XLM-R base (Facebook, 2019): 100 languages from CommonCrawl, SentencePiece vocabulary
     of 250k entries. Strong general multilingual model, but saw little romanised Hindi.
     Pad id 1 (not 0!), which is why the collate function needs ``tokenizer.pad_token_id``.
   Both have 12 layers, 12 attention heads, hidden size 768 ("base" size).

Q: What is fine-tuning, and why not train from scratch?
A: Our ~4k HASOC posts are far too few to learn a language from scratch. The pre-trained
   encoder already "knows" Hindi/English words and grammar; we add a small, randomly
   initialised classification layer on top and train EVERYTHING together for a few epochs
   with a small learning rate (2e-5). This is transfer learning.

Q: Why the [CLS] token?
A: The tokenizer puts a special token at position 0 ([CLS] for MuRIL, <s> for XLM-R). Through
   self-attention it can look at every other token, so its final hidden vector works as a
   summary of the whole post. BERT's own fine-tuning recipe classifies from it.
   Alternative: mean-pooling over all token vectors. Works similarly, but then "the [CLS]
   attention" would no longer be what the classifier reads, and our rationale loss is
   defined on the [CLS] attention (as in the HateXplain paper).

Q: What is "attention" here, exactly?
A: In every layer and head, each token i computes softmax(q_i . k_j / sqrt(d)) over all
   tokens j: a probability distribution saying how much token i takes from token j.
   That gives a (L x L) matrix per head. Row 0 is how much [CLS] reads from each token.
   We average row 0 over the heads of one layer (the last by default) -> one weight per token.

Q: Why ``attn_implementation="eager"``?
A: Recent versions of transformers use fast fused kernels ("sdpa", flash-attention) that
   never build the attention matrix, so they cannot return it. "eager" is the plain PyTorch
   implementation, which can. It is a bit slower and uses a bit more memory; for 128-token
   posts this is negligible.

Q: Why store the attention on the model (``last_cls_attention``) instead of returning it?
A: Member 2's ``fit`` calls ``logits = model(...)`` and ``loss_fn(logits, batch)``. If forward
   returned a tuple, ``fit`` would break. Keeping it as an attribute lets the rationale loss
   (which holds a reference to the model) read it, with zero changes to ``fit``/``predict``.
   It is overwritten on every forward pass, so it always belongs to the current batch.

Q: Why does an attention row sum to slightly more than 1 during training?
A: Attention DROPOUT: during training, transformers randomly zero 10% of the attention
   probabilities and scale the rest by 1/0.9. The returned matrix is the one after dropout.
   The rationale loss re-normalises the attention over the real word positions, so this is
   harmless; in ``model.eval()`` mode (Member 4) dropout is off and rows sum to exactly 1.
"""

# torch: tensors and no_grad.
import torch
# nn: Module, Linear, Dropout.
from torch import nn


class TransformerClassifier(nn.Module):
    """Pre-trained encoder + dropout + linear layer on the [CLS] vector."""

    def __init__(self, model_name, num_classes=2, dropout=0.1, attention_layer=-1, encoder=None):
        """Load the encoder and create the classification head.

        Args:
            model_name: HuggingFace hub name ("google/muril-base-cased", "xlm-roberta-base")
                or a local folder made with ``save_pretrained`` (used by the tests).
            num_classes: 2 (NOT / HOF).
            dropout: dropout before the linear layer. 0.1 = BERT's own value.
            attention_layer: which layer's [CLS] attention to expose / supervise.
                -1 = last layer (as in HateXplain); 0 = first, 5 = sixth, ...
            encoder: an already-built encoder (optional; tests pass a tiny random one).
        """
        # Initialise nn.Module's internals (parameter registration etc.).
        super().__init__()
        # Imported here so that importing xhate.models does not require transformers.
        from transformers import AutoModel
        # Keep the name; it is saved in the run config so the model can be rebuilt.
        self.model_name = model_name
        # Which layer to read attention from (Python indexing, so -1 = last).
        self.attention_layer = attention_layer
        # Load the pre-trained weights (downloads ~900 MB the first time, then cached).
        # AutoModel picks the right class (BertModel for MuRIL, XLMRobertaModel for XLM-R)
        # from the checkpoint's config. "eager" so attention matrices can be returned (see Q&A).
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            model_name, attn_implementation="eager")
        # Size of each token's hidden vector (768 for base models).
        hidden = self.encoder.config.hidden_size
        # Dropout: randomly zeroes 10% of the [CLS] features in training (regularisation).
        self.dropout = nn.Dropout(dropout)
        # The new, randomly initialised layer: 768 features -> 2 class scores (logits).
        self.classifier = nn.Linear(hidden, num_classes)
        # [CLS] attention of the last forward pass, shape (B, L). None until the first call.
        self.last_cls_attention = None

    def forward(self, input_ids, attention_mask):
        """Return logits (B, C) and store the [CLS] attention in ``self.last_cls_attention``.

        Args:
            input_ids: (B, L) sub-word ids, padded.
            attention_mask: (B, L) 1 = real token, 0 = padding (padding gets no attention).
        """
        # Run the encoder. output_attentions=True -> a tuple with one (B, H, L, L) tensor per layer.
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # Attention of the chosen layer: (B, heads, L, L).
        layer_att = out.attentions[self.attention_layer]
        # Row 0 = what [CLS] attends to: (B, heads, L); mean over heads -> (B, L).
        # Averaging the heads is what HateXplain does; one head alone is noisy.
        self.last_cls_attention = layer_att[:, :, 0, :].mean(dim=1)
        # Final hidden vectors (B, L, hidden); take position 0 = [CLS] -> (B, hidden).
        cls_vector = out.last_hidden_state[:, 0]
        # Dropout, then the linear layer -> (B, C) unnormalised class scores.
        return self.classifier(self.dropout(cls_vector))

    @torch.no_grad()  # explanation only: no gradients, less memory
    def get_attention(self, input_ids, attention_mask):
        """[CLS] attention (B, L) for explanation (Member 4). Runs in eval mode (no dropout).

        Map it back to words with the tokenizer's ``word_ids()``: e.g. sum or average
        the weights of the sub-words of each word.
        """
        # Remember whether we were training, so we can restore it afterwards.
        was_training = self.training
        # Eval mode: dropout off, attention rows sum to exactly 1.
        self.eval()
        # A forward pass fills last_cls_attention.
        self(input_ids, attention_mask)
        # Restore the previous mode.
        self.train(was_training)
        # Return the (B, L) attention weights.
        return self.last_cls_attention


def build_transformer(model_cfg):
    """Create a TransformerClassifier from the ``model:`` section of a config.

    Example section:
        model:
          type: transformer
          name: google/muril-base-cased
          dropout: 0.1
          attention_layer: -1
    """
    # Every value except the name has a default.
    return TransformerClassifier(model_cfg["name"],                                # hub name or local folder
                                 num_classes=model_cfg.get("num_classes", 2),      # NOT / HOF
                                 dropout=model_cfg.get("dropout", 0.1),            # head dropout
                                 attention_layer=model_cfg.get("attention_layer", -1))  # supervised layer
