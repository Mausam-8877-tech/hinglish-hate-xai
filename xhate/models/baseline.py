"""Member 2: CNN and BiLSTM(+attention) baseline classifiers.

These are the "floor": every later model (MuRIL / XLM-R, rationale-supervised)
has to beat them. Both models take a batch from ``xhate.models.vocab.collate_words``:

    logits = model(batch["input_ids"], batch["attention_mask"])   # (B, 2)

    # BiLSTM only: also get one attention weight per word (for Member 4).
    logits, attn = model(batch["input_ids"], batch["attention_mask"], return_attention=True)

Build either one from a config with ``build_model(cfg["model"], vocab_size)``.

Q&A
---
Q: What is the Kim (2014) CNN?
A: Words -> embeddings -> several 1-D convolutions with different window sizes
   (3, 4, 5 words) -> ReLU -> max over the whole post -> concatenate -> dropout
   -> linear layer. Each filter learns to detect one n-gram pattern (e.g. an
   insult phrase); max-pooling keeps its strongest match anywhere in the post.
   Paper: Yoon Kim, "Convolutional Neural Networks for Sentence Classification", EMNLP 2014.

Q: Why a BiLSTM?
A: It reads the post left-to-right AND right-to-left, so every word's hidden state
   knows its context on both sides ("nahi" before or after a word changes meaning).

Q: What does the attention layer add?
A: Instead of using only the last hidden state, it scores every word, turns the
   scores into weights that sum to 1 (softmax), and uses the weighted average of
   the hidden states. The weights tell us which words the model relied on, so
   Member 4 can show them as an explanation. (Additive attention as in
   Yang et al., "Hierarchical Attention Networks", NAACL 2016.)

Q: Is attention a faithful explanation?
A: Not necessarily. "Attention is not Explanation" (Jain & Wallace, 2019) shows
   attention weights can disagree with what really drives the prediction. That is
   exactly why Member 4 measures faithfulness instead of trusting the weights.

Q: Why mask the padding?
A: Padding is not part of the post. Without the mask, the same post would get a
   different prediction depending on how long the OTHER posts in its batch are.
   With the mask, a post scores the same alone or in any batch (we test this).

Q: Why do the models return logits and not probabilities?
A: ``nn.CrossEntropyLoss`` applies softmax itself, in a numerically stable way.
   Use ``logits.softmax(-1)`` when you need probabilities (e.g. for LIME).
"""

# torch: tensors.
import torch
# nn: layers (Embedding, Conv1d, LSTM, Linear, ...).
from torch import nn
# pack_padded_sequence / pad_packed_sequence: let the LSTM skip padding.
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Id of <pad> (always 0).
from xhate.models.vocab import PAD_ID


class CNNClassifier(nn.Module):
    """Kim (2014) style CNN text classifier."""

    def __init__(self, vocab_size, embed_dim=300, num_filters=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5, num_classes=2):
        """Create the layers.

        Args:
            vocab_size: number of words in the vocabulary (incl. <pad>, <unk>).
            embed_dim: size of each word vector.
            num_filters: filters PER kernel size (output = num_filters * len(kernel_sizes)).
            kernel_sizes: window sizes in words, e.g. (3, 4, 5) = trigrams to 5-grams.
            dropout: dropout probability before the output layer.
            num_classes: 2 (NOT / HOF).
        """
        # Initialise nn.Module.
        super().__init__()
        # Remember the kernel sizes (needed in forward).
        self.kernel_sizes = list(kernel_sizes)
        # Word id -> vector. padding_idx keeps the <pad> vector at zero and never trains it.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        # One convolution per window size. Conv1d slides over the words (the "time" axis).
        self.convs = nn.ModuleList(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)  # k-word window
            for k in self.kernel_sizes  # e.g. 3 convolutions for kernel_sizes (3, 4, 5)
        )  # ModuleList registers the convolutions so their weights are trained and saved
        # Dropout: randomly zeroes features during training to reduce overfitting.
        self.dropout = nn.Dropout(dropout)
        # Final layer: all pooled features -> class scores.
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None):
        """Return logits of shape (B, num_classes)."""
        # Without a mask, every non-<pad> id counts as a real word.
        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()  # 1 = real word
        # Real length of each post; at least 1 so an empty post still works.
        lengths = attention_mask.sum(dim=1).clamp(min=1)  # (B,)
        # A post shorter than the widest window gives no convolution output at all,
        # so pad the batch on the right up to the widest window. <pad> vectors are
        # zeros, so this does not change the result.
        max_k = max(self.kernel_sizes)  # the widest window, e.g. 5
        # Only needed when the whole batch is shorter than the widest window.
        if input_ids.size(1) < max_k:
            extra = max_k - input_ids.size(1)  # how many <pad> columns are missing
            # (0, extra) = add 0 columns on the left and ``extra`` on the right.
            input_ids = nn.functional.pad(input_ids, (0, extra), value=PAD_ID)
        # Look up the word vectors: (B, L, E).
        emb = self.embedding(input_ids)
        # Conv1d wants channels first: (B, E, L).
        emb = emb.transpose(1, 2)
        # Pooled features from every window size.
        pooled = []
        # Go through each window size k with its convolution.
        for k, conv in zip(self.kernel_sizes, self.convs):
            # Convolution + ReLU: (B, F, L - k + 1); position t covers words t .. t+k-1.
            feats = torch.relu(conv(emb))
            # Number of windows that start inside the real post (at least 1).
            n_valid = (lengths - k + 1).clamp(min=1)  # (B,)
            # True where the window starts in the real post, False where it is only padding.
            positions = torch.arange(feats.size(2), device=feats.device)  # 0, 1, 2, ...
            valid = positions.unsqueeze(0) < n_valid.unsqueeze(1)  # (B, L - k + 1)
            # Padding windows get -inf so max-pooling can never pick them.
            feats = feats.masked_fill(~valid.unsqueeze(1), float("-inf"))
            # Max over time: the strongest match of every filter anywhere in the post.
            pooled.append(feats.max(dim=2).values)  # (B, F)
        # Concatenate all window sizes: (B, F * number of kernel sizes).
        features = torch.cat(pooled, dim=1)
        # Dropout (only active in model.train() mode), then the output layer.
        return self.fc(self.dropout(features))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with optional additive attention."""

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=1,
                 dropout=0.5, num_classes=2, use_attention=True):
        """Create the layers.

        Args:
            vocab_size: number of words in the vocabulary (incl. <pad>, <unk>).
            embed_dim: size of each word vector.
            hidden_dim: LSTM size PER direction (output is 2 * hidden_dim).
            num_layers: stacked LSTM layers.
            dropout: dropout on the embeddings and before the output layer.
            num_classes: 2 (NOT / HOF).
            use_attention: True = attention pooling; False = last hidden states.
        """
        # Initialise nn.Module.
        super().__init__()
        # Remember whether to use attention.
        self.use_attention = use_attention
        # Word id -> vector (the <pad> vector stays zero).
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        # Dropout layer, reused on embeddings and on the final features.
        self.dropout = nn.Dropout(dropout)
        # The BiLSTM. batch_first=True means tensors are (B, L, ...).
        # Dropout BETWEEN layers only makes sense with 2+ layers (PyTorch warns otherwise).
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True,  # read the post in both directions
                            dropout=dropout if num_layers > 1 else 0.0)  # between-layer dropout
        # Attention: score_i = v . tanh(W h_i + b)  (additive attention).
        self.attn_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  # W and b
        self.attn_vector = nn.Linear(2 * hidden_dim, 1, bias=False)  # v
        # Final layer: sentence vector (2 * hidden_dim) -> class scores.
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, return_attention=False):
        """Return logits (B, num_classes), and the attention weights (B, L) if asked.

        Q: What are the attention weights when ``use_attention=False``?
        A: There are none, so we return None in their place.
        """
        # Without a mask, every non-<pad> id counts as a real word.
        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()  # 1 = real word
        # Real length of each post (at least 1). pack_padded_sequence needs them on the CPU.
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        # Embeddings with dropout: (B, L, E).
        emb = self.dropout(self.embedding(input_ids))
        # "Pack" the batch so the LSTM runs only over the real words of each post.
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Run the LSTM. h_n: final hidden state of every layer and direction.
        packed_out, (h_n, _) = self.lstm(packed)
        # Unpack back to (B, L, 2H). total_length keeps L equal to the input length.
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True,  # back to (B, L, 2H)
                                         total_length=input_ids.size(1))  # padding positions become zeros
        # Attention weights (None if attention is off).
        weights = None
        # Attention pooling (the default).
        if self.use_attention:
            # One score per word: (B, L).
            scores = self.attn_vector(torch.tanh(self.attn_proj(outputs))).squeeze(-1)
            # Padding gets -inf, so after softmax its weight is exactly 0.
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            # Softmax over the words: weights are >= 0 and sum to 1 per post.
            weights = torch.softmax(scores, dim=1)  # (B, L)
            # Weighted average of the hidden states = the sentence vector: (B, 2H).
            sentence = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        # No attention: use the final hidden states instead.
        else:
            # Last layer's forward state (h_n[-2]) and backward state (h_n[-1]), joined: (B, 2H).
            sentence = torch.cat([h_n[-2], h_n[-1]], dim=1)
        # Dropout, then the output layer.
        logits = self.fc(self.dropout(sentence))
        # Return the weights too if the caller asked for them.
        return (logits, weights) if return_attention else logits


def build_model(model_cfg, vocab_size):
    """Create a baseline from the ``model:`` section of a config.

    Example config sections:
        model: {type: cnn, embed_dim: 300, num_filters: 100, kernel_sizes: [3, 4, 5], dropout: 0.5}
        model: {type: bilstm, embed_dim: 300, hidden_dim: 128, use_attention: true, dropout: 0.5}

    Q: Why not just pass the whole dict to the class?
    A: We copy it and remove "type" first, so a typo in a key name (e.g. "hiden_dim")
       raises a clear TypeError instead of being silently ignored.
    """
    # Copy so we do not change the caller's config.
    params = dict(model_cfg)
    # Which model to build.
    model_type = params.pop("type")
    # Table of the baselines we know.
    classes = {"cnn": CNNClassifier, "bilstm": BiLSTMClassifier}
    # Unknown type: say which ones exist.
    if model_type not in classes:
        raise ValueError(f"unknown baseline type {model_type!r}; expected one of {sorted(classes)}")  # clear message
    # Create the model: the remaining keys become keyword arguments (**params).
    # YAML gives lists; the CNN is happy with a list for kernel_sizes.
    return classes[model_type](vocab_size=vocab_size, **params)
