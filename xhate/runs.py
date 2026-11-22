"""Reload a trained run from results/<run_name>/ (Member 4).

``scripts/train.py`` saves everything a model needs in ONE folder:

    results/<run_name>/
        config.yaml   the exact settings (model type, layer sizes, encoder name, max_length)
        vocab.json    word -> id table (CNN / BiLSTM only)
        best.pt       the weights of the best epoch

``load_run`` turns that folder back into a ready-to-use model:

    from xhate.runs import load_run
    run = load_run("results/muril_vanilla_seed42")        # or results/baseline_bilstm
    run.model        # the nn.Module, in eval mode, with the best weights
    run.kind         # "baseline" or "transformer"
    run.tokenizer    # HuggingFace tokenizer (transformers) / None
    run.vocab        # Vocab (baselines) / None

Q&A
---
Q: Why is this needed at all? Doesn't train.py already have the model?
A: train.py runs once (often on Kaggle) and ends. Explaining a model happens LATER, on
   another machine, maybe days after. The only thing that survives is the run folder, so we
   rebuild the model from it: same architecture (from config.yaml), same word table (vocab.json)
   and same weights (best.pt).

Q: Why not just ``torch.save(model)`` the whole model object?
A: Pickling a whole model ties the file to the exact code and class paths at save time.
   A state_dict (name -> tensor) is the recommended PyTorch format: small, portable, and
   it only needs the same architecture to load.

Q: For a transformer, why build the encoder from its CONFIG instead of ``from_pretrained``?
A: ``from_pretrained`` would download ~900 MB of pre-trained weights that ``best.pt``
   immediately overwrites. ``AutoConfig`` (a few KB) is enough to build the right empty
   architecture; ``load_state_dict`` then fills in OUR fine-tuned weights.

Q: Why ``attn_implementation="eager"``?
A: Same reason as in Member 3's transformer.py: the fast attention kernels (sdpa / flash)
   never build the attention matrix, so they cannot return the [CLS] attention we explain.

Q: Why ``map_location``?
A: A model trained on a Kaggle GPU saves CUDA tensors. ``map_location="cpu"`` loads them on a
   laptop without a GPU instead of crashing.

Q: Why ``model.eval()``?
A: Explanations must be deterministic: dropout OFF, and attention rows that sum to exactly 1.
"""

# dataclass: a small class that only holds data (fields + automatic __init__ / __repr__).
from dataclasses import dataclass
# Path: OS-independent file paths.
from pathlib import Path
# Any: type hint for "any object" (tokenizer / vocab can be different types or None).
from typing import Any

# torch: loads best.pt and picks the device.
import torch

# Config reading and device choice (shared helpers).
from xhate.utils import get_device, load_config

# Model types trained with a word vocabulary (Member 2).
BASELINE_TYPES = {"cnn", "bilstm"}
# Model types built on a pre-trained HuggingFace encoder (Member 3).
TRANSFORMER_TYPES = {"transformer"}


@dataclass
class Run:
    """Everything needed to use one trained model (filled in by ``load_run``)."""

    # Folder the run was loaded from, e.g. results/muril_vanilla_seed42.
    run_dir: Path
    # The run's config.yaml as a dict.
    cfg: dict
    # The model with the best weights, in eval mode.
    model: Any
    # "baseline" (CNN / BiLSTM, word ids) or "transformer" (MuRIL / XLM-R, sub-word ids).
    kind: str
    # Baselines: the Vocab built from training words; transformers: None.
    vocab: Any = None
    # Transformers: the HuggingFace tokenizer; baselines: None.
    tokenizer: Any = None
    # Posts are cut to this many words (baselines) or sub-words (transformers).
    max_length: int = 128
    # Device the model lives on (CPU / GPU).
    device: Any = None

    @property
    def name(self):
        """Short name of the run (the folder name), used in reports and tables."""
        # e.g. "muril_vanilla_seed42".
        return self.run_dir.name

    @property
    def model_type(self):
        """The config's model.type: "cnn", "bilstm" or "transformer"."""
        # Read straight from the saved config.
        return self.cfg["model"]["type"]


def _build_transformer_from_config(model_cfg):
    """Empty MuRIL / XLM-R architecture (no pre-trained download) + our classification head.

    Q: How does the result end up with the trained weights?
    A: ``load_run`` calls ``load_state_dict(best.pt)`` right after; best.pt contains the
       encoder AND the classifier weights, so every parameter is overwritten.
    """
    # Imported here so baseline-only setups do not need transformers installed.
    from transformers import AutoConfig, AutoModel

    # Member 3's classifier class (encoder + dropout + linear layer on [CLS]).
    from xhate.models.transformer import TransformerClassifier
    # Only the architecture description (layer count, hidden size, vocab size): a few KB.
    config = AutoConfig.from_pretrained(model_cfg["name"], attn_implementation="eager")
    # Randomly initialised encoder with that architecture (weights come from best.pt).
    encoder = AutoModel.from_config(config)
    # Wrap it exactly as train.py did, so the parameter names match best.pt.
    return TransformerClassifier(model_cfg["name"],                                    # saved for reference
                                 num_classes=model_cfg.get("num_classes", 2),          # NOT / HOF
                                 dropout=model_cfg.get("dropout", 0.1),                # unused in eval mode
                                 attention_layer=model_cfg.get("attention_layer", -1),  # explained layer
                                 encoder=encoder)                                      # skip the download


def load_run(run_dir, device="cpu"):
    """Rebuild the model saved in ``run_dir`` and load its best weights.

    Args:
        run_dir: a folder written by scripts/train.py (must contain config.yaml and best.pt).
        device: "cpu", "cuda" or "auto" (GPU if available).

    Returns:
        A ``Run`` with the model in eval mode.

    Q: What if a file is missing?
    A: FileNotFoundError naming the file, so the user knows which download from Drive failed.
    """
    # Accept a str or a Path.
    run_dir = Path(run_dir)
    # The three files a run folder may need.
    config_path, weights_path, vocab_path = run_dir / "config.yaml", run_dir / "best.pt", run_dir / "vocab.json"
    # config.yaml and best.pt are always needed.
    for path in (config_path, weights_path):
        # Fail early with a clear message.
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: is {run_dir} a folder written by scripts/train.py?")
    # The exact settings of that run.
    cfg = load_config(config_path)
    # cnn / bilstm / transformer.
    model_type = cfg["model"]["type"]
    # torch.device object ("auto" -> GPU if there is one).
    device = get_device(device)
    # Transformers: tokenizer + empty encoder architecture.
    if model_type in TRANSFORMER_TYPES:
        # Imported here: only transformer runs need it.
        from transformers import AutoTokenizer
        # The tokenizer of the same checkpoint (small download, then cached).
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        # Architecture without the 900 MB pre-trained download (see Q&A).
        model = _build_transformer_from_config(cfg["model"])
        # Transformers count sub-words; train.py's default was 128.
        kind, vocab, max_length = "transformer", None, cfg["data"].get("max_length", 128)
    # Baselines: vocab.json + CNN / BiLSTM.
    elif model_type in BASELINE_TYPES:
        # Imported here so transformer-only use does not import the baseline code.
        from xhate.models.baseline import build_model
        from xhate.models.vocab import Vocab
        # Baselines cannot be rebuilt without their word table.
        if not vocab_path.exists():
            raise FileNotFoundError(f"{vocab_path} not found (baseline runs need their vocabulary)")
        # The word -> id table built from that run's training posts.
        vocab = Vocab.load(vocab_path)
        # Same architecture; the embedding table has one row per vocabulary entry.
        model = build_model(cfg["model"], vocab_size=len(vocab))
        # Baselines count words; train.py's default was 100.
        kind, tokenizer, max_length = "baseline", None, cfg["data"].get("max_length", 100)
    # Anything else is not a model train.py can produce.
    else:
        raise ValueError(f"unknown model.type {model_type!r} in {config_path}")
    # The saved weights; map_location moves GPU tensors to our device (see Q&A).
    state = torch.load(weights_path, map_location=device)
    # Copy them into the model; strict (default) fails loudly if any name does not match.
    model.load_state_dict(state)
    # Move the model to the device and switch dropout off (see Q&A).
    model.to(device).eval()
    # Everything bundled together.
    return Run(run_dir=run_dir, cfg=cfg, model=model, kind=kind, vocab=vocab,
               tokenizer=tokenizer, max_length=max_length, device=device)
