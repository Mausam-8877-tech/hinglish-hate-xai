"""Tests for the baselines (Member 2). Run with:  pytest tests/test_baseline.py

Everything runs on tiny made-up posts on the CPU in a few seconds, so no real
(licensed) data is needed.
"""

# json: to write fake processed files.
import json
# sys / Path: to import scripts/train.py.
import sys
from pathlib import Path

# pytest: test runner.
import pytest

# Skip this whole file if torch is not installed (e.g. a data-only setup).
torch = pytest.importorskip("torch")

# Modules under test.
from torch.utils.data import DataLoader  # noqa: E402

from xhate.models.baseline import BiLSTMClassifier, CNNClassifier, build_model  # noqa: E402
from xhate.models.vocab import PAD_ID, UNK_ID, Vocab, WordDataset, collate_words  # noqa: E402
from xhate.training.losses import class_weights  # noqa: E402
from xhate.training.trainer import compute_metrics, default_loss_fn, fit, predict  # noqa: E402
from xhate.utils import load_config  # noqa: E402

# Make scripts/ importable (for the end-to-end test).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def make_records(n=40):
    """Tiny, easy dataset: HOF posts contain "gaddar", NOT posts contain "accha"."""
    # Half HOF, half NOT, with a shared filler word so posts differ in length.
    records = []
    for i in range(n):
        label = i % 2  # alternate NOT / HOF
        word = "gaddar" if label else "accha"  # the word that decides the class
        tokens = ["ye", "log"] + ["bahut"] * (i % 4) + [word]  # length 3 .. 6
        records.append({"id": f"p{i}", "text": " ".join(tokens), "tokens": tokens, "label": label,
                        "has_rationale": False})
    return records


# ------------------------------------------------------------------ vocab
def test_vocab_specials_frequency_and_unknown():
    """<pad>=0, <unk>=1, rare words dropped, unseen words map to <unk>."""
    # "a" x3, "b" x2, "c" x1.
    vocab = Vocab.build([["a", "b", "a"], ["a", "b", "c"]], min_freq=2)
    # Specials first, then by frequency.
    assert vocab.itos == ["<pad>", "<unk>", "a", "b"]
    # "c" was too rare and "zzz" was never seen: both become <unk>.
    assert vocab.encode(["a", "c", "zzz"]) == [2, UNK_ID, UNK_ID]
    # Decoding gives the words back.
    assert vocab.decode([2, 3]) == ["a", "b"]


def test_vocab_ignores_unk_token_in_data_and_saves(tmp_path):
    """A literal "<unk>" in the data is not counted twice; save/load gives the same table."""
    # Member 1 emits "<unk>" for empty HateXplain tokens.
    vocab = Vocab.build([["<unk>", "x", "x"]])
    assert vocab.itos == ["<pad>", "<unk>", "x"]
    # Round trip through JSON.
    vocab.save(tmp_path / "vocab.json")
    assert Vocab.load(tmp_path / "vocab.json").itos == vocab.itos


def test_dataset_truncates_and_collate_pads():
    """Long posts are cut; the batch is right-padded with <pad> and masked."""
    records = [{"tokens": ["a"] * 10, "label": 1}, {"tokens": ["a", "b"], "label": 0}]
    vocab = Vocab.build([r["tokens"] for r in records])
    ds = WordDataset(records, vocab, max_length=4)
    batch = collate_words([ds[0], ds[1]])
    # Longest post (after truncation) = 4 words.
    assert batch["input_ids"].shape == (2, 4)
    # Second post: 2 real words, then padding.
    assert batch["input_ids"][1, 2:].tolist() == [PAD_ID, PAD_ID]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["label"].tolist() == [1, 0]


# ----------------------------------------------------------------- models
@pytest.mark.parametrize("model", [
    CNNClassifier(vocab_size=20, embed_dim=8, num_filters=4, kernel_sizes=(2, 3, 5)),
    BiLSTMClassifier(vocab_size=20, embed_dim=8, hidden_dim=6),
    BiLSTMClassifier(vocab_size=20, embed_dim=8, hidden_dim=6, use_attention=False),
])
def test_prediction_does_not_depend_on_padding(model):
    """A post gets the same logits alone or padded inside a bigger batch."""
    model.eval()  # dropout off
    # A 2-word post (shorter than the widest CNN window, 5).
    short = torch.tensor([[3, 4]])
    alone = model(short, torch.ones_like(short))
    # The same post padded to 7 positions, next to a longer post.
    batch = torch.tensor([[3, 4, 0, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 11]])
    mask = (batch != PAD_ID).long()
    together = model(batch, mask)
    # Shape (B, 2) and identical scores for the short post.
    assert together.shape == (2, 2)
    assert torch.allclose(alone[0], together[0], atol=1e-5)


def test_bilstm_attention_weights():
    """Attention weights are >= 0, sum to 1 per post, and are 0 on padding."""
    model = BiLSTMClassifier(vocab_size=20, embed_dim=8, hidden_dim=6).eval()
    batch = torch.tensor([[3, 4, 5, 0], [6, 7, 8, 9]])
    logits, attn = model(batch, (batch != PAD_ID).long(), return_attention=True)
    assert logits.shape == (2, 2) and attn.shape == (2, 4)
    assert torch.allclose(attn.sum(dim=1), torch.ones(2))
    assert attn[0, 3].item() == 0.0  # the padding position


def test_build_model_from_config_and_rejects_typos():
    """build_model reads the model section; a misspelt key fails loudly."""
    assert isinstance(build_model({"type": "cnn", "embed_dim": 8}, vocab_size=10), CNNClassifier)
    assert isinstance(build_model({"type": "bilstm", "hidden_dim": 4}, vocab_size=10), BiLSTMClassifier)
    with pytest.raises(TypeError):
        build_model({"type": "bilstm", "hiden_dim": 4}, vocab_size=10)  # typo
    with pytest.raises(ValueError):
        build_model({"type": "gru"}, vocab_size=10)


# --------------------------------------------------------------- training
def test_class_weights_balanced_formula():
    """weight_c = N / (C * count_c)."""
    w = class_weights([0] * 6 + [1] * 4)
    assert torch.allclose(w, torch.tensor([10 / 12, 10 / 8]))


def test_compute_metrics():
    """Accuracy, macro-F1 and confusion matrix on a hand-checked example."""
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    # F1(NOT) = 2/3, F1(HOF) = 0.8  -> macro = 0.7333
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


@pytest.mark.parametrize("model_type", ["cnn", "bilstm"])
def test_fit_learns_easy_task_and_saves_checkpoint(tmp_path, model_type):
    """Both baselines learn the toy task; the best weights are saved and restored."""
    torch.manual_seed(0)
    records = make_records()
    vocab = Vocab.build([r["tokens"] for r in records])
    loader = DataLoader(WordDataset(records, vocab), batch_size=8, shuffle=True,
                        collate_fn=collate_words)
    model = build_model({"type": model_type, "embed_dim": 16, "dropout": 0.0}, len(vocab))
    ckpt = tmp_path / "best.pt"
    result = fit(model, loader, loader, torch.device("cpu"), epochs=30, lr=0.01, patience=30,
                 loss_fn=default_loss_fn(), checkpoint_path=ckpt, log=lambda *_: None)
    # The task is trivial, so the model must get (almost) everything right.
    assert result["best_val"]["macro_f1"] >= 0.95
    # The checkpoint exists and the restored model scores the best epoch's value.
    assert ckpt.exists()
    out = predict(model, loader, torch.device("cpu"))
    assert compute_metrics(out["labels"], out["preds"])["macro_f1"] == result["best_val"]["macro_f1"]


# ------------------------------------------------------------- end to end
def test_load_config_overrides(tmp_path):
    """--set style overrides change nested values with the right type."""
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 0.001\nmodel:\n  type: cnn\n", encoding="utf-8")
    cfg = load_config(path, ["train.lr=0.01", "model.kernel_sizes=[2, 3]", "train.class_weights=false"])
    assert cfg["train"]["lr"] == 0.01
    assert cfg["model"]["kernel_sizes"] == [2, 3]
    assert cfg["train"]["class_weights"] is False


@pytest.mark.parametrize("config", ["baseline_cnn.yaml", "baseline_bilstm.yaml"])
def test_train_script_end_to_end(tmp_path, config):
    """scripts/train.py runs the real configs on toy data and writes every output file."""
    import train  # scripts/train.py

    # Write toy processed data in Member 1's layout: <dir>/hasoc/{train,val,test}.jsonl
    for split in ("train", "val", "test"):
        path = tmp_path / "processed" / "hasoc" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in make_records(24)), encoding="utf-8")
    # The real config, made small and pointed at the toy data.
    repo = Path(__file__).resolve().parents[1]
    metrics = train.main([
        "--config", str(repo / "configs" / config), "--results-dir", str(tmp_path / "results"),
        "--set", f"data.processed_dir={tmp_path / 'processed'}", "data.min_freq=1",
        "train.epochs=3", "model.embed_dim=16", "run_name=toy",
    ])
    run_dir = tmp_path / "results" / "toy"
    for name in ("config.yaml", "vocab.json", "best.pt", "history.json", "metrics.json",
                 "test_predictions.jsonl"):
        assert (run_dir / name).exists(), name
    assert 0.0 <= metrics["test"]["macro_f1"] <= 1.0
    # One prediction line per test post.
    assert len((run_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 24
