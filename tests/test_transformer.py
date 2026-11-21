"""Tests for the transformer model and rationale loss (Member 3). Run with:

    pytest tests/test_transformer.py

No download, no GPU, no real data: we build a TINY random BERT (2 layers, hidden size 16)
and a tokenizer from a hand-written vocabulary, save both to a temp folder, and use that
folder as ``model.name``. Exactly the same code path as MuRIL, just ~10,000x smaller.
"""

# json: to write fake processed files.
import json
# sys / Path: to import scripts/train.py and scripts/run_seeds.py.
import sys
from pathlib import Path

# pytest: test runner.
import pytest

# Skip the whole file if torch / transformers are not installed.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from torch.utils.data import DataLoader  # noqa: E402

from xhate.data.dataset import HateSpeechDataset, collate_batch  # noqa: E402
from xhate.models.transformer import TransformerClassifier, build_transformer  # noqa: E402
from xhate.training.losses import rationale_loss, rationale_supervised_loss_fn  # noqa: E402
from xhate.training.trainer import default_loss_fn, fit  # noqa: E402

# Make scripts/ importable (for the end-to-end tests).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Words of the fake posts; "gaddar" decides HOF, "accha" decides NOT.
WORDS = ["ye", "log", "bahut", "gaddar", "accha", "hai", "they", "are", "bad"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Save a tiny random BERT + its tokenizer to a folder (usable as model.name)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    # Folder shared by every test of this file.
    folder = tmp_path_factory.mktemp("tiny_bert")
    # WordPiece vocabulary: special tokens first ([PAD] = id 0, like MuRIL).
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["##s"]
    (folder / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    # A fast tokenizer (needed for word_ids()) from that vocabulary.
    tok = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(folder)
    # 2 layers, 2 heads, hidden size 16: tiny but a real BERT.
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    BertModel(config).save_pretrained(folder)
    return folder


def make_records(n=40, rationale=False):
    """Easy posts. With rationale=True, HOF posts mark "gaddar" as the responsible word."""
    records = []
    for i in range(n):
        label = i % 2
        word = "gaddar" if label else "accha"
        tokens = ["ye", "log"] + ["bahut"] * (i % 3) + [word, "hai"]
        has = rationale and label == 1
        marks = [1.0 if t == "gaddar" else 0.0 for t in tokens]
        records.append({"id": f"p{i}", "text": " ".join(tokens), "tokens": tokens, "label": label,
                        "has_rationale": has,
                        "rationale": [int(m) for m in marks] if has else None,
                        "rationale_soft": marks if has else None})
    return records


# ------------------------------------------------------------------ model
def test_forward_shapes_and_attention(tiny_model_dir):
    """Logits are (B, 2); [CLS] attention is (B, L), >= 0, sums to 1 in eval, 0 on padding."""
    model = TransformerClassifier(str(tiny_model_dir))
    ids = torch.tensor([[2, 5, 6, 3, 0], [2, 5, 6, 7, 3]])
    mask = (ids != 0).long()
    att = model.get_attention(ids, mask)
    assert model(ids, mask).shape == (2, 2)
    assert att.shape == (2, 5)
    assert torch.all(att >= 0)
    assert torch.allclose(att.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert att[0, 4].item() == pytest.approx(0.0, abs=1e-6)  # padding gets no attention


def test_get_attention_restores_training_mode(tiny_model_dir):
    """get_attention switches to eval for the explanation and back again."""
    model = TransformerClassifier(str(tiny_model_dir)).train()
    model.get_attention(torch.tensor([[2, 5, 3]]), torch.ones(1, 3, dtype=torch.long))
    assert model.training


def test_build_transformer_reads_config(tiny_model_dir):
    """build_transformer passes the config values on."""
    model = build_transformer({"type": "transformer", "name": str(tiny_model_dir), "attention_layer": 0})
    assert model.attention_layer == 0 and model.classifier.out_features == 2


# ------------------------------------------------------------------- loss
def test_rationale_loss_is_zero_without_rationale_and_keeps_graph():
    """A batch with no rationale contributes 0 but backward() still works."""
    att = torch.full((2, 4), 0.25, requires_grad=True)
    loss = rationale_loss(att, torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))
    assert loss.item() == 0.0
    loss.backward()  # must not raise


def test_rationale_loss_is_smallest_when_attention_matches_target():
    """CE(target, attn) is minimal (= entropy of target) when attn == target."""
    rationale = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    mask = torch.tensor([[0, 1, 1, 1]])  # position 0 = [CLS], not a word
    # Attention on exactly the two marked words (plus some on [CLS], which is ignored).
    good = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    # Attention on the wrong word.
    bad = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
    good_loss = rationale_loss(good, rationale, mask)
    # target = [0.5, 0.5] on the words -> entropy = log 2.
    assert good_loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-4)
    assert rationale_loss(bad, rationale, mask) > good_loss


def test_rationale_loss_averages_only_over_posts_with_rationale():
    """A post without rationale does not dilute the mean."""
    att = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    rat = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 1, 1], [0, 0, 0]])
    both = rationale_loss(att, rat, mask)
    only_first = rationale_loss(att[:1], rat[:1], mask[:1])
    assert both.item() == pytest.approx(only_first.item())


def test_lambda_zero_equals_plain_cross_entropy(tiny_model_dir):
    """The control run (lambda 0) uses exactly the CE loss."""
    model = TransformerClassifier(str(tiny_model_dir))
    tok = transformers.AutoTokenizer.from_pretrained(str(tiny_model_dir))
    batch = collate_batch([HateSpeechDataset(make_records(4, rationale=True), tok)[i] for i in range(4)])
    logits = model(batch["input_ids"], batch["attention_mask"])
    plain = default_loss_fn()(logits, batch)
    assert rationale_supervised_loss_fn(model, lam=0.0)(logits, batch).item() == pytest.approx(plain.item())
    assert rationale_supervised_loss_fn(model, lam=1.0)(logits, batch).item() > plain.item()


def test_rationale_training_moves_attention_to_rationale(tiny_model_dir):
    """Training with a large lambda puts more [CLS] attention on the marked word."""
    tok = transformers.AutoTokenizer.from_pretrained(str(tiny_model_dir))
    records = make_records(16, rationale=True)
    loader = DataLoader(HateSpeechDataset(records, tok), batch_size=8, collate_fn=collate_batch)
    batch = next(iter(DataLoader(HateSpeechDataset(records, tok), batch_size=16, collate_fn=collate_batch)))

    def attention_on_rationale(model):
        """Mean share of [CLS] attention (over word positions) that falls on rationale words."""
        att = model.get_attention(batch["input_ids"], batch["attention_mask"]) * batch["rationale_mask"]
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-8)
        has = batch["rationale"].sum(-1) > 0
        return (att * (batch["rationale"] > 0)).sum(-1)[has].mean().item()

    torch.manual_seed(0)
    model = TransformerClassifier(str(tiny_model_dir))
    before = attention_on_rationale(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    fit(model, loader, loader, torch.device("cpu"), epochs=15, patience=100,
        loss_fn=rationale_supervised_loss_fn(model, lam=10.0), optimizer=opt, log=lambda *_: None)
    assert attention_on_rationale(model) > before


# ------------------------------------------------------------ end to end
def write_processed(folder, dataset, records_by_split):
    """Write fake processed files in Member 1's format."""
    for split, recs in records_by_split.items():
        path = folder / dataset / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


def write_config(tmp_path, model_dir, processed, dataset, lam):
    """A transformer config pointing at the tiny model and the fake data."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
run_name: tiny
seed: 1
data: {{dataset: {json.dumps(dataset)}, eval_dataset: hasoc, processed_dir: "{processed.as_posix()}", max_length: 16}}
model: {{type: transformer, name: "{model_dir.as_posix()}"}}
loss: {{lambda: {lam}}}
train: {{batch_size: 8, epochs: 2, lr: 0.001, patience: 5, device: cpu}}
""", encoding="utf-8")
    return cfg


def test_train_script_end_to_end_hasoc_plus_hatexplain(tmp_path, tiny_model_dir):
    """scripts/train.py trains a rationale model on HASOC + HateXplain and writes its outputs."""
    import train
    processed = tmp_path / "processed"
    write_processed(processed, "hasoc", {s: make_records(16) for s in ("train", "val", "test")})
    write_processed(processed, "hatexplain", {s: make_records(16, rationale=True) for s in ("train", "val", "test")})
    cfg = write_config(tmp_path, tiny_model_dir, processed, ["hasoc", "hatexplain"], 1.0)
    metrics = train.main(["--config", str(cfg), "--results-dir", str(tmp_path / "results")])
    run = tmp_path / "results" / "tiny"
    for name in ("best.pt", "config.yaml", "history.json", "metrics.json", "test_predictions.jsonl"):
        assert (run / name).exists(), name
    assert not (run / "vocab.json").exists()  # transformers have no word vocabulary
    # Train = both datasets, val / test = HASOC only.
    assert metrics["n_posts"] == {"train": 32, "val": 16, "test": 16}
    assert metrics["dataset"] == "hasoc+hatexplain"
    assert metrics["lambda"] == 1.0 and metrics["train_posts_with_rationale"] == 8
    # The saved weights load back into a fresh model.
    fresh = TransformerClassifier(str(tiny_model_dir))
    fresh.load_state_dict(torch.load(run / "best.pt"))


def test_run_seeds_reports_mean_and_std(tmp_path, tiny_model_dir):
    """run_seeds.py runs every seed in its own folder and summarises them."""
    import run_seeds
    processed = tmp_path / "processed"
    write_processed(processed, "hasoc", {s: make_records(16) for s in ("train", "val", "test")})
    cfg = write_config(tmp_path, tiny_model_dir, processed, "hasoc", 0.5)
    summary = run_seeds.main(["--config", str(cfg), "--seeds", "1", "2",
                              "--results-dir", str(tmp_path / "results")])
    assert summary["run_name"] == "tiny_lam0.5"
    assert (tmp_path / "results" / "tiny_lam0.5_seed2" / "metrics.json").exists()
    assert len(summary["macro_f1"]["values"]) == 2 and summary["macro_f1"]["std"] >= 0
