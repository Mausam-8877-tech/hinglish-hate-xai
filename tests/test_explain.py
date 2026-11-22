"""Tests for explanations and their evaluation (Member 4). Run with:  pytest tests/test_explain.py

No real data, no download, no GPU: a hand-written "model" whose reasoning we KNOW (it only looks
at the word "gaddar"), a tiny BiLSTM, and a tiny random BERT, all on made-up posts.
"""

# json: fake processed files and output checks.
import json
# sys / Path: import the scripts.
import sys
from pathlib import Path

# numpy: score arrays.
import numpy as np
# pytest: test runner.
import pytest

# Pure-Python parts (no torch needed).
from xhate.evaluation.faithfulness import faithfulness, plausibility, token_f1, top_k_indices
from xhate.explain import explain_posts, scores_for_prediction
from xhate.explain.attention import subword_to_word
from xhate.explain.visualize import highlight_html

# Make scripts/ importable (end-to-end tests).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class KeywordModel:
    """A fake predictor that ONLY uses the word "gaddar": P(HOF) = 0.9 with it, 0.1 without.

    Because we know exactly what drives it, we know what a faithful explanation must say.
    """

    def predict_proba(self, token_lists):
        """(N, 2) probabilities."""
        p_hof = np.array([0.9 if "gaddar" in toks else 0.1 for toks in token_lists])
        return np.stack([1 - p_hof, p_hof], axis=1)


POSTS = [["ye", "log", "gaddar", "hai"], ["tum", "sab", "gaddar", "ho", "yaar"], ["aaj", "mausam", "accha", "hai"]]


# ------------------------------------------------------------- top-k / helpers
def test_top_k_sizes_and_ties():
    """k = ceil(fraction * n), at least 1; ties go to the earlier word."""
    scores = np.zeros(20)
    assert len(top_k_indices(scores, fraction=0.01)) == 1   # 1% of 20 -> 1
    assert len(top_k_indices(scores, fraction=0.20)) == 4   # 20% of 20 -> 4
    assert top_k_indices([0.5, 0.9, 0.9, 0.1], k=2).tolist() == [1, 2]
    assert top_k_indices([], fraction=0.5).tolist() == []


def test_subword_to_word_sums_and_renormalises():
    """[CLS] you are tr ##ash: pieces of "trash" are added, specials dropped."""
    out = subword_to_word([0.3, 0.1, 0.2, 0.2, 0.2], [None, 0, 1, 2, 2], 3)
    assert out == pytest.approx([0.1 / 0.7, 0.2 / 0.7, 0.4 / 0.7])
    # A word cut off by truncation gets 0.
    assert subword_to_word([0.5, 0.5], [None, 0], 2).tolist() == [1.0, 0.0]


def test_scores_for_prediction_flips_lime_for_not():
    """LIME is signed towards HOF; for a NOT prediction the sign is flipped."""
    assert scores_for_prediction([0.5, -0.2], "lime", 0).tolist() == [-0.5, 0.2]
    assert scores_for_prediction([0.5, -0.2], "lime", 1).tolist() == [0.5, -0.2]
    assert scores_for_prediction([0.5, 0.2], "attention", 0).tolist() == [0.5, 0.2]


# ------------------------------------------------------------- faithfulness
def test_faithful_explanation_beats_random_and_wrong_one():
    """Pointing at "gaddar" -> high comprehensiveness, ~0 sufficiency; pointing elsewhere -> the reverse."""
    model, hof_posts = KeywordModel(), POSTS[:2]
    right = [np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0, 0])]
    wrong = [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0, 0])]
    good = faithfulness(model, hof_posts, right)
    bad = faithfulness(model, hof_posts, wrong)
    # Deleting "gaddar" drops P(HOF) from 0.9 to 0.1 in EVERY bin (it is always ranked first).
    assert good["comprehensiveness"] == pytest.approx(0.8)
    # Keeping only "gaddar" keeps P(HOF) at 0.9.
    assert good["sufficiency"] == pytest.approx(0.0)
    assert bad["comprehensiveness"] < good["comprehensiveness"]
    assert bad["sufficiency"] > good["sufficiency"]
    # Per-bin and per-post details are there.
    assert set(good["per_bin"]) == {"0.01", "0.05", "0.1", "0.2", "0.5"}
    assert len(good["per_post"]) == 2 and good["n_posts"] == 2


def test_faithfulness_rejects_wrong_length_scores():
    """A score array that does not match the tokens is an error, not a silent mismatch."""
    with pytest.raises(ValueError):
        faithfulness(KeywordModel(), [["a", "b"]], [np.array([1.0])])


# ------------------------------------------------------------- plausibility
def test_token_f1_hand_example():
    """P = 1/2, R = 1/2 -> F1 = 0.5; no overlap -> 0."""
    assert token_f1([1, 1, 0, 0], [0, 1, 1, 0]) == pytest.approx(0.5)
    assert token_f1([1, 0], [0, 1]) == 0.0


def test_plausibility_uses_average_k_and_skips_empty_rationales():
    """k = average human rationale size; posts with no marked word are ignored."""
    scores = [np.array([0.1, 0.9, 0.0]), np.array([0.8, 0.2, 0.1, 0.0]), np.array([0.5, 0.5])]
    gold = [[0, 1, 0], [1, 1, 0, 0], [0, 0]]  # sizes 1 and 2 -> k = round(1.5) = 2; third skipped
    out = plausibility(scores, gold)
    assert out["n_posts"] == 2 and out["k"] == 2
    # Post 1: top-2 = {1, 0} vs {1} -> F1 2/3. Post 2: top-2 = {0, 1} vs {0, 1} -> 1.
    assert out["token_f1"] == pytest.approx((2 / 3 + 1) / 2)
    # Both rankings put the gold words first -> average precision 1.
    assert out["auprc"] == pytest.approx(1.0)
    assert plausibility([np.array([1.0])], [None])["n_posts"] == 0


# ------------------------------------------------------------- LIME / random
def test_lime_finds_the_word_the_model_uses():
    """On the keyword model, LIME's top word is "gaddar"; NOT posts / empty posts work too."""
    pytest.importorskip("lime")
    scores = explain_posts(KeywordModel(), POSTS + [[]], "lime", lime_samples=200)
    assert int(np.argmax(scores[0])) == 2 and int(np.argmax(scores[1])) == 2
    assert len(scores[2]) == 4 and len(scores[3]) == 0
    # Devanagari words stay whole (no matra splitting): one score per token.
    hindi = ["ये", "लोग", "गद्दार", "हैं"]
    assert len(explain_posts(KeywordModel(), [hindi], "lime", lime_samples=50)[0]) == 4


def test_random_scores_are_seeded_and_unknown_method_fails():
    """Same seed -> same random baseline; a typo in the method name raises."""
    a = explain_posts(KeywordModel(), POSTS, "random", seed=1)
    b = explain_posts(KeywordModel(), POSTS, "random", seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    with pytest.raises(ValueError):
        explain_posts(KeywordModel(), POSTS, "gradients")


# ------------------------------------------------------------- visualisation
def test_highlight_escapes_html_and_underlines_gold():
    """"<user>" is shown as text, not read as an HTML tag; human words are underlined."""
    out = highlight_html(["<user>", "gaddar"], [0.0, 1.0], gold=[0, 1])
    assert "&lt;user&gt;" in out and "<user>" not in out
    assert out.count("gold") == 1


# ======================================================== real (tiny) models
torch = pytest.importorskip("torch")

from xhate.explain.attention import attention_scores  # noqa: E402
from xhate.explain.predictor import Predictor  # noqa: E402
from xhate.models.baseline import BiLSTMClassifier, CNNClassifier  # noqa: E402
from xhate.models.vocab import Vocab  # noqa: E402


@pytest.fixture()
def bilstm_predictor():
    """A random tiny BiLSTM with attention, wrapped as a Predictor (max 4 words)."""
    torch.manual_seed(0)
    vocab = Vocab.build([toks for toks in POSTS])
    model = BiLSTMClassifier(len(vocab), embed_dim=8, hidden_dim=4)
    return Predictor(model, "baseline", vocab=vocab, max_length=4)


def test_baseline_predictor_probabilities_and_empty_post(bilstm_predictor):
    """Rows sum to 1, an empty post still gets a prediction, batching does not change results."""
    probs = bilstm_predictor.predict_proba(POSTS + [[]])
    assert probs.shape == (4, 2) and np.allclose(probs.sum(axis=1), 1.0)
    bilstm_predictor.batch_size = 1
    assert np.allclose(bilstm_predictor.predict_proba(POSTS), probs[:3], atol=1e-6)


def test_bilstm_attention_per_word_and_truncation(bilstm_predictor):
    """One weight per word, summing to 1; the 5th word (beyond max_length 4) gets 0."""
    scores = attention_scores(bilstm_predictor, POSTS)
    assert [len(s) for s in scores] == [4, 5, 4]
    assert all(s.sum() == pytest.approx(1.0) for s in scores)
    assert scores[1][4] == 0.0


def test_cnn_has_no_attention():
    """The CNN cannot be explained with attention (LIME still works)."""
    vocab = Vocab.build(POSTS)
    pred = Predictor(CNNClassifier(len(vocab), embed_dim=8, num_filters=2), "baseline", vocab=vocab)
    with pytest.raises(ValueError):
        attention_scores(pred, POSTS)


@pytest.fixture(scope="module")
def tiny_bert_dir(tmp_path_factory):
    """A tiny random BERT + tokenizer saved to a folder (usable as model.name). No download."""
    transformers = pytest.importorskip("transformers")
    folder = tmp_path_factory.mktemp("tiny_bert_explain")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "ye", "log", "gadd", "##ar", "hai", "tum", "accha"]
    # write_bytes: exact "\n" line ends on every OS (the vocab file is read line by line).
    (folder / "vocab.txt").write_bytes("\n".join(vocab).encode("utf-8"))
    transformers.BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True).save_pretrained(folder)
    torch.manual_seed(0)
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    transformers.BertModel(config).save_pretrained(folder)
    return folder


def test_transformer_attention_is_per_word(tiny_bert_dir):
    """"gaddar" is 2 sub-words but gets ONE score; scores sum to 1 over the words."""
    from transformers import AutoTokenizer

    from xhate.models.transformer import TransformerClassifier
    tok = AutoTokenizer.from_pretrained(str(tiny_bert_dir))
    pred = Predictor(TransformerClassifier(str(tiny_bert_dir)), "transformer", tokenizer=tok, max_length=16)
    posts = [["ye", "log", "gaddar", "hai"], []]
    scores = attention_scores(pred, posts)
    assert len(scores[0]) == 4 and scores[0].sum() == pytest.approx(1.0)
    assert len(scores[1]) == 0
    assert pred.predict_proba(posts).shape == (2, 2)


# ======================================================== end to end
def toy_records(n=24):
    """Easy posts in Member 1's format: HOF contains "gaddar", NOT contains "accha"."""
    records = []
    for i in range(n):
        label = i % 2
        tokens = ["ye", "log"] + ["bahut"] * (i % 3) + ["gaddar" if label else "accha", "hai"]
        records.append({"id": f"p{i}", "text": " ".join(tokens), "tokens": tokens, "label": label,
                        "has_rationale": False, "rationale": None, "rationale_soft": None})
    return records


@pytest.fixture()
def trained_bilstm(tmp_path):
    """Train the real BiLSTM config for 2 epochs on toy data; return (run folder, processed folder)."""
    import train
    processed = tmp_path / "processed"
    for split in ("train", "val", "test"):
        path = processed / "hasoc" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in toy_records()), encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    train.main(["--config", str(repo / "configs" / "baseline_bilstm.yaml"), "--results-dir", str(tmp_path / "results"),
                "--set", f"data.processed_dir={processed}", "data.min_freq=1", "train.epochs=2",
                "model.embed_dim=8", "model.hidden_dim=4", "run_name=toy"])
    return tmp_path / "results" / "toy", processed


def test_load_run_gives_the_same_predictions_as_training(trained_bilstm):
    """load_run rebuilds the model: its P(HOF) matches test_predictions.jsonl written by train.py."""
    from xhate.runs import load_run
    run_dir, _ = trained_bilstm
    run = load_run(run_dir)
    assert run.kind == "baseline" and run.model_type == "bilstm" and not run.model.training
    saved = [json.loads(line) for line in (run_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    probs = Predictor.from_run(run).predict_proba([r["tokens"] for r in toy_records()])
    assert np.allclose(probs[:, 1], [s["prob_hof"] for s in saved], atol=1e-4)


def test_load_run_transformer_without_download(tmp_path, tiny_bert_dir):
    """A transformer run reloads from config.yaml + best.pt (architecture from AutoConfig)."""
    from xhate.models.transformer import TransformerClassifier
    from xhate.runs import load_run
    from xhate.utils import save_config
    model = TransformerClassifier(str(tiny_bert_dir))
    torch.save(model.state_dict(), tmp_path / "best.pt")
    save_config({"model": {"type": "transformer", "name": str(tiny_bert_dir)}, "data": {"max_length": 16}},
                tmp_path / "config.yaml")
    run = load_run(tmp_path)
    ids = torch.tensor([[2, 5, 6, 3]])
    mask = torch.ones_like(ids)
    model.eval()
    assert run.kind == "transformer" and run.max_length == 16
    assert torch.allclose(run.model(ids, mask), model(ids, mask), atol=1e-5)


def test_explain_script_end_to_end(trained_bilstm):
    """scripts/explain.py writes metrics.json, explanations.jsonl and report.html."""
    pytest.importorskip("lime")
    import explain
    run_dir, processed = trained_bilstm
    metrics = explain.main(["--run", str(run_dir), "--processed-dir", str(processed), "--lime-samples", "30",
                            "--max-posts", "10", "--annotations", str(run_dir / "none.jsonl")])
    out = run_dir / "explain_hasoc_test"
    for name in ("metrics.json", "explanations.jsonl", "report.html"):
        assert (out / name).exists(), name
    assert set(metrics["methods"]) == {"attention", "lime", "random"} and metrics["n_posts"] == 10
    # HASOC without annotations: faithfulness yes, plausibility not computable.
    assert metrics["methods"]["lime"]["plausibility"]["n_posts"] == 0
    assert metrics["methods"]["lime"]["faithfulness"]["all"]["comprehensiveness"] is not None
    lines = (out / "explanations.jsonl").read_text(encoding="utf-8").splitlines()
    post = json.loads(lines[0])
    assert len(lines) == 10 and len(post["scores"]["lime"]) == len(post["tokens"])


def test_explanation_table_groups_seeds(tmp_path):
    """Two seeds of one configuration become one row with mean ± std."""
    import explanation_table
    for seed, comp in ((42, 0.4), (43, 0.6)):
        out = tmp_path / f"muril_vanilla_seed{seed}" / "explain_hasoc_test"
        out.mkdir(parents=True)
        method = {"faithfulness": {"all": {"comprehensiveness": comp, "sufficiency": 0.1},
                                   "predicted_hof": {"comprehensiveness": comp, "sufficiency": 0.1}},
                  "plausibility": {"token_f1": None, "auprc": None}}
        (out / "metrics.json").write_text(json.dumps({"run_name": f"muril_vanilla_seed{seed}", "lambda": 0.0,
                                                      "methods": {"lime": method}}), encoding="utf-8")
    table = explanation_table.main(["--results-dir", str(tmp_path)])
    assert len(table) == 1 and table[0]["configuration"] == "muril_vanilla" and table[0]["seeds"] == 2
    assert table[0]["comp"].startswith("0.5000 ± 0.1414") and table[0]["token_f1"] == "–"


# ======================================================== annotations
def test_parse_rationale_numbers_ranges_and_errors():
    """ "1 3-4" -> words 1, 3, 4; out-of-range numbers are rejected."""
    from sample_for_annotation import parse_rationale
    assert parse_rationale("1, 3-4", 5) == [0, 1, 0, 1, 1]
    assert parse_rationale("", 3) == [0, 0, 0]
    with pytest.raises(ValueError):
        parse_rationale("7", 3)


def test_annotation_sheets_round_trip(tmp_path):
    """Sample HOF test posts into sheets, fill one, collect it, attach it back to the posts."""
    import csv

    import sample_for_annotation as sfa

    from xhate.data import load_split
    from xhate.evaluation.annotations import attach_annotations, load_annotations
    processed = tmp_path / "processed"
    path = processed / "hasoc" / "test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in toy_records(20)), encoding="utf-8")
    sheets, out = tmp_path / "sheets", tmp_path / "ann.jsonl"
    common = ["--processed-dir", str(processed), "--sheets-dir", str(sheets), "--out", str(out)]
    paths = sfa.main(["--n", "10"] + common)
    # 10 HOF posts over 4 sheets: 3 / 3 / 2 / 2; NOT posts are never sampled.
    rows = [list(csv.DictReader(p.open(encoding="utf-8-sig"))) for p in paths]
    assert [len(r) for r in rows] == [3, 3, 2, 2]
    # Making sheets again without --force refuses to overwrite filled work.
    with pytest.raises(SystemExit):
        sfa.main(["--n", "10"] + common)
    # Member 1 marks the word "gaddar" in their first post.
    first = rows[0][0]
    position = first["numbered_words"].split(" | ").index(
        next(w for w in first["numbered_words"].split(" | ") if w.endswith("=gaddar")))
    rows[0][0]["rationale"] = str(position)
    with paths[0].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sfa.COLUMNS)
        writer.writeheader()
        writer.writerows(rows[0])
    collected = sfa.main(["--collect"] + common)
    assert len(collected) == 1 and "text" not in collected[0]  # licence: no text in the committed file
    # Attach to the processed posts: the marked word is "gaddar".
    records = load_split("hasoc", "test", processed)
    assert attach_annotations(records, load_annotations(out)) == 1
    rec = next(r for r in records if r.get("annotator"))
    assert rec["has_rationale"] and rec["tokens"][rec["rationale"].index(1)] == "gaddar"


def test_attach_annotations_refuses_misaligned_mask():
    """If the post's token count changed, the old mask is not used."""
    from xhate.evaluation.annotations import attach_annotations
    records = [{"id": "x", "tokens": ["a", "b", "c"]}]
    with pytest.warns(UserWarning):
        assert attach_annotations(records, {"x": {"id": "x", "n_tokens": 2, "rationale": [1, 0]}}) == 0
