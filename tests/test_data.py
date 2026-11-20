"""Tests for xhate.data (Member 1). Run with:  pytest tests/test_data.py

All tests use tiny made-up examples written to a temporary folder, so they run
without the real (licensed) data.
"""

# json: to write fake HateXplain files.
import json

# pytest: test runner (tmp_path gives a fresh temporary folder per test).
import pytest

# Functions under test.
from xhate.data import HOF, NOT, clean_text, load_split, normalize_tokens, to_binary, write_jsonl
from xhate.data.hasoc import load_hasoc
from xhate.data.hatexplain import aggregate_rationales, load_hatexplain, majority_label


# ----------------------------------------------------------------- labels
def test_to_binary_maps_all_datasets():
    """HASOC and HateXplain labels map to the same binary scheme."""
    # HASOC task A.
    assert to_binary("HOF") == HOF and to_binary("NOT") == NOT
    # HateXplain (any case, extra spaces).
    assert to_binary(" hatespeech ") == HOF and to_binary("Offensive") == HOF
    # HateXplain normal.
    assert to_binary("normal") == NOT


def test_to_binary_rejects_unknown():
    """An unknown label fails loudly instead of being guessed."""
    # Expect a ValueError.
    with pytest.raises(ValueError):
        to_binary("maybe")


# ------------------------------------------------------------ preprocess
def test_clean_text_placeholders_hashtags_emoji():
    """URLs, mentions, hashtags, emojis and RT are handled as decided in issue #1."""
    # Run the cleaner on a typical tweet.
    tokens = clean_text("RT @abc: Tum log nhi sudhroge!!! 😡 #ModiHatao http://t.co/x")
    # Check the exact result.
    assert tokens == ["<user>", "tum", "log", "nahi", "sudhroge", "!!",
                      "emoji_enraged_face", "modi", "hatao", "<url>"]


def test_clean_text_repeats_and_spelling():
    """Repeated letters are squashed and spelling variants are normalised."""
    # "bahuuuut" -> "bahut" (via the 1-repeat fallback), "yaaar" -> "yaar", "h" -> "hai".
    assert clean_text("bahuuuut accha h yaaar") == ["bahut", "accha", "hai", "yaar"]


def test_clean_text_keeps_devanagari():
    """Hindi script is kept as-is (only punctuation is split off)."""
    # Devanagari words, including vowel signs, stay in one piece.
    assert clean_text("तुम लोग देशद्रोही हो") == ["तुम", "लोग", "देशद्रोही", "हो"]


def test_clean_text_devanagari_hashtag_and_mention():
    """Hindi hashtags / mentions are not cut at vowel signs (bug seen on real HASOC 2019)."""
    # "#शांतीदूत" keeps all its letters; "@राहुल_गांधी" becomes one <user>.
    assert clean_text("सब #शांतीदूत के साथ, @राहुल_गांधी जी") == ["सब", "शांतीदूत", "के", "साथ", "<user>", "जी"]


def test_clean_text_drops_lone_hash_and_at():
    """A lone "#" / "@" or "#@#" censoring is not kept as a token."""
    # Truncated tweet ending in "#", and a censored word "#@#".
    assert clean_text("ye #@# hai @ bas #") == ["ye", "hai", "bas"]


def test_clean_text_handles_missing():
    """NaN / None become an empty token list."""
    # None input.
    assert clean_text(None) == []
    # NaN input (how pandas stores empty cells).
    assert clean_text(float("nan")) == []


def test_normalize_tokens_never_changes_length():
    """The most important rule: rationale alignment needs a 1-to-1 mapping."""
    # Tricky tokens: emoji glued to text, hashtag, URL, mention, punctuation, empty-ish.
    tokens = ["<user>", "#MAGA", "lol😂", "sooooo", "http://x.com", "@bob", "nhi", "!!!!", "👍🏽"]
    # Normalise.
    out = normalize_tokens(tokens)
    # Same length as the input.
    assert len(out) == len(tokens)
    # No token became empty or contains a space.
    assert all(tok and " " not in tok for tok in out)
    # Spot-check a few values.
    assert out[:2] == ["<user>", "maga"] and out[4:7] == ["<url>", "<user>", "nahi"]


# ------------------------------------------------------------ hatexplain
def test_majority_label():
    """Majority vote, with None when all annotators disagree."""
    # Two of three agree.
    assert majority_label(["offensive", "offensive", "normal"]) == "offensive"
    # All three disagree.
    assert majority_label(["offensive", "hatespeech", "normal"]) is None


def test_aggregate_rationales_soft_and_hard():
    """Soft = fraction of lists; hard = soft >= 0.5; wrong-length lists are skipped."""
    # Two valid lists and one with the wrong length.
    hard, soft, n_used = aggregate_rationales([[1, 0, 1], [1, 0, 0], [1, 1]], 3)
    # Only the two valid lists are used.
    assert n_used == 2
    # Fractions per token.
    assert soft == [1.0, 0.0, 0.5]
    # 0.5 counts as "at least half".
    assert hard == [1, 0, 1]


def _write_fake_hatexplain(folder):
    """Create a 3-post fake HateXplain release in ``folder``."""
    # Three posts: HOF with rationales, NOT, and one with no majority.
    posts = {
        "p1": {"post_id": "p1", "post_tokens": ["you", "are", "trash"],
               "annotators": [{"label": "offensive"}, {"label": "offensive"}, {"label": "normal"}],
               "rationales": [[0, 0, 1], [0, 1, 1]]},
        "p2": {"post_id": "p2", "post_tokens": ["nice", "day"],
               "annotators": [{"label": "normal"}] * 3, "rationales": []},
        "p3": {"post_id": "p3", "post_tokens": ["hmm"],
               "annotators": [{"label": "normal"}, {"label": "offensive"}, {"label": "hatespeech"}],
               "rationales": []},
    }
    # Official split ids.
    divisions = {"train": ["p1", "p3"], "val": ["p2"], "test": []}
    # Write both files.
    (folder / "dataset.json").write_text(json.dumps(posts), encoding="utf-8")
    (folder / "post_id_divisions.json").write_text(json.dumps(divisions), encoding="utf-8")


def test_load_hatexplain_utf16_files(tmp_path):
    """Files saved as UTF-16 (Windows PowerShell "> file") are read correctly."""
    # Create normal UTF-8 fake files first.
    _write_fake_hatexplain(tmp_path)
    # Re-save both files as UTF-16 (with the FF FE byte-order mark), like PowerShell 5.1 does.
    for name in ("dataset.json", "post_id_divisions.json"):
        # Read the UTF-8 text.
        text = (tmp_path / name).read_text(encoding="utf-8")
        # Write it back as UTF-16.
        (tmp_path / name).write_text(text, encoding="utf-16")
    # Loading must still work and give the same result as UTF-8.
    splits, _ = load_hatexplain(tmp_path)
    # p1 in train, p2 in val (p3 dropped: no majority).
    assert [r["id"] for r in splits["train"]] == ["p1"] and [r["id"] for r in splits["val"]] == ["p2"]


def test_load_hasoc_utf8_bom(tmp_path):
    """A CSV saved with a UTF-8 BOM (Excel / Notepad) keeps its id column."""
    # BOM + header + one row.
    (tmp_path / "train.csv").write_text("text_id,text,task_1\nz1,tum gadhe ho,HOF\n", encoding="utf-8-sig")
    # Load the folder.
    splits, _ = load_hasoc(tmp_path)
    # The id comes from the file (not a generated "<source>#0" id).
    assert splits["train"][0]["id"] == "z1"


def test_load_hatexplain(tmp_path):
    """End to end on a fake release: voting, dropping, rationale aggregation."""
    # Create the fake files.
    _write_fake_hatexplain(tmp_path)
    # Load them.
    splits, stats = load_hatexplain(tmp_path)
    # p3 (no majority) is dropped.
    assert stats["dropped_no_majority"] == 1 and len(splits["train"]) == 1
    # p1 is HOF with an aggregated rationale.
    p1 = splits["train"][0]
    assert p1["label"] == HOF and p1["has_rationale"]
    assert p1["rationale_soft"] == [0.0, 0.5, 1.0] and p1["rationale"] == [0, 1, 1]
    # p2 is NOT, with all-zero rationale.
    p2 = splits["val"][0]
    assert p2["label"] == NOT and not p2["has_rationale"] and p2["rationale"] == [0, 0]


# ----------------------------------------------------------------- hasoc
def test_load_hasoc_autodetects_columns_and_split(tmp_path):
    """A 2019-style TSV (train) and a 2020-style CSV (test) are both understood."""
    # Folder for the 2019 file.
    (tmp_path / "2019").mkdir()
    # 2019 layout: text_id / text / task_1, tab-separated. Row 3 duplicates row 1.
    (tmp_path / "2019" / "hindi_dataset.tsv").write_text(
        "text_id\ttext\ttask_1\n"
        "a1\ttum gadhe ho\tHOF\n"
        "a2\taaj mausam accha hai\tNOT\n"
        "a3\ttum gadhe ho\tHOF\n"
        "a4\tleak post\tNOT\n",
        encoding="utf-8")
    # 2020 layout: tweet_id / text / task1, comma-separated, "test" in the name.
    (tmp_path / "hasoc_2020_hi_test.csv").write_text(
        "tweet_id,text,task1\nb1,bakwas band karo,HOF\nb2,leak post,NOT\n", encoding="utf-8")
    # Load everything.
    splits, stats = load_hasoc(tmp_path)
    # Train: duplicate a3 removed, leaked "leak post" removed -> a1, a2.
    assert [r["id"] for r in splits["train"]] == ["a1", "a2"]
    # Test: both rows.
    assert [r["id"] for r in splits["test"]] == ["b1", "b2"]
    # Source records the year.
    assert splits["train"][0]["source"] == "hasoc2019/hindi_dataset.tsv"
    # Labels mapped to binary.
    assert [r["label"] for r in splits["test"]] == [HOF, NOT]


# --------------------------------------------------------------- dataset
def _tiny_wordpiece_tokenizer():
    """Build a small BERT-style (WordPiece) fast tokenizer OFFLINE, no internet needed."""
    # Skip these tests if transformers / tokenizers are not installed.
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    # Vocabulary: special tokens, whole words, and "##" continuation pieces.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "you", "are", "tr", "##ash", "nice", "day"]
    # WordPiece model: unknown words become [UNK]; "trash" -> "tr" + "##ash".
    model = tokenizers.models.WordPiece({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]")
    # Wrap the model in a tokenizer object.
    tok = tokenizers.Tokenizer(model)
    # Split input on whitespace before WordPiece.
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    # Add [CLS] ... [SEP] around every input, like BERT / MuRIL.
    tok.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    # Turn it into a HuggingFace fast tokenizer (needed for word_ids()).
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                                cls_token="[CLS]", sep_token="[SEP]")


def test_dataset_aligns_rationale_to_subwords():
    """Each sub-word gets its word's rationale; [CLS]/[SEP] are masked out."""
    # Skip if torch is not installed.
    pytest.importorskip("torch")
    # Import here so the rest of the file works without torch.
    from xhate.data.dataset import HateSpeechDataset
    # One HOF post whose last word "trash" splits into two sub-words.
    rec = {"tokens": ["you", "are", "trash"], "label": 1, "has_rationale": True,
           "rationale": [0, 1, 1], "rationale_soft": [0.0, 0.5, 1.0]}
    # Build the dataset with the offline tokenizer.
    item = HateSpeechDataset([rec], _tiny_wordpiece_tokenizer())[0]
    # Sub-words: [CLS] you are tr ##ash [SEP]
    assert item["input_ids"].tolist() == [2, 4, 5, 6, 7, 3]
    # Soft rationale copied to both pieces of "trash"; specials get 0.
    assert item["rationale"].tolist() == [0.0, 0.0, 0.5, 1.0, 1.0, 0.0]
    # Only real sub-words count in the loss.
    assert item["rationale_mask"].tolist() == [0, 1, 1, 1, 1, 0]
    # Label is kept.
    assert item["label"].item() == 1


def test_dataset_no_rationale_and_collate_padding():
    """Posts without a rationale are fully masked; collate pads to the longest post."""
    # Skip if torch is not installed.
    pytest.importorskip("torch")
    # Import the Dataset and the collate function.
    from xhate.data.dataset import HateSpeechDataset, collate_batch
    # A HASOC-style post (no rationale) and a longer HateXplain-style post.
    recs = [{"tokens": ["nice", "day"], "label": 0, "has_rationale": False, "rationale": None, "rationale_soft": None},
            {"tokens": ["you", "are", "trash"], "label": 1, "has_rationale": True,
             "rationale": [0, 1, 1], "rationale_soft": [0.0, 0.5, 1.0]}]
    # Build the dataset.
    ds = HateSpeechDataset(recs, _tiny_wordpiece_tokenizer())
    # The post without a rationale is never used in the rationale loss.
    assert ds[0]["rationale_mask"].sum().item() == 0
    # Batch both posts together.
    batch = collate_batch([ds[0], ds[1]], pad_token_id=0)
    # Both rows padded to the longer length (6 sub-words).
    assert batch["input_ids"].shape == (2, 6) and batch["rationale"].shape == (2, 6)
    # Padding positions have attention 0 and rationale_mask 0.
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert batch["rationale_mask"][0].tolist() == [0, 0, 0, 0, 0, 0]
    # Labels stacked in order.
    assert batch["label"].tolist() == [0, 1]


# ---------------------------------------------------------------- loader
def test_load_split_roundtrip(tmp_path):
    """write_jsonl + load_split give back the same records (Hindi included)."""
    # One record with Devanagari text.
    rec = {"id": "x", "text": "तुम", "tokens": ["तुम"], "label": 1}
    # Write it where load_split expects it.
    write_jsonl([rec], tmp_path / "hasoc" / "train.jsonl")
    # Read it back.
    assert load_split("hasoc", "train", processed_dir=tmp_path) == [rec]


def test_load_split_missing_file_message(tmp_path):
    """A missing file tells the user which command to run."""
    # Expect FileNotFoundError mentioning prepare_data.py.
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_split("hasoc", "val", processed_dir=tmp_path)
