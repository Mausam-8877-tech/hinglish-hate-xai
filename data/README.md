# Data

Raw and processed data are **git-ignored**. Every member downloads the data on their own machine.

```
data/
├── raw/
│   ├── hasoc/          HASOC files, as downloaded (any year, sub-folders allowed)
│   └── hatexplain/     dataset.json + post_id_divisions.json
├── processed/          cleaned data, written by scripts/prepare_data.py
│   ├── hasoc/          train.jsonl  val.jsonl  test.jsonl
│   ├── hatexplain/     train.jsonl  val.jsonl  test.jsonl
│   └── stats.json      counts and class balance per split
├── sample/             tiny SYNTHETIC sample in the processed format (committed)
└── annotations/        our own Hinglish rationale annotations (committed)
```

## 1. Download

### HASOC (Hindi / Hinglish)
Register at <https://hasocfire.github.io>, choose a year, and download the Hindi track.
Put the files in `data/raw/hasoc/` (sub-folders such as `2019/` are fine).

The loader **auto-detects** the format:

| What | Accepted |
|---|---|
| File types | `.csv`, `.tsv`, `.xlsx`, `.xls` |
| Text column | `text`, `tweet`, `tweet_text`, `post`, `content`, `sentence` |
| Label column (task A) | `task_1`, `task1`, `task_a`, `subtask_a`, `label`, `class` |
| Id column (optional) | `text_id`, `tweet_id`, `_id`, `id`, `hasoc_id` |
| Split | file name contains `test` → **test**, otherwise **train** |

Files without a text + label column are skipped with a warning.

- **Extract the zips first.** A `.zip` left in the folder is not read (the script tells you).
- Extracting with "Extract here" is fine: the extra `__MACOSX/` folder and `._*.tsv` files
  (macOS metadata, not data) are ignored automatically.
- The year in `source` comes from the folder name (`2019/`) or, if missing, from another file's
  name in the same folder (`hasoc2019_hi_test_gold_2919.tsv`).

### HateXplain (English, with rationales)
Download `dataset.json` and `post_id_divisions.json` from
<https://github.com/hate-alert/HateXplain/tree/master/Data> into `data/raw/hatexplain/`.

## 2. Prepare

```bash
python scripts/prepare_data.py                         # both datasets
python scripts/prepare_data.py --datasets hatexplain   # just one
python scripts/prepare_data.py --val-size 0.1 --seed 42
```

A dataset with no raw files is skipped with a message, so you can run this before you have HASOC.

## 3. Load (for Members 2, 3 and 4)

```python
from xhate.data import load_split, load_all, load_dataframe

train = load_split("hasoc", "train")          # list of dicts
splits = load_all("hatexplain")               # {"train": [...], "val": [...], "test": [...]}
df = load_dataframe("hasoc", "test")          # pandas DataFrame
```

For transformers, `xhate.data.dataset.HateSpeechDataset` + `collate_batch` (needs torch) give
`input_ids`, `attention_mask`, `label`, and sub-word-aligned `rationale` / `rationale_mask`.
For XLM-R pass the right pad id: `functools.partial(collate_batch, pad_token_id=tokenizer.pad_token_id)`.

### ⚠️ Hindi (Devanagari) tokenisation: use `tokens`, or split on spaces

Many libraries split words with the regex `\w`, which does **not** match Hindi vowel signs (matras),
so Hindi words are cut into pieces (`हरामी` → `ह`, `र`, `म`, …). We found this in end-to-end testing:

| Tool | Default behaviour | Use instead |
|---|---|---|
| scikit-learn `TfidfVectorizer` / `CountVectorizer` | breaks Hindi words (macro-F1 on HASOC 0.753 vs **0.794**) | `TfidfVectorizer(analyzer=lambda toks: toks)` on the `tokens` column, or `analyzer=str.split` on `text` |
| LIME `LimeTextExplainer` | explains letters like `ह`, `म` | `LimeTextExplainer(split_expression=r"\s+", bow=False)` on `text` |
| Your own vocab (CNN / LSTM) | `re.findall(r"\w+", ...)` breaks words | use the `tokens` list as-is |

`text` is always `" ".join(tokens)`, so splitting `text` on spaces gives back exactly `tokens`.

## 4. Label scheme

One **binary** scheme for both datasets (`xhate/data/labels.py`):

| id | name | HASOC | HateXplain |
|---|---|---|---|
| 0 | `NOT` | `NOT` (task B: `NONE`) | `normal` |
| 1 | `HOF` | `HOF` (task B: `HATE`, `OFFN`, `PRFN`) | `hatespeech`, `offensive` |

HateXplain: the majority vote of the 3 annotators is taken on the original 3 classes, then mapped.
Posts with no majority are dropped (the official `post_id_divisions.json` already leaves them out).

## 5. Processed format

Each line of a `.jsonl` file is one JSON object:

| Key | Type | Description |
|---|---|---|
| `id` | str | original post id (generated as `<source>#<row>` if the file has none) |
| `dataset` | str | `"hasoc"` or `"hatexplain"` |
| `source` | str | release / file, e.g. `"hasoc2019/hindi_dataset.tsv"` |
| `lang` | str | `"hi-en"` (HASOC) or `"en"` (HateXplain) |
| `split` | str | `"train"`, `"val"` or `"test"` |
| `text` | str | cleaned text = `" ".join(tokens)` |
| `raw_text` | str | original text |
| `tokens` | list[str] | cleaned tokens |
| `label` | int | `0` = NOT, `1` = HOF |
| `has_rationale` | bool | `true` only for HOF HateXplain posts with at least one valid rationale list |
| `rationale` | list[int] \| null | hard 0/1 per token: 1 if **≥ half** of the rationale lists marked it |
| `rationale_soft` | list[float] \| null | fraction of rationale lists that marked each token (0–1) |

`rationale` and `rationale_soft` always have **exactly** `len(tokens)` entries. They are `null` for
HASOC (no rationales) and all zeros for NOT posts in HateXplain.

Example (HateXplain):

```json
{"id": "p1", "dataset": "hatexplain", "source": "hatexplain", "lang": "en", "split": "train",
 "text": "you are trash", "raw_text": "you are trash", "tokens": ["you", "are", "trash"],
 "label": 1, "has_rationale": true, "rationale": [0, 1, 1], "rationale_soft": [0.0, 0.5, 1.0]}
```

## 6. Preprocessing

`xhate/data/preprocess.py`:

| Input | Output |
|---|---|
| URL | `<url>` |
| `@mention` | `<user>` |
| `#ModiHatao` | `modi hatao` (HASOC); `modihatao` (HateXplain: no splitting, to keep the token count) |
| emoji 😡 | `emoji_enraged_face` |
| upper case | lower case (Devanagari is kept as-is) |
| `sooooo` | `soo` (3+ repeated characters → 2) |
| `nhi`, `nai`, `nahin` | `nahi` (editable dictionary `SPELLING_VARIANTS`) |
| leading `RT` | removed (HASOC) |

HateXplain is already tokenised, so it uses `normalize_tokens`, which maps every token to exactly one
token. **The token count never changes**, so the rationales stay aligned.

## 7. Splits

- **HASOC:** the official test file(s) are the test set. Validation is a stratified 10 % of train
  (`--seed 42`). Duplicate posts are removed, and training posts that also appear in test are removed.
  If no file has `test` in its name, a stratified test set is also carved out of train (with a warning).
- **HateXplain:** the official `post_id_divisions.json` split.

## 8. Fake HASOC (until you have the password)

```bash
python scripts/make_fake_hasoc.py                                    # ~500 FAKE posts -> data/raw/hasoc_fake/
python scripts/prepare_data.py --datasets hasoc --raw-dir data/raw/hasoc_fake
```

This writes `data/processed/hasoc/` from template-generated posts in the real 2019 / 2020 / 2021 file
layouts (with about 10 % of the labels flipped on purpose), so `load_split("hasoc", ...)` already works.
Every row has `FAKE` in its `source`. **Never report results on it.** Once the real data is in
`data/raw/hasoc/`, run `python scripts/prepare_data.py --datasets hasoc` to overwrite it.

## 9. Synthetic sample

`data/sample/hasoc_sample.jsonl` (10 posts) and `data/sample/hatexplain_sample.jsonl` (6 posts) are
**made up** by us, in the processed format, so other members can write code before they have the
data. They are not real data. Do not report results on them.
