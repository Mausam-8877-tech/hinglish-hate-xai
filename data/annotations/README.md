# Hinglish rationale annotations

Our own word-level rationales for ~250 HOF posts of the HASOC test set (issue #4). HASOC has
labels but no rationales, so these are what plausibility on **code-mixed** text is measured against.

| File | Committed? | What it is |
|---|---|---|
| `sheets/member1.csv` … `member4.csv` | **No** (git-ignored, contains HASOC text) | one sheet per member to fill in |
| `hasoc_rationales.jsonl` | **Yes** | collected result: post id, token count, 0/1 mask, annotator. No text. |

## Workflow

```bash
python scripts/sample_for_annotation.py             # 1. make the 4 sheets (seed 42, same for everyone)
# 2. each member fills the "rationale" column of their own sheet and saves it as CSV (UTF-8)
python scripts/sample_for_annotation.py --collect   # 3. -> hasoc_rationales.jsonl, then commit it
```

Share the filled sheets with the person running step 3 privately (Drive), not on GitHub.

## Guidelines (read before annotating)

Every post in your sheet is labelled **HOF** (hateful or offensive). For each post, write in the
`rationale` column the **numbers** of the words that make it hateful or offensive. The numbers are in
the `numbered_words` column (`0=tum | 1=log | 2=gaddar | ...`).

- Format: numbers separated by spaces, e.g. `2 5`. A range of neighbouring words: `2-4` (= 2 3 4).
- Mark the **smallest set of words** that would justify the HOF label on its own: insults, slurs,
  threats, abusive emojis (`emoji_...`), dehumanising words.
- For hate against a group, also mark the **target** word if it is part of the insult
  (e.g. "ye *log* *gaddar* hai"). Do not mark neutral words such as "hai", "ye", "<user>".
- Mark the word in whichever script it appears (Devanagari or Roman).
- If you think the post is **not** actually offensive, leave `rationale` empty and write `not HOF` in
  `comment`. Empty rows are skipped when collecting.
- Do not change the `id`, `text` or `numbered_words` columns.

These follow the HateXplain instructions (Mathew et al., 2021), so our Hinglish rationales and the
English HateXplain rationales are comparable.
