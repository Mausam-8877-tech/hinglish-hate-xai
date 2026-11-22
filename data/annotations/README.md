# Hinglish rationale annotations

Our own word-level rationales for 250 HOF posts of the HASOC 2019 Hindi test set (issue #4). HASOC has
labels but no rationales, so these are what plausibility on **code-mixed** text is measured against.

**Method: LLM pre-annotation + human review.** An LLM (Claude Opus 5.5) suggested the rationale words
for every post (`llm_suggestions.jsonl`). Each team member then reviews their own sheet: checks every
row, corrects the words where needed and marks the row `reviewed = yes`. Only reviewed rows become
annotations. The original suggestion is stored next to the final mask, so we can report how often
reviewers changed it. Report it as: *"rationales pre-annotated with an LLM and verified by the four
team members"*.

| File | Committed? | What it is |
|---|---|---|
| `sheets/member1.csv` … `member4.csv` | **No** (git-ignored, contains HASOC text) | one sheet per member to review |
| `llm_suggestions.jsonl` | Yes | the LLM's suggested word numbers per post id (no text) |
| `hasoc_rationales.jsonl` | Yes | collected result: post id, token count, final 0/1 mask, reviewer, LLM mask, changed flag. No text. |

## Status (collected)

All four reviewers returned their sheets (as PDF exports of the Google Sheets); the `reviewed` /
`comment` cells were copied into the CSVs from those exports, and `--collect` was run.

| | Posts |
|---|---|
| sampled HOF test posts | 250 |
| reviewed (`reviewed = yes`) | 249 (one row in member1's sheet has no `yes`: `hasoc_hi_5835`) |
| reviewed with at least one marked word → **in `hasoc_rationales.jsonl`** | **218** |
| reviewed, no word marked (LLM found none, reviewer added none) | 31 |
| LLM suggestions changed by a reviewer | **0** |

Limitation to report: reviewers accepted every suggestion unchanged, so these rationales are
effectively LLM rationales that four humans checked and agreed with; anchoring on the suggestions
cannot be ruled out, and there is no independent second annotation to measure agreement.

Practical note: Google Sheets turned some suggestions into dates (`0 1 12` was shown as `2000 1 12`).
Because no row was changed, the original numbers from `llm_suggestions.jsonl` were kept. For future
sheets, format the `llm_suggestion` and `rationale` columns as *Plain text* before editing.

## Who reviews which sheet

| Sheet | Posts | Reviewer |
|---|---|---|
| `member1.csv` | 63 | Mausam Kumari |
| `member2.csv` | 63 | Manoj Dewda |
| `member3.csv` | 62 | Kuldeep Saraswat |
| `member4.csv` | 62 | Anushri Mishra |

Sheets are shared privately (Drive / WhatsApp), never on GitHub.

## How to review (about 15 minutes)

1. Open your CSV in Excel or Google Sheets.
2. For each row, read `text`. The words are numbered in `numbered_words` (`0=तुम | 1=लोग | 2=हरामी ...`).
3. `llm_suggestion` shows what the LLM marked (do not edit this column).
   `rationale` holds the same numbers to start with: **correct it** if needed.
   - add a missed word, remove a wrong one, e.g. change `2 7` to `2 5 7`
   - ranges are allowed: `3-5` = words 3, 4 and 5
   - if the post is not offensive to you, empty `rationale` and write `not HOF` in `comment`
   - 32 rows have no suggestion (the LLM found no clear word): fill them yourself
4. Write `yes` in `reviewed` for **every row you checked**, also when you changed nothing.
   Rows without `yes` are not used.
5. Save as **CSV UTF-8** (Excel: *File → Save As → CSV UTF-8*) with the same file name and send it back.

## Guidelines

Every post is labelled **HOF** (hateful or offensive). Mark the words that make it hateful or offensive:

- the **smallest set of words** that would justify the HOF label on its own: insults, slurs, threats,
  abusive emojis (`emoji_...`), dehumanising words (e.g. calling people animals)
- for hate against a group, also mark the **target** word when it is part of the insult
  (e.g. "ये *मुसलमान* *गद्दार* है"); do not mark neutral words such as "है", "ये", "<user>"
- mark the word in whichever script it appears (Devanagari or Roman)

These follow the HateXplain instructions (Mathew et al., 2021), so our Hinglish rationales and the
English HateXplain rationales are comparable.

## Commands

```bash
python scripts/sample_for_annotation.py             # make the 4 sheets (seed 42, same for everyone)
python scripts/sample_for_annotation.py --prefill   # fill them with llm_suggestions.jsonl
python scripts/sample_for_annotation.py --collect   # reviewed rows -> hasoc_rationales.jsonl, then commit it
```
