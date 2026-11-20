# Data

Raw and processed data are **git-ignored**. Every member downloads the data on their own machine.

```
data/
├── raw/
│   ├── hasoc/          HASOC files, as downloaded
│   └── hatexplain/     HateXplain files, as downloaded
├── processed/          cleaned data, written by scripts/prepare_data.py
└── annotations/        our own Hinglish rationale annotations (committed)
```

## HASOC (Hindi / Hinglish)
Register at <https://hasocfire.github.io>, choose a year, and download the Hindi track.
Put the files in `data/raw/hasoc/`.

## HateXplain (English, with rationales)
Download `dataset.json` and `post_id_divisions.json` from
<https://github.com/hate-alert/HateXplain/tree/master/Data> into `data/raw/hatexplain/`.
