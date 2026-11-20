"""Generate a FAKE HASOC-style dataset so we can work before we have the real one.

Usage (from the repo root):

    python scripts/make_fake_hasoc.py                       # ~500 posts into data/raw/hasoc_fake/
    python scripts/prepare_data.py --datasets hasoc --raw-dir data/raw/hasoc_fake

The second command writes data/processed/hasoc/{train,val,test}.jsonl, so every
other task can already call ``load_split("hasoc", ...)``. When the real HASOC
data arrives, run ``python scripts/prepare_data.py --datasets hasoc`` and the
fake processed files are overwritten.

What the fake files look like (same layouts as the real releases, so the loader
is tested properly):

    data/raw/hasoc_fake/hasoc/
        2019/FAKE_hindi_dataset.tsv            text_id, text, task_1, task_2, task_3   (train)
        2019/FAKE_hasoc2019_hi_test_gold.tsv   same columns                             (test)
        2020/FAKE_hasoc_2020_hi_train.xlsx     tweet_id, text, task1, task2             (train)
        2021/FAKE_hi_Hasoc2021_train.csv       _id, text, task_1, task_2                (train)
        2021/FAKE_hi_Hasoc2021_test.csv        _id, text, task_1, task_2                (test)

Q&A
---
Q: Is this real data? Can we report results on it?
A: NO. Every post is generated from small hand-written templates. It only checks
   that the code runs end to end. Every processed row has "FAKE" in its
   ``source`` field so it can never be mistaken for real data.

Q: Why does it add label noise?
A: Without noise, a model could reach 100% just by spotting the template
   words, and bugs in training code would be hidden. With ~10% flipped labels,
   a working model scores well but not perfectly.

Q: Why the noise in the text (emojis, @mentions, "nhi", "sooo")?
A: So the preprocessing code (preprocess.py) has something real to clean.

Q: Why is it written to data/raw/hasoc_fake/ and not data/raw/hasoc/?
A: So it never gets mixed with the real files. data/raw/ is git-ignored, so the
   fake files are not committed either (only this script is).
"""

# argparse: command-line options.
import argparse
# random: reproducible random choices (fixed seed).
import random
# Path: OS-independent paths.
from pathlib import Path

# pandas: writes tsv / csv / xlsx files.
import pandas as pd

# Repo root (this file is repo/scripts/make_fake_hasoc.py).
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Word lists used to fill the templates (mild words only, no real slurs).
# ---------------------------------------------------------------------------
# Who / what a post talks about.
SUBJECTS = ["ye neta", "tum log", "ye channel", "ye banda", "aap ki party", "ye media",
            "ye actor", "tera dost", "ye team", "ye sarkar"]
# Insults that make a post offensive (HOF).
INSULTS = ["gadha", "bewakoof", "nalayak", "chor", "jhootha", "ghatiya", "pagal", "dhokebaaz"]
# Neutral / positive words for NOT posts.
POSITIVES = ["accha", "mast", "badhiya", "shandaar", "zabardast", "sahi"]
# Neutral topics for NOT posts.
TOPICS = ["match", "movie", "exam", "mausam", "khana", "gaana", "trip", "cricket"]
# Fake hashtags (camelCase so hashtag splitting is exercised).
HASHTAGS = ["#IndiaFirst", "#ModiSarkar", "#CricketFever", "#ExamTime", "#BoycottBollywood",
            "#DeshKeGaddar", "#HappyDiwali", "#JusticeForAll"]
# Emojis, angry and happy.
ANGRY_EMOJIS = ["😡", "🤬", "😤", "👎"]
HAPPY_EMOJIS = ["😂", "😊", "❤️", "👍", "🔥"]  # used for NOT posts

# Templates for offensive posts; {s} = subject, {i} = insult.
HOF_TEMPLATES = [
    "{s} ek number ka {i} hai",
    "{s} sab {i} hai, sharam nahi aati?",
    "{s} jaise {i} logo ko desh se nikalo",
    "abe {i}, {s} ki bakwas band kar",
    "{s} se bada {i} koi nahi",
    "tu {i} hai aur {s} bhi {i} hai",
    "{s} ko jail mein daalo, {i} kahin ke",
]
# Templates for non-offensive posts; {t} = topic, {p} = positive word.
NOT_TEMPLATES = [
    "aaj ka {t} bahut {p} tha yaar",
    "kal {t} ke liye ready ho?",
    "{t} dekh ke maza aa gaya, ekdum {p}",
    "kya koi bata sakta hai {t} kab hai?",
    "mujhe {t} bahut {p} laga",
    "{t} ke baad chai peete hai",
    "sab log {t} ka plan bana lo",
]
# A few Devanagari posts per class (HASOC Hindi has both scripts).
HOF_DEVANAGARI = ["तुम जैसे लोग देश के दुश्मन हो", "ये नेता एक नंबर का चोर है", "बकवास बंद करो पागल"]
NOT_DEVANAGARI = ["मुझे यह फिल्म बहुत पसंद आई", "आज मौसम बहुत अच्छा है", "कल परीक्षा है, सबको शुभकामनाएं"]  # Devanagari NOT examples
# Spelling variants to inject (canonical -> SMS form), so normalisation is tested.
SPELLING_NOISE = {"nahi": "nhi", "hai": "h", "bahut": "bht", "kya": "kia", "yaar": "yr"}


def add_noise(text, rng, angry):
    """Randomly add Twitter-style noise to a sentence."""
    # Sometimes swap a canonical word for its SMS spelling ("nahi" -> "nhi").
    if rng.random() < 0.4:
        # Split into words, replace known words, join back.
        text = " ".join(SPELLING_NOISE.get(w, w) if rng.random() < 0.6 else w for w in text.split(" "))
    # Sometimes stretch the last letter ("hai" -> "haiiii").
    if rng.random() < 0.2:
        text = text + text[-1] * rng.randint(2, 5)  # repeat the last character 2-5 more times
    # Sometimes start with a mention.
    if rng.random() < 0.3:
        text = f"@user{rng.randint(1, 99)} {text}"  # e.g. '@user42 ...'
    # Sometimes start with RT (retweet).
    if rng.random() < 0.1:
        text = f"RT {text}"  # retweet marker
    # Sometimes add an emoji that fits the tone.
    if rng.random() < 0.5:
        text = f"{text} {rng.choice(ANGRY_EMOJIS if angry else HAPPY_EMOJIS)}"  # angry for HOF, happy for NOT
    # Sometimes add a hashtag.
    if rng.random() < 0.4:
        text = f"{text} {rng.choice(HASHTAGS)}"  # e.g. '#IndiaFirst'
    # Sometimes add a URL.
    if rng.random() < 0.15:
        text = f"{text} https://t.co/{rng.randint(10000, 99999)}"  # fake short link
    # Sometimes add exclamation marks.
    if rng.random() < 0.25:
        text = text + "!" * rng.randint(1, 4)  # 1-4 exclamation marks
    # Return the noisy text.
    return text


def make_post(rng, hof):
    """Generate one (text, task_2 label) pair for the given class."""
    # 10% of posts are Devanagari.
    if rng.random() < 0.1:
        # Pick a Devanagari sentence of the right class.
        text = rng.choice(HOF_DEVANAGARI if hof else NOT_DEVANAGARI)
    elif hof:  # Roman-script offensive post
        # Fill an offensive template.
        text = rng.choice(HOF_TEMPLATES).format(s=rng.choice(SUBJECTS), i=rng.choice(INSULTS))
    else:
        # Fill a neutral template.
        text = rng.choice(NOT_TEMPLATES).format(t=rng.choice(TOPICS), p=rng.choice(POSITIVES))
    # Add Twitter-style noise.
    text = add_noise(text, rng, angry=hof)
    # Task-B label: one of HATE/OFFN/PRFN for HOF posts, NONE otherwise.
    task2 = rng.choice(["HATE", "OFFN", "PRFN"]) if hof else "NONE"
    # Return both.
    return text, task2


def make_rows(n, rng, noise):
    """Generate ``n`` rows as dicts with keys text, task_1, task_2."""
    # Collected rows.
    rows = []
    # One row at a time.
    for _ in range(n):
        # About half the posts are HOF (like real HASOC Hindi).
        hof = rng.random() < 0.5
        # Generate the text for that class.
        text, task2 = make_post(rng, hof)
        # Flip the label with probability ``noise`` (keeps the task from being trivial).
        if rng.random() < noise:
            # Flip HOF <-> NOT.
            hof = not hof
            # Keep task_2 consistent with the flipped task_1.
            task2 = rng.choice(["HATE", "OFFN", "PRFN"]) if hof else "NONE"
        # Store the row with HASOC's label strings.
        rows.append({"text": text, "task_1": "HOF" if hof else "NOT", "task_2": task2})
    # Return all rows.
    return rows


def parse_args():
    """Read the command-line options."""
    # Parser with the module docstring as help.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Total number of posts (split over the five files).
    parser.add_argument("--n", type=int, default=500, help="total number of posts")
    # Fraction of labels flipped on purpose.
    parser.add_argument("--label-noise", type=float, default=0.1, help="fraction of flipped labels")
    # Output folder (prepare_data.py reads <out>/hasoc/).
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "hasoc_fake",
                        help="output folder; files go to <out>/hasoc/")
    # Random seed (same seed = same fake data for everyone).
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Return the options.
    return parser.parse_args()


def main():
    """Write the five fake files."""
    # Read the options.
    args = parse_args()
    # Random generator with a fixed seed.
    rng = random.Random(args.seed)
    # Where the files go.
    out = args.out / "hasoc"
    # Share of the total for each file (sums to 1.0): 2019 train/test, 2020 train, 2021 train/test.
    shares = {"2019_train": 0.40, "2019_test": 0.12, "2020_train": 0.24, "2021_train": 0.16, "2021_test": 0.08}
    # Number of rows per file.
    sizes = {k: max(1, round(v * args.n)) for k, v in shares.items()}
    # Generate the rows for every file.
    rows = {k: make_rows(n, rng, args.label_noise) for k, n in sizes.items()}
    # Add 3 exact duplicates to 2019 train, so de-duplication is exercised.
    rows["2019_train"] += rows["2019_train"][:3]

    # --- 2019 layout: text_id, text, task_1, task_2, task_3 (tab-separated) ---
    for key, name in [("2019_train", "FAKE_hindi_dataset.tsv"), ("2019_test", "FAKE_hasoc2019_hi_test_gold.tsv")]:
        # Build the table; task_3 (targeted / untargeted) is "NONE" for NOT posts.
        df = pd.DataFrame([{"text_id": f"fake_hi19_{key}_{i}", "text": r["text"], "task_1": r["task_1"],
                            "task_2": r["task_2"], "task_3": "NONE" if r["task_1"] == "NOT" else rng.choice(["TIN", "UNT"])}
                           for i, r in enumerate(rows[key])])
        # Make the folder.
        (out / "2019").mkdir(parents=True, exist_ok=True)
        # Write as TSV, UTF-8.
        df.to_csv(out / "2019" / name, sep="\t", index=False, encoding="utf-8")

    # --- 2020 layout: tweet_id, text, task1, task2 (Excel) ---
    # Build the table.
    df = pd.DataFrame([{"tweet_id": f"fake_hi20_{i}", "text": r["text"], "task1": r["task_1"], "task2": r["task_2"]}
                       for i, r in enumerate(rows["2020_train"])])
    # Make the folder.
    (out / "2020").mkdir(parents=True, exist_ok=True)
    # Write as .xlsx (needs openpyxl, which is in requirements.txt).
    df.to_excel(out / "2020" / "FAKE_hasoc_2020_hi_train.xlsx", index=False)

    # --- 2021 layout: _id, text, task_1, task_2 (comma-separated) ---
    for key, name in [("2021_train", "FAKE_hi_Hasoc2021_train.csv"), ("2021_test", "FAKE_hi_Hasoc2021_test.csv")]:
        # Build the table.
        df = pd.DataFrame([{"_id": f"fake_hi21_{key}_{i}", "text": r["text"], "task_1": r["task_1"],
                            "task_2": r["task_2"]} for i, r in enumerate(rows[key])])
        # Make the folder.
        (out / "2021").mkdir(parents=True, exist_ok=True)
        # Write as CSV, UTF-8.
        df.to_csv(out / "2021" / name, index=False, encoding="utf-8")

    # Report what was written.
    print(f"Wrote {sum(len(r) for r in rows.values())} FAKE posts to {out}")
    # And the next command to run.
    print(f"Next: python scripts/prepare_data.py --datasets hasoc --raw-dir {args.out}")


# Run main() only when executed as a script.
if __name__ == "__main__":
    main()  # start the script
