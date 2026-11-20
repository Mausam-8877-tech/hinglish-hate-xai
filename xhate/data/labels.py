"""Label scheme shared by all datasets.

We use ONE binary scheme for every dataset:

    0 = NOT  (not hateful / not offensive)
    1 = HOF  (hate OR offensive)

Q&A (questions a reviewer might ask)
------------------------------------
Q: Why binary and not 3 classes (hate / offensive / normal)?
A: HASOC sub-task A is already binary (HOF vs NOT). HateXplain has 3 classes, but
   "hatespeech" and "offensive" both fold cleanly into HOF. A shared binary scheme
   lets us train on English (HateXplain) and test on Hinglish (HASOC), which is
   exactly what our research question needs.

Q: Where is the original fine-grained label?
A: We do not store it (team decision in issue #1). It can be recovered from the
   raw files if a later experiment needs it.
"""

# Integer id for the "not hateful / not offensive" class.
NOT = 0
# Integer id for the "hate or offensive" class.
HOF = 1

# Human-readable name for each integer id (index = id).
LABEL_NAMES = ["NOT", "HOF"]

# Map every raw label string we might see (lower-cased) to our binary id.
RAW_TO_BINARY = {
    # --- HASOC task A labels ---
    "not": NOT,          # HASOC "NOT" = not offensive
    "hof": HOF,          # HASOC "HOF" = hate or offensive
    # --- HASOC task B labels (in case a file only has task B) ---
    "none": NOT,         # task B "NONE" = clean post
    "hate": HOF,         # task B "HATE" = hate speech
    "offn": HOF,         # task B "OFFN" = offensive
    "prfn": HOF,         # task B "PRFN" = profane
    # --- HateXplain labels ---
    "normal": NOT,       # HateXplain "normal" = not hateful
    "hatespeech": HOF,   # HateXplain "hatespeech" = hate
    "offensive": HOF,    # HateXplain "offensive" = offensive
}


def to_binary(raw_label):
    """Convert one raw label string (any case) to 0 (NOT) or 1 (HOF).

    Q: What happens with an unknown label?
    A: We raise a ValueError instead of guessing, so a broken file fails loudly
       instead of silently corrupting the training data.
    """
    # Normalise: turn into a string, strip spaces, and lower-case it.
    key = str(raw_label).strip().lower()
    # Refuse labels we do not know about.
    if key not in RAW_TO_BINARY:
        # The message tells the user which label was the problem.
        raise ValueError(f"Unknown label {raw_label!r}; expected one of {sorted(RAW_TO_BINARY)}")
    # Look up and return the binary id.
    return RAW_TO_BINARY[key]
