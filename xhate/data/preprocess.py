"""Text cleaning and tokenisation for Hinglish (and English) posts.

There are TWO entry points, because the two datasets need different care:

* ``clean_text(text) -> list[str]``
    For raw strings (HASOC). We are free to split, merge or drop tokens here,
    because HASOC has no word-level rationales that must stay aligned.

* ``normalize_tokens(tokens) -> list[str]``
    For data that is ALREADY tokenised and has per-token rationales (HateXplain).
    It maps every token to exactly ONE output token, so ``len(out) == len(tokens)``
    always holds and the rationales keep lining up with the words.

What the cleaning does (team decisions from issue #1):
    URLs          -> "<url>"
    @mentions     -> "<user>"
    #HashTags     -> "hash tags"  (the "#" is dropped and camelCase is split; HASOC only)
    emojis        -> "emoji_<name>", e.g. 😡 -> "emoji_enraged_face"
    case          -> lower-case (Devanagari has no case, so it is unaffected)
    "sooooo"      -> "soo"        (3+ repeated characters squashed to 2)
    nhi / nai     -> "nahi"       (small, editable spelling dictionary below)

Q&A
---
Q: Why keep Devanagari instead of transliterating it to Roman script?
A: MuRIL and XLM-R were pre-trained on both scripts, and transliteration tools add
   their own errors. Keeping the original script is the safer default.

Q: Why turn emojis into words instead of deleting them?
A: Emojis like 😡 or 🤬 carry a lot of the offensive signal. As words, they become
   normal tokens that every model (CNN, LSTM, transformer) can use and that LIME
   can highlight as a rationale.

Q: Why squash to 2 repeated letters and not to 1?
A: Squashing to 1 would break real words ("good" -> "god", "accha" -> "acha").
   Two repeats keep those words intact and still merge "nahiiiii" with "nahiii".
   The spelling lookup below also tries the 1-repeat form as a fallback.
"""

# html: to turn "&amp;" back into "&" (tweets are often HTML-escaped).
import html
# re: regular expressions, used for every pattern below.
import re

# emoji: third-party package that knows the name of every emoji.
import emoji

# ---------------------------------------------------------------------------
# Spelling dictionary
# ---------------------------------------------------------------------------
# Maps a common Hinglish spelling variant (lower-case) to one canonical form.
# HOW TO EXTEND: add  "variant": "canonical",  lines. Keep keys lower-case.
# We avoid ambiguous short words (e.g. "he", "or", "main") because they are also
# English or Hindi words with a different meaning.
SPELLING_VARIANTS = {
    # --- "no" ---
    "nhi": "nahi",       # very common SMS spelling
    "nai": "nahi",       # informal
    "nahin": "nahi",     # formal spelling
    "nahee": "nahi",     # phonetic spelling
    "nhin": "nahi",      # vowel dropped
    # --- "is / are" ---
    "h": "hai",          # one-letter SMS spelling of "hai"
    "hy": "hai",         # phonetic
    "hae": "hai",        # phonetic
    # --- "what" ---
    "kia": "kya",        # phonetic
    "kyaa": "kya",       # stretched
    "ky": "kya",         # vowel dropped
    # --- "why" ---
    "kyu": "kyun",       # nasal dropped
    "kyon": "kyun",      # alternative spelling
    "kiu": "kyun",       # phonetic
    "kyoon": "kyun",     # phonetic
    # --- "also" ---
    "bhee": "bhi",       # stretched vowel
    # --- "good / okay" ---
    "acha": "accha",     # single c
    "achha": "accha",    # with h
    "acchha": "accha",   # double c with h
    "thik": "theek",     # short vowel
    "thk": "theek",      # vowels dropped
    # --- "friend" ---
    "yar": "yaar",       # short vowel
    "yr": "yaar",        # vowels dropped
    # --- "know" ---
    "pta": "pata",       # vowel dropped
    # --- "do" verb forms ---
    "kr": "kar",         # vowel dropped
    "kro": "karo",       # vowel dropped
    "krna": "karna",     # vowel dropped
    "krte": "karte",     # vowel dropped
    "krta": "karta",     # vowel dropped
    # --- "-ing" helper verb forms ---
    "rha": "raha",       # vowel dropped
    "rhi": "rahi",       # vowel dropped
    "rhe": "rahe",       # vowel dropped
    # --- pronouns ---
    "tm": "tum",         # vowel dropped
    "ap": "aap",         # short vowel
    "mjhe": "mujhe",     # vowel dropped
    "muje": "mujhe",     # h dropped
    "tjhe": "tujhe",     # vowel dropped
    "sb": "sab",         # vowel dropped
    # --- "very" ---
    "bht": "bahut",      # vowels dropped
    "bhut": "bahut",     # vowel dropped
    "bohot": "bahut",    # phonetic
    "bahot": "bahut",    # phonetic
    "bhot": "bahut",     # phonetic
    # --- time words ---
    "abi": "abhi",       # h dropped
    "kbhi": "kabhi",     # vowel dropped
    "jb": "jab",         # vowel dropped
    "tb": "tab",         # vowel dropped
    # --- "something" ---
    "kuchh": "kuch",     # extra h
    "kch": "kuch",       # vowel dropped
    # --- English SMS spellings that also appear in Hinglish posts ---
    "u": "you",          # SMS spelling
    "ur": "your",        # SMS spelling
    "pls": "please",     # abbreviation
    "plz": "please",     # abbreviation
    "thx": "thanks",     # abbreviation
}

# ---------------------------------------------------------------------------
# Regular expressions (compiled once, reused for every post)
# ---------------------------------------------------------------------------
# A URL: http(s)://... or www.... up to the next whitespace.
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
# An @mention: "@" followed by everything up to the next space, "#" or "@".
# NOTE: we do NOT use \w here, because Python's \w does not match Devanagari vowel
# signs (matras), so "@राहुल" would be cut after "@र". (Bug found on real HASOC 2019.)
MENTION_RE = re.compile(r"@[^\s#@]+")
# A hashtag: "#" followed by everything up to the next space, "#" or "@"; group 1 is
# the word part. Same reason as above: "#शांतीदूत" must stay one piece.
# Trailing punctuation ("#INDvBAN,") is removed later by the tokenizer.
HASHTAG_RE = re.compile(r"#([^\s#@]+)")
# Inside a hashtag: a lower-case letter followed by an upper-case one ("modiHatao").
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
# Inside a hashtag: a letter next to a digit ("india2020") or the reverse.
DIGIT_SPLIT_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
# Any character repeated 3 or more times in a row; group 1 is the character.
REPEAT_RE = re.compile(r"(.)\1{2,}")
# Any character repeated 2 or more times (used for the 1-repeat spelling fallback).
DOUBLE_RE = re.compile(r"(.)\1+")
# Placeholder tokens such as <url>, <user>, <number> (HateXplain uses these too).
PLACEHOLDER_RE = re.compile(r"^<[a-z_]+>$")
# Tokenizer: a placeholder, OR a run of "!"/"?" as its own token, OR a run of characters that are
# neither whitespace nor punctuation. Devanagari letters and vowel signs (matras) are
# not in the excluded set, so Hindi words stay in one piece.
TOKEN_RE = re.compile(r"<[a-z_]+>|[!?]+|[^\s!?.,;:\"“”‘’()\[\]{}…।|/\\*~^=+<>]+")
# Characters that are not allowed inside an emoji name token (we keep a-z, 0-9 and "_").
EMOJI_NAME_BAD_CHARS_RE = re.compile(r"[^a-z0-9_]+")


def _emoji_to_name(chars, data_dict):
    """Callback for ``emoji.replace_emoji``: return the emoji's name as a word.

    ``data_dict["en"]`` looks like ":pouting_face:". We turn it into
    "emoji_enraged_face" and pad it with spaces so it becomes its own token.
    """
    # Get the English name, e.g. ":pouting_face:" (fallback: "unknown").
    name = data_dict.get("en", ":unknown:")
    # Remove the surrounding colons and lower-case it.
    name = name.strip(":").lower()
    # Replace any character that is not a-z, 0-9 or "_" (e.g. "-", ":") with "_".
    name = EMOJI_NAME_BAD_CHARS_RE.sub("_", name).strip("_")
    # Prefix with "emoji_" and surround with spaces so it splits off as one token.
    return f" emoji_{name} "


def _normalize_word(word):
    """Lower-case one word, squash repeated letters and fix its spelling.

    Used by both ``clean_text`` and ``normalize_tokens``. Always returns a string.
    """
    # Placeholders like "<url>" are already canonical: return them unchanged.
    if PLACEHOLDER_RE.match(word):
        return word  # placeholder: nothing to change
    # Lower-case (has no effect on Devanagari, which has no case).
    word = word.lower()
    # Squash 3+ repeated characters to 2: "sooooo" -> "soo", "!!!!" -> "!!".
    word = REPEAT_RE.sub(r"\1\1", word)
    # If this exact spelling is in the dictionary, return the canonical form.
    if word in SPELLING_VARIANTS:
        return SPELLING_VARIANTS[word]  # e.g. 'nhi' -> 'nahi'
    # Fallback: squash ALL repeats to 1 ("nahii" -> "nahi") and look up again.
    single = DOUBLE_RE.sub(r"\1", word)
    # If the 1-repeat form is a known variant, map it to its canonical form.
    if single in SPELLING_VARIANTS:
        return SPELLING_VARIANTS[single]  # e.g. 'nhii' -> 'nhi' -> 'nahi'
    # If the 1-repeat form is itself a canonical word (e.g. "nahi"), use it.
    if single in _CANONICAL_WORDS:
        return single  # e.g. 'nahii' -> 'nahi'
    # Otherwise keep the (lower-cased, squashed) word as it is.
    return word


# Set of all canonical spellings, used by the 1-repeat fallback above.
_CANONICAL_WORDS = set(SPELLING_VARIANTS.values())


def _split_hashtag(match):
    """Callback for ``HASHTAG_RE.sub``: "#ModiHatao2024" -> " Modi Hatao 2024 "."""
    # The part after "#".
    body = match.group(1)
    # Underscores separate words in some hashtags ("#stop_hate").
    body = body.replace("_", " ")
    # Insert a space at every lower->upper case boundary ("ModiHatao" -> "Modi Hatao").
    body = CAMEL_RE.sub(" ", body)
    # Insert a space between letters and digits ("india2020" -> "india 2020").
    body = DIGIT_SPLIT_RE.sub(" ", body)
    # Pad with spaces so the words do not glue onto neighbouring text.
    return f" {body} "


def clean_text(text):
    """Clean ONE raw post string and return its list of tokens.

    Example:
        >>> clean_text("RT @abc: Tum log nhi sudhroge!!! 😡 #ModiHatao http://t.co/x")
        ['<user>', 'tum', 'log', 'nahi', 'sudhroge', '!!', 'emoji_enraged_face', 'modi', 'hatao', '<url>']

    Q: Why return tokens instead of a string?
    A: Every later step (vocab building, rationales, LIME) works on tokens. The
       cleaned string is just ``" ".join(tokens)``.

    Q: Why is "RT" removed?
    A: "RT" (retweet) at the start of a tweet is a Twitter artefact, not content.
    """
    # Missing values (NaN / None) become an empty post.
    if text is None or (isinstance(text, float) and text != text):
        # Empty post -> empty token list.
        return []
    # Make sure we have a string (some files store numbers in the text column).
    text = str(text)
    # Undo HTML escaping: "&amp;" -> "&", "&lt;" -> "<".
    text = html.unescape(text)
    # Replace every URL with the placeholder <url> (padded with spaces).
    text = URL_RE.sub(" <url> ", text)
    # Replace every @mention with the placeholder <user>.
    text = MENTION_RE.sub(" <user> ", text)
    # Split hashtags into their words (must run BEFORE lower-casing, to see camelCase).
    text = HASHTAG_RE.sub(_split_hashtag, text)
    # Replace every emoji with " emoji_<name> ".
    text = emoji.replace_emoji(text, replace=_emoji_to_name)
    # Lower-case everything (placeholders are already lower-case).
    text = text.lower()
    # Cut the text into tokens with the tokenizer regex.
    raw_tokens = TOKEN_RE.findall(text)
    # Normalise every token (repeats + spelling dictionary).
    tokens = [_normalize_word(tok) for tok in raw_tokens]
    # Drop a leading "rt" (retweet marker), possibly repeated.
    while tokens and tokens[0] == "rt":
        # Remove the first token.
        tokens.pop(0)
    # Drop empty tokens and tokens made only of "#" / "@" (e.g. a lone "#" at the end of a
    # truncated tweet, or censoring like "#@#"); they carry no meaning. Seen in real HASOC 2019.
    tokens = [tok for tok in tokens if tok.strip("#@")]
    # Return the final token list.
    return tokens


def normalize_token(token):
    """Normalise ONE pre-split token and return exactly ONE token (never empty).

    Used for HateXplain, whose tokens are aligned with rationales.

    Q: Why do hashtags here only lose the "#", without camelCase splitting?
    A: Splitting "#ModiHatao" into two tokens would change the number of tokens
       and break the alignment with the rationale mask.
    """
    # Make sure we have a string.
    token = str(token)
    # Placeholders such as "<user>" or "<number>" are kept as they are.
    if PLACEHOLDER_RE.match(token):
        return token  # e.g. '<number>' stays '<number>'
    # A token that is a URL becomes "<url>".
    if URL_RE.fullmatch(token):
        return "<url>"  # the whole token was a URL
    # A token that is an @mention becomes "<user>".
    if MENTION_RE.fullmatch(token):
        return "<user>"  # the whole token was a mention
    # A hashtag token just loses its leading "#" ("#maga" -> "maga").
    if token.startswith("#") and len(token) > 1:
        token = token[1:]  # drop the first character, the '#'
    # Replace emojis with their names (adds spaces around each name).
    token = emoji.replace_emoji(token, replace=_emoji_to_name)
    # Join the pieces with "_" so the result stays ONE token ("lol emoji_x" -> "lol_emoji_x").
    token = "_".join(token.split())
    # Lower-case, squash repeats and fix spelling.
    token = _normalize_word(token)
    # Never return an empty token: that would silently drop a word.
    return token if token else "<unk>"


def normalize_tokens(tokens):
    """Normalise a list of tokens 1-to-1. ``len(output) == len(tokens)`` ALWAYS.

    Q: How do we know the length never changes?
    A: ``normalize_token`` returns exactly one non-empty string per input, and we
       assert the lengths match before returning (cheap insurance).
    """
    # Normalise every token on its own.
    out = [normalize_token(tok) for tok in tokens]
    # Safety check: the rationale alignment depends on this.
    assert len(out) == len(tokens), "normalize_tokens must not change the token count"
    # Return the normalised tokens.
    return out
