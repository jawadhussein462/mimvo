"""Normalize text and compare two strings.

Uses only the Python standard library. Detectors call these helpers before
matching patterns or measuring how alike two memories are. Nothing here is
stored back into the memory.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import functools
import html
import re
import unicodedata
import urllib.parse

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Invisible characters used to split trigger words ("ig\u200bnore").
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"))
# Common Cyrillic/Greek look-alikes of Latin letters.
_CONFUSABLES = str.maketrans(
    "\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0456\u0458\u0455\u0501\u0410\u0415\u041e"
    "\u0420\u0421\u0425\u0406\u0408\u0405\u03bf\u03b9\u03b1\u03b5\u039f\u0399\u0391\u0395",
    "aeopcyxijsdAEOPCXIJSoiaeOIAE",
)
# Single letters joined by separators: "i-g-n-o-r-e", "i.g.n.o.r.e", "i g n o r e".
_SPACED = re.compile(r"\b[A-Za-z](?:[-._*]|\s)(?:[A-Za-z](?:[-._*]|\s)){2,}[A-Za-z]\b")
_SEPARATORS = re.compile(r"[-._*\s]")


@functools.lru_cache(maxsize=512)
def deobfuscate(text: str) -> str:
    """Undo common tricks used to hide a phrase from a regex.

    The result is only for matching. It is not written back to the store.

    The steps are: Unicode NFKC normalize, drop invisible characters, map
    Cyrillic and Greek look-alikes to Latin, strip accents, and join
    letters spelled out one by one (`"i-g-n-o-r-e"` becomes `"ignore"`).

    Args:
        text: The memory content, unchanged.

    Returns:
        A new string that is easier for phrase patterns to match.
    """
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_CONFUSABLES)
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return _SPACED.sub(lambda m: _SEPARATORS.sub("", m.group(0)), text)


# -- encoded payloads -----------------------------------------------------------

_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_RUN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){12,}(?![0-9A-Fa-f])")
_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")
_ENTITY = re.compile(r"&(?:#\d{2,6}|#x[0-9A-Fa-f]{2,6}|[A-Za-z]{2,8});")
_WORDS = re.compile(r"[A-Za-z]{2,}")
_MAX_DECODED_RUNS = 8
_MAX_RUN_CHARS = 20_000


def _looks_like_text(raw: bytes) -> str | None:
    """Return `raw` as text when it reads like words, not binary or a key."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t\r")
    if printable / len(text) < 0.95:
        return None
    words = _WORDS.findall(text)
    letters = sum(len(w) for w in words)
    # At least three words, and mostly letters and spaces: rules out hashes and tokens.
    if len(words) < 3 or letters < 0.6 * len(text.replace(" ", "")):
        return None
    return text


def decoded_views(text: str) -> list[tuple[str, str]]:
    """Decode payloads hidden in `text` so phrase patterns can read them.

    A stored memory can carry an instruction as base64, hex, URL
    percent-encoding, HTML entities, or written backwards. Each candidate is
    decoded only when the result reads as natural-language text, so API keys,
    hashes, and ordinary words are left alone. Work is capped: at most a few
    runs per text are decoded, and very long runs are skipped.

    Args:
        text: The memory content, unchanged.

    Returns:
        `(encoding, decoded text)` pairs, such as `("base64", "ignore previous
        instructions")`. `encoding` is one of `base64`, `hex`, `url`, `html`,
        or `reversed`. Empty when nothing decoded to text.
    """
    views: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, decoded: str | None) -> None:
        if decoded and decoded not in seen and decoded != text:
            seen.add(decoded)
            views.append((kind, decoded))

    for count, match in enumerate(_BASE64_RUN.finditer(text)):
        if count >= _MAX_DECODED_RUNS:
            break
        run = match.group(0)
        if len(run) > _MAX_RUN_CHARS or run.isalpha() or run.isdigit():
            continue
        body = run.rstrip("=")
        padded = body + "=" * (-len(body) % 4)
        decoder = base64.urlsafe_b64decode if ("-" in body or "_" in body) else base64.b64decode
        try:
            add("base64", _looks_like_text(decoder(padded)))
        except (binascii.Error, ValueError):
            continue
    for count, match in enumerate(_HEX_RUN.finditer(text)):
        if count >= _MAX_DECODED_RUNS:
            break
        if len(match.group(0)) <= _MAX_RUN_CHARS:
            add("hex", _looks_like_text(bytes.fromhex(match.group(0))))
    if len(_PERCENT.findall(text)) >= 3:
        add("url", urllib.parse.unquote(text))
    if _ENTITY.search(text):
        add("html", html.unescape(text))
    # Text written backwards reads as words only when reversed: compare how
    # many common English words each direction contains.
    reversed_text = text[::-1]
    forward, backward = _common_words(text), _common_words(reversed_text)
    if backward >= 2 and backward > forward:
        add("reversed", reversed_text)
    return views


_COMMON = frozenset(
    [
        "the",
        "and",
        "you",
        "your",
        "to",
        "of",
        "all",
        "any",
        "previous",
        "prior",
        "instructions",
        "ignore",
        "is",
        "are",
        "this",
        "that",
        "with",
        "for",
        "from",
        "now",
        "not",
    ]
)


def _common_words(text: str) -> int:
    return sum(1 for word in _WORDS.findall(text.lower()) if word in _COMMON)


# -- typo-tolerant keywords -----------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z]{6,}")


def _within_one_edit(a: str, b: str) -> bool:
    """Whether `a` becomes `b` with one insert, delete, substitution, or swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and a[diffs[0]] == b[diffs[1]]
            and a[diffs[1]] == b[diffs[0]]
        )
    if la > lb:
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        elif skipped:
            return False
        else:
            skipped = True
        j += 1
    return True


def correct_keywords(text: str, keywords: frozenset[str]) -> str:
    """Replace near-misspellings of `keywords` with the keyword.

    Attackers misspell trigger words ("ignore previous iunstructions") to
    slip past exact patterns. A word of six or more letters that is one edit
    away from a keyword (insert, delete, substitute, or swap two neighbours)
    is rewritten to that keyword, keeping its case style. Shorter words are
    left alone, so ordinary vocabulary is not rewritten into triggers.

    Args:
        text: Text to correct, usually already deobfuscated.
        keywords: Lowercase words to snap to, such as `"instructions"`.

    Returns:
        A copy of `text` with near-misses replaced.
    """

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in keywords:
            return word
        for keyword in keywords:
            if abs(len(keyword) - len(lower)) <= 1 and _within_one_edit(lower, keyword):
                return keyword.capitalize() if word[:1].isupper() else keyword
        return word

    return _TOKEN.sub(fix, text)


# -- sentences ------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+|(?<=:)\s+(?=[A-Z])")


def sentence_around(text: str, start: int, end: int) -> str:
    """Return the sentence that contains `text[start:end]`.

    Sentences end at `.`, `!`, `?`, `;`, a line break, or a colon followed
    by a capitalised word. Used to judge a pattern hit by its own sentence,
    so a warning in one sentence does not excuse a claim in the next.

    Args:
        text: The full text.
        start: Start offset of the hit.
        end: End offset of the hit.

    Returns:
        The enclosing sentence, stripped.
    """
    left = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        left = match.end()
    right_match = _SENTENCE_END.search(text, end)
    right = right_match.start() if right_match else len(text)
    return text[left:right].strip()


# Common filler words ignored when comparing token sets.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "with",
        "that",
        "this",
        "it",
        "as",
        "by",
        "from",
        "now",
        "then",
    }
)


def normalize(text: str) -> str:
    """Make two strings easier to compare as words.

    Args:
        text: Any string.

    Returns:
        `text` in lowercase, with punctuation replaced by spaces and
        repeated whitespace collapsed to a single space.
    """
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split `text` into words after `normalize`.

    Args:
        text: Any string.

    Returns:
        Lowercase words, in order. Empty when `text` has no words.
    """
    return normalize(text).split()


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    """Return the unique words in `text`.

    Args:
        text: Any string. It is normalized before the split.
        drop_stopwords: When `True` (the default), drop filler words such
            as "the" and "and" so they do not count as similarity.

    Returns:
        A set of lowercase words. Order is not preserved.
    """
    tokens = set(tokenize(text))
    if drop_stopwords:
        tokens -= _STOPWORDS
    return tokens


def jaccard(a: str, b: str) -> float:
    """Score how much the word sets of `a` and `b` overlap.

    Args:
        a: First text.
        b: Second text.

    Returns:
        A float from 0 to 1. `1` means the same words (after dropping
        filler words). `0` means no shared words. Two empty texts return `1`.
    """
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def ratio(a: str, b: str) -> float:
    """Score how similar `a` and `b` are as character sequences.

    Args:
        a: First text.
        b: Second text.

    Returns:
        A float from 0 to 1 from `difflib.SequenceMatcher` on the normalized
        strings. `1` means the normalized texts are identical.
    """
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def similarity(a: str, b: str) -> float:
    """Score how alike two texts are, using whichever measure is higher.

    Args:
        a: First text.
        b: Second text.

    Returns:
        The larger of `jaccard(a, b)` and `ratio(a, b)`, from 0 to 1.
    """
    return max(jaccard(a, b), ratio(a, b))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(
                previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def rouge_l(a: str, b: str) -> float:
    """Score how much of the word order is shared, via the longest common subsequence.

    Args:
        a: First text.
        b: Second text.

    Returns:
        An F1 score from 0 to 1. `0` means no shared word sequence.
        Higher means the two texts reuse a longer run of the same words,
        even with gaps.
    """
    ta, tb = tokenize(a), tokenize(b)
    lcs = _lcs_length(ta, tb)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(tb), lcs / len(ta)
    return 2 * precision * recall / (precision + recall)


def cosine(a: list[float], b: list[float]) -> float:
    """Score how aligned two embedding vectors are.

    Args:
        a: First vector.
        b: Second vector. Must be the same length as `a`.

    Returns:
        Cosine similarity. `1` means the same direction, `0` means
        perpendicular or a zero-length vector. Negative values mean
        opposite directions.

    Raises:
        ValueError: The two lists have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "correct_keywords",
    "cosine",
    "decoded_views",
    "deobfuscate",
    "jaccard",
    "normalize",
    "ratio",
    "rouge_l",
    "sentence_around",
    "similarity",
    "token_set",
    "tokenize",
]
