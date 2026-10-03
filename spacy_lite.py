"""
A tiny pure-Python stand-in for the parts of spaCy that Kokoro's pronunciation step (misaki) uses:
splitting text into tokens and giving each a Penn Treebank part-of-speech tag.

Why: on this PC, Windows Smart App Control blocks spaCy's compiled files ("An Application Control
policy has blocked this file"), so real spaCy can't load. voice.py installs this module as `spacy`
only when the real one fails to import.

The tagger is rule-based, so it's less accurate than spaCy: a few words that are spelled the same
but pronounced differently depending on use (e.g. "read", "record", "live") may get the wrong one.
"""

import re
import sys
import types

TOKEN = re.compile(
    r"\d+(?:[.,:]\d+)*"                                   # numbers: 12,767  3.5  10:30
    r"|[A-Za-z]+(?:\.[A-Za-z])+\.?"                       # abbreviations: e.g.  U.S.
    r"|[A-Za-z]+(?:['’][A-Za-z]+)*"                       # words, with contractions: don't  it's
    r"|\S"                                                # any other single character
)
PUNCT_TAG = {
    ".": ".", "!": ".", "?": ".", ",": ",", ";": ":", ":": ":", "-": ":", "–": ":", "—": ":", "…": ":",
    "(": "-LRB-", ")": "-RRB-", "[": "-LRB-", "]": "-RRB-", "{": "-LRB-", "}": "-RRB-",
    "“": "``", "‘": "``", "”": "''", "’": "''", "$": "$", "€": "$", "£": "$", "₹": "$", "#": "#",
}
PRONOUNS = {"i", "you", "we", "they", "he", "she", "it", "who"}
MODALS = {"will", "would", "can", "could", "shall", "should", "may", "might", "must", "to", "please", "let's"}
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her", "its", "no", "each", "every"}
PAST_AUX = {"have", "has", "had", "was", "were", "been", "is", "are", "be"}


class Token:
    def __init__(self, text, tag, whitespace):
        self.text, self.tag_, self.whitespace_ = text, tag, whitespace


def _tag(word, prev):
    low, p = word.lower(), (prev or "").lower()
    if word in PUNCT_TAG:
        return PUNCT_TAG[word]
    if word == '"':
        return "``" if prev is None or prev in PUNCT_TAG else "''"
    if word[0].isdigit():
        return "CD"
    if not word[0].isalpha():
        return "NFP"
    if low in PRONOUNS:
        return "PRP"
    if low in DETERMINERS:
        return "DT"
    if low.endswith("ly"):
        return "RB"
    if p in MODALS:
        return "VB"
    if p in PAST_AUX and low.endswith("ed"):
        return "VBN"
    if p in PRONOUNS:
        return "VBD" if low.endswith("ed") else "VBP"
    if low.endswith("ing"):
        return "VBG"
    if low.endswith("ed"):
        return "VBD"
    return "NNP" if word[0].isupper() and prev not in (None, ".", "``") else "NN"


class Language:
    def __call__(self, text):
        tokens, prev = [], None
        for m in TOKEN.finditer(text):
            end = m.end()
            space = text[end:end + 1] if end < len(text) and text[end].isspace() else ""
            tag = _tag(m.group(), prev)
            tokens.append(Token(m.group(), tag, space))
            prev = m.group() if tag not in (".",) else None
        return tokens


def _alignment_unsupported(*args, **kwargs):
    raise NotImplementedError("spacy_lite does not support misaki's [text](/phonemes/) markup")


def install():
    """Register this module as `spacy` (removing any half-imported real spaCy first)."""
    for name in [n for n in sys.modules if n == "spacy" or n.startswith(("spacy.", "thinc"))]:
        del sys.modules[name]
    spacy = types.ModuleType("spacy")
    spacy.load = lambda name, **kwargs: Language()
    spacy.util = types.SimpleNamespace(is_package=lambda name: True)
    spacy.cli = types.SimpleNamespace(download=lambda name: None)
    spacy.training = types.SimpleNamespace(Alignment=types.SimpleNamespace(from_strings=_alignment_unsupported))
    sys.modules["spacy"] = spacy
