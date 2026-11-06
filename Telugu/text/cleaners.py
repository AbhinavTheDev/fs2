"""Text cleaners for Telugu."""
import re
import unicodedata

_whitespace_re = re.compile(r"\s+")

# Characters that are removed completely (zero-width joiners, BOM ...)
_REMOVE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Punctuation normalisation: sentence-final marks -> ".", clause marks -> ",", the rest -> space
_MAP = {}
for ch in "\u0964\u0965":          # danda, double danda
    _MAP[ord(ch)] = "."
for ch in ";:\u2026":              # ; : ellipsis
    _MAP[ord(ch)] = ","
for ch in "\"\u201c\u201d\u2018\u2019'`()[]{}<>/\\|_*#@~^=+%&$-\u2013\u2014":
    _MAP[ord(ch)] = " "


def telugu_cleaners(text):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE)
    text = text.translate(_MAP)
    text = _whitespace_re.sub(" ", text).strip()
    return text
