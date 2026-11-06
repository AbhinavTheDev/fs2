"""Symbol table for the Telugu (character-level) FastSpeech2.

Telugu script is almost phonemic, so we feed Unicode code points directly to the encoder:
every character of the Telugu Unicode block (U+0C00 - U+0C7F: consonants, vowels, vowel signs,
virama, anusvara ...) plus a space and a few punctuation marks. A static table (instead of one
built from the dataset) guarantees that training and inference always agree.
"""

_pad = "_"
_punctuation = "!,.? "
_telugu = [chr(c) for c in range(0x0C00, 0x0C80)]

symbols = [_pad] + list(_punctuation) + _telugu
