#!/usr/bin/env python3
"""Text normalisation applied identically to reference and hypothesis before WER.

**Diacritics are kept.** `ñ á ç ã ö` are semantic (`año`/`ano`, `sé`/`se`), so
stripping them would mask real model errors. This is where the normaliser
differs from `whisper.normalizers`, which is too aggressive for non-English.
"""

import re
import unicodedata

from msbench.languages import NUM2WORDS

try:
    from num2words import num2words
except ImportError:      # optional dependency
    num2words = None

# Punctuation to strip. The apostrophe is kept: it is word-internal in English
# (don't, it's) and Dutch ('s-Gravenhage), so removing it would change tokenisation.
PUNCT = r"""!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~«»„“”‘’…–—¿¡"""
PUNCT_RE = re.compile(f"[{re.escape(PUNCT)}]")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def expand_numbers(text, lang):
    code = NUM2WORDS.get(lang)
    if not code or num2words is None:
        return text

    def repl(m):
        raw = m.group(0).replace(",", ".")
        try:
            val = float(raw) if "." in raw else int(raw)
            return num2words(val, lang=code)
        except Exception:  # noqa: BLE001
            return m.group(0)

    return NUM_RE.sub(repl, text)


def normalize(text, lang):
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = expand_numbers(s, lang)
    s = PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    samples = [
        ("es-MX", "¿Cuántos años tiene? ¡Tiene 25 años!"),
        ("pt-BR", "São 3 irmãos, não 2."),
        ("nl-NL", "Het kostte 15 euro — 's morgens."),
        ("en-US", "It's 42 degrees, isn't it?"),
        ("ky", "Ал 25 жашта, туурабы?"),   # no num2words locale for Kyrgyz
    ]
    for lang, t in samples:
        print(f"{lang:<6} {t!r}\n       -> {normalize(t, lang)!r}")
