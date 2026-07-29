#!/usr/bin/env python3
"""Sanity-check espeak-ng's phoneme output for every language.

espeak does not emit real IPA for every voice even with `--ipa`. The Kyrgyz
voice mixes in X-SAMPA-style ASCII (`oe` for ø, `N` for ŋ, `tS` for tʃ, plus a
`[` marker before front vowels). Character-level tokenisation would split `oe`
into `o` + `e` and build the S3 diphone inventory on phonemes that do not exist.

Run this after adding a language: if it reports non-IPA symbols, that voice
needs a mapping in `msbench.corpus.phonemes` before the language can be used.

    python -m msbench.corpus.check_phonemes --n 300
"""

import argparse
import unicodedata
from collections import Counter

from phonemizer.backend import EspeakBackend  # noqa: E402

from msbench.corpus.inventory import load, select_target
from msbench.languages import CODES, ESPEAK_VOICE, LANGUAGES

# Legal beyond IPA letters: stress, length, syllable boundaries, nasalisation.
IPA_MARKS = set("ˈˌːˑ̃ ̯͡‿|.")


def is_ipa_char(ch):
    if ch in IPA_MARKS or ch.isspace():
        return True
    name = unicodedata.name(ch, "")
    return ("LATIN" in name and not ch.isascii()) or (ch.isascii() and ch.islower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./cv")
    ap.add_argument("--n", type=int, default=300, help="texts per language")
    ap.add_argument("--langs", nargs="+", default=CODES)
    args = ap.parse_args()

    for lang in args.langs:
        df = load(args.data_dir, LANGUAGES[lang].locale, with_duration=False)
        sub = select_target(df, lang)
        texts = (sub.sentence.dropna().drop_duplicates()
                 .sample(min(args.n, sub.sentence.nunique()), random_state=0).tolist())

        phones = EspeakBackend(ESPEAK_VOICE[lang]).phonemize(texts)
        chars = Counter("".join(phones))
        # Suspicious: ASCII uppercase (X-SAMPA) and bracketing characters.
        suspicious = {c: n for c, n in chars.items()
                      if c.isascii() and (c.isupper() or c in "[]{}()<>")}

        print("=" * 74)
        print(f"{lang}  (espeak {ESPEAK_VOICE[lang]})  {len(texts)} texts")
        print(f"  distinct symbols: {len(chars)}")
        if suspicious:
            top = ", ".join(f"{c!r}x{n}" for c, n in
                            sorted(suspicious.items(), key=lambda kv: -kv[1])[:10])
            print(f"  NON-IPA symbols: {top}")
            for t, p in zip(texts, phones, strict=True):
                if any(c in p for c in suspicious):
                    print(f"    example: {t[:44]!r}\n             -> {p[:60]!r}")
                    break
        else:
            print("  no non-IPA symbols")


if __name__ == "__main__":
    main()
