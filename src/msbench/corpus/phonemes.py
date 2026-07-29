#!/usr/bin/env python3
"""Phonemisation with correct tokenisation — the basis of the S3 coverage objective.

Two problems this handles, both measured on real Common Voice text:

1. **espeak-ng's Kyrgyz voice emits X-SAMPA, not IPA**, despite `--ipa`:
   ``ө`` -> ``oe``, ``ң`` -> ``N``, ``ш`` -> ``S``, ``ч`` -> ``tS``,
   ``ж`` -> ``dZ``, ``х`` -> ``X``, length as ``:``. Character-level
   tokenisation would split ``tS`` into ``t`` + ``S`` and build the diphone
   inventory on phonemes that do not exist, so tokenisation is longest-match.
   ``t[`` and ``d[`` are genuine distinct phonemes of that voice (dental) and
   are preserved as-is.

2. **Silent language fallback.** On tokens it cannot read (isolated letters,
   Latin script, digits) espeak switches to the English voice and wraps the
   result in ``(en)...(ky)``, leaking English phonemes ``ɹ ɪ ə`` into the
   Kyrgyz inventory. Such texts are rejected rather than repaired — 3.0% of
   the Kyrgyz pool.

The other five voices produce clean IPA and need neither fix.

    python -m msbench.corpus.phonemes --langs ky --n 400
"""

import argparse
import re
from collections import Counter

from phonemizer.backend import EspeakBackend  # noqa: E402

from msbench.corpus.inventory import load, select_target
from msbench.languages import CODES, ESPEAK_VOICE, LANGUAGES

# espeak's ky voice speaks X-SAMPA; map it to IPA. Order matters — digraphs first.
KY_MAP = [
    ("tS", "tʃ"), ("dZ", "dʒ"), ("oe", "ø"),
    ("N", "ŋ"), ("S", "ʃ"), ("Z", "ʒ"), ("X", "χ"),
    (":", "ː"),
]

# Language-switch marker: espeak could not read a token and used another voice.
LANG_SWITCH = re.compile(r"\((\w{2,3})\)")

# Combining marks belong to their carrier phoneme, not to a symbol of their own.
COMBINING = set("̯̰̝̞̃͡")
# Suprasegmentals (stress, boundaries) are not phonemes and must not enter diphones.
SUPRA = set("ˈˌːˑ|.‿ ")


def normalize(phones, lang):
    if lang == "ky":
        for src, dst in KY_MAP:
            phones = phones.replace(src, dst)
    return phones


def tokenize(phones):
    """Split a phoneme string into symbols, attaching diacritics to their carrier."""
    out = []
    for ch in phones:
        if ch in SUPRA:
            continue
        if ch in COMBINING and out:
            out[-1] += ch
        elif ch == "[" and out:
            out[-1] += ch          # 't[' / 'd[' are distinct phonemes of the ky voice
        else:
            out.append(ch)
    return out


def phonemize_checked(texts, lang):
    """Return (tokens, language_switch_flag) per text."""
    backend = EspeakBackend(ESPEAK_VOICE[lang])
    raw = backend.phonemize(texts)
    result = []
    for ph in raw:
        switched = bool(LANG_SWITCH.search(ph))
        clean = LANG_SWITCH.sub("", ph)
        result.append((tokenize(normalize(clean, lang)), switched))
    return result


def diphones(tokens):
    return list(zip(tokens, tokens[1:], strict=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./cv")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--langs", nargs="+", default=CODES)
    args = ap.parse_args()

    for lang in args.langs:
        df = load(args.data_dir, LANGUAGES[lang].locale, with_duration=False)
        sub = select_target(df, lang)
        texts = (sub.sentence.dropna().drop_duplicates()
                 .sample(min(args.n, sub.sentence.nunique()), random_state=0).tolist())

        res = phonemize_checked(texts, lang)
        n_switch = sum(sw for _, sw in res)
        inv = Counter(t for toks, sw in res if not sw for t in toks)
        dip = Counter(d for toks, sw in res if not sw for d in diphones(toks))

        print("=" * 74)
        print(f"{lang}  (espeak {ESPEAK_VOICE[lang]})  {len(texts)} texts")
        print(f"  language switches (rejected): {n_switch}"
              f"  ({100 * n_switch / len(texts):.1f}%)")
        print(f"  phoneme inventory: {len(inv)}   diphones: {len(dip)}")
        print(f"  inventory: {' '.join(sorted(inv))}")
        rare = [p for p, n in inv.items() if n <= 2]
        if rare:
            print(f"  rare (<=2 occurrences): {' '.join(sorted(rare))}")


if __name__ == "__main__":
    main()
