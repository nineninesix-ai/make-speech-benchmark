#!/usr/bin/env python3
"""Language registry — the single source of truth for every per-language setting.

Adding a language to the benchmark means adding one entry here. Nothing else in
the pipeline hard-codes a language.

The two non-obvious fields:

``match``
    Common Voice has no ``es-MX`` / ``pt-BR`` / ``nl-NL`` locales. Regional
    variants live in free-text, self-declared ``accents`` / ``variant`` columns,
    and the field is multi-valued. Worse, the labels themselves contain commas
    inside parentheses::

        "España: Norte peninsular (Asturias, Castilla y León, Cantabria, ...)"
        "India and South Asia (India, Pakistan, Sri Lanka)"

    so a naive ``split(",")`` shreds them. Labels are therefore split on commas
    at parenthesis depth zero (``split_labels``) and matched exactly — substring
    matching gives false positives, e.g. the Caribbean label contains
    "Costa del golfo de México" and would be picked up by ``contains("México")``.

``espeak``
    Use ``es-419`` for Mexican Spanish, not ``es``: the latter has /θ/, which
    Latin American Spanish does not. Getting this wrong silently corrupts the
    phonetic coverage objective.

Run ``python -m msbench.languages`` for a self-check of the registry.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

# Word-count bins for target texts. Shared boundaries, per-language *shares*.
LEN_BINS = ["3-5", "6-9", "10-12", "13-16", "17+"]

# Prompt duration strata, measured AFTER VAD trimming. Boundaries are the
# terciles of the observed distribution (4,284 prompts: p05 2.4s, median 4.1s,
# p95 5.5s). The original 3-4 / 4-6 / 6-8 left only 8-29 clips per language in
# the top bin: trimming removes ~1.6s and Common Voice has almost no clips
# longer than 8s.
PROMPT_DUR_BINS = ["<3.7", "3.7-4.5", ">=4.5"]


def split_labels(s: str) -> list[str]:
    """Split a multi-valued CV label field without breaking parenthesised labels."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [x for x in out if x]


@dataclass(frozen=True)
class Language:
    """Everything the pipeline needs to know about one benchmark subset."""

    code: str                    # subset name, e.g. "es-MX"
    locale: str                  # Common Voice locale the data comes from
    column: str                  # CV column holding the variant label
    match: Callable[[list[str]], bool]   # predicate over parsed labels
    espeak: str                  # espeak-ng voice for phonemisation
    alphabet: str                # regex matching one in-alphabet character
    n_main: int                  # target size of the main subset
    cap: int                     # max examples per speaker
    len_target: list[float]      # shares over LEN_BINS, must sum to 1
    target_f: float | None    # min female-voice share; None = best effort
    asr: str                     # v1: "whisper:<lang>" or "gigaam" — frozen
    asr_v2: tuple                # v2 backends to run, primary first
    num2words: str | None     # num2words locale; None = leave digits alone
    notes: str = ""              # anything a user of the subset must know
    tags: tuple = field(default_factory=tuple)   # hard-set language extras


LANGUAGES: dict[str, Language] = {
    "en-US": Language(
        code="en-US", locale="en", column="accents",
        match=lambda L: "United States English" in L,
        espeak="en-us", alphabet=r"[a-z']",
        n_main=1500, cap=2,
        len_target=[0.15, 0.30, 0.30, 0.23, 0.02],
        target_f=0.40, asr="whisper:en", asr_v2=("whisper", "mms"),
        num2words="en",
        notes="99% of the slice lives in CV's train split; held-out is only "
              "~600 pairs. The only subset with a real long tail (max 39 words).",
        tags=("heteronym", "silent_letter"),
    ),
    "es-ES": Language(
        code="es-ES", locale="es", column="accents",
        match=lambda L: any(x.startswith("España: Norte peninsular")
                            or x.startswith("España: Centro-Sur") for x in L),
        espeak="es", alphabet=r"[a-záéíóúüñ¿¡]",
        n_main=1500, cap=3,
        len_target=[0.15, 0.30, 0.30, 0.25, 0.00],
        target_f=0.35, asr="whisper:es", asr_v2=("whisper", "mms"),
        num2words="es",
        notes="Castilian norm: Norte + Centro-Sur peninsular. Sur peninsular "
              "(Andalusian) is excluded — seseo/ceceo puts it closer to es-MX, "
              "and 82% of that slice is a single speaker. /θ/ is present here "
              "and absent from es-MX, which is the point of having both.",
        tags=("inverted_punct", "theta_pair"),
    ),
    "es-MX": Language(
        code="es-MX", locale="es", column="accents",
        match=lambda L: "México" in L,
        espeak="es-419", alphabet=r"[a-záéíóúüñ¿¡]",
        n_main=1500, cap=2,
        len_target=[0.15, 0.30, 0.30, 0.25, 0.00],
        target_f=0.35, asr="whisper:es", asr_v2=("whisper", "mms"),
        num2words="es",
        notes="espeak voice is es-419, not es: Latin American Spanish has no /θ/.",
        tags=("inverted_punct",),
    ),
    "nl-NL": Language(
        code="nl-NL", locale="nl", column="accents",
        match=lambda L: "Nederlands Nederlands" in L,
        espeak="nl", alphabet=r"[a-zàáéèêëïíîóöúü]",
        n_main=1500, cap=4,
        len_target=[0.15, 0.30, 0.30, 0.23, 0.02],
        target_f=0.15, asr="whisper:nl", asr_v2=("whisper", "mms"),
        num2words="nl",
        notes="All voice diversity sits in dev+test (472 speakers); train is "
              "22 speakers. Hence the very high held-out share.",
        tags=("compound_long",),
    ),
    "pt-BR": Language(
        code="pt-BR", locale="pt", column="variant",
        match=lambda L: "Portuguese (Brasil)" in L,
        espeak="pt-br", alphabet=r"[a-záàâãéêíóôõúüç]",
        n_main=1500, cap=7,
        len_target=[0.35, 0.35, 0.20, 0.10, 0.00],
        target_f=None, asr="whisper:pt", asr_v2=("whisper", "mms"),
        num2words="pt_BR",
        notes="Splits are broken upstream: 97% of test sentences also appear in "
              "train and 9,464 clips are physically duplicated between train and "
              "dev, so dedup must be by `path`. Only 19 female speakers exist in "
              "the whole variant. Bins shifted left — median utterance is 6 words. "
              "cap=7 because only 247 speakers survived QC and 247*6 < 1500.",
        tags=("nasal",),
    ),
    "ky": Language(
        code="ky", locale="ky", column="accents",
        match=lambda L: True,          # Kyrgyz has no variant labels at all
        espeak="ky", alphabet=r"[а-яёөүң]",
        n_main=700, cap=4,
        len_target=[0.35, 0.55, 0.10, 0.00, 0.00],
        target_f=None, asr="gigaam", asr_v2=("gigaam", "mms"),
        num2words=None,
        notes="Corpus-limited, not a design choice: 5,016 clips from 211 "
              "speakers is the entire language, and CV22 is the same size. "
              "Max 13 words in the whole language, so the two longest bins are "
              "empty. Gender undeclared for 44% of speakers. The espeak voice "
              "emits X-SAMPA rather than IPA and silently falls back to English "
              "on unreadable tokens — see msbench.corpus.phonemes.",
        tags=("cyrillic_special", "vowel_harmony"),
    ),
}

# Convenience views used across the pipeline.
CODES = list(LANGUAGES)
CV_LOCALE = {k: v.locale for k, v in LANGUAGES.items()}
ESPEAK_VOICE = {k: v.espeak for k, v in LANGUAGES.items()}
N_MAIN = {k: v.n_main for k, v in LANGUAGES.items()}
CAP = {k: v.cap for k, v in LANGUAGES.items()}
TARGET_F = {k: v.target_f for k, v in LANGUAGES.items()}
LEN_TARGET = {k: v.len_target for k, v in LANGUAGES.items()}
ALPHABET = {k: v.alphabet for k, v in LANGUAGES.items()}
NUM2WORDS = {k: v.num2words for k, v in LANGUAGES.items()}
LANG_TAGS = {k: list(v.tags) for k, v in LANGUAGES.items()}


def select(df, code):
    """Rows of `df` belonging to the language variant `code`."""
    spec = LANGUAGES[code]
    labels = df[spec.column].fillna("").apply(split_labels)
    return df[labels.apply(spec.match)].copy()


def prompt_dur_bin(seconds: float) -> str:
    if seconds < 3.7:
        return "<3.7"
    return "3.7-4.5" if seconds < 4.5 else ">=4.5"


def len_bin(n_words: int) -> str:
    if n_words <= 5:
        return "3-5"
    if n_words <= 9:
        return "6-9"
    if n_words <= 12:
        return "10-12"
    if n_words <= 16:
        return "13-16"
    return "17+"


def _selfcheck():
    for code, spec in LANGUAGES.items():
        assert spec.code == code
        assert len(spec.len_target) == len(LEN_BINS), code
        assert abs(sum(spec.len_target) - 1.0) < 1e-9, (code, sum(spec.len_target))
        assert spec.asr == "gigaam" or spec.asr.startswith("whisper:"), code
        # v2 must read every subset with at least two recogniser families: one
        # ASR cannot separate a system's errors from its own idiosyncrasies.
        assert len(spec.asr_v2) >= 2, code
        assert all(b in {"whisper", "mms", "gigaam"} for b in spec.asr_v2), code
    # the parser must survive labels that contain commas inside parentheses
    hard = ("España: Norte peninsular (Asturias, Castilla y León),"
            "España: Islas Canarias")
    assert split_labels(hard) == [
        "España: Norte peninsular (Asturias, Castilla y León)",
        "España: Islas Canarias"], split_labels(hard)
    # and exact matching must not confuse Caribbean Spanish with Mexican
    caribbean = split_labels("Caribe: Cuba, Venezuela, México caribeño, "
                             "Costa del golfo de México")
    assert not LANGUAGES["es-MX"].match(caribbean), caribbean
    print(f"{len(LANGUAGES)} languages registered: {', '.join(CODES)}")
    for code, spec in LANGUAGES.items():
        print(f"  {code:<6} locale={spec.locale:<3} espeak={spec.espeak:<7} "
              f"N={spec.n_main:<5} cap={spec.cap}  ASR={spec.asr}")


if __name__ == "__main__":
    _selfcheck()
