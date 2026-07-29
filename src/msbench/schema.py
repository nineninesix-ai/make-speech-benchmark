#!/usr/bin/env python3
"""Dataset schema — the contract for what ends up in the published parquet.

Imported by the selection (S3), packaging (S4) and eval stages so that
SCHEMA.md and the actual columns cannot drift apart. Per-language settings live
in `languages.py`; this module only re-exports them for convenience.

Run `python -m msbench.schema` to self-check the schema and the capacity margins.
"""


from datasets import Audio, Features, Sequence, Value

from msbench.languages import (  # noqa: E402,F401  (re-exported on purpose)
    CAP,
    CODES,
    CV_LOCALE,
    ESPEAK_VOICE,
    LANG_TAGS,
    LANGUAGES,
    LEN_BINS,
    LEN_TARGET,
    N_MAIN,
    PROMPT_DUR_BINS,
    TARGET_F,
    len_bin,
    prompt_dur_bin,
)

LANGS = CODES
N_HARD = {code: 200 for code in CODES}

# Share of question / exclamation texts available in the Common Voice pool:
#   en-US 0.9/0.7  es-ES 0.5/0.6  es-MX 0.2/0.3  ky 4.4/0.7  nl 5.3/1.4  pt 9.0/1.8
# The original ">=10% of each" quota is unreachable for every language — CV
# sentences are overwhelmingly declarative. The selector takes everything
# available up to a 10% cap; genuine prosody testing needs the hard set.
PUNCT_BEST_EFFORT = True

# --- Hard-set taxonomy (see SCHEMA.md section 3) ----------------------------
# Not built in v1. The schema reserves subset="hard" and these categories so
# the hard set can be added later without a breaking change.

HARD_CATEGORIES = {
    "numeric": ["cardinal", "ordinal", "decimal", "fraction", "range", "phone", "large"],
    "datetime": ["date", "time", "year", "duration"],
    "currency_unit": ["currency", "measure", "percent"],
    "abbreviation": ["initialism", "acronym", "contraction", "title"],
    "proper_noun": ["person", "place", "brand", "org"],
    "loanword": ["anglicism", "code_switch"],
    "homograph": ["stress_pair", "pos_pair"],
    "phonetic_stress": ["tongue_twister", "cluster", "minimal_pair"],
    "repetition": ["word_repeat", "phrase_repeat"],
    "long_form": ["multi_sentence", "25plus"],
    "prosody": ["question", "exclamation", "list_enum", "parenthetical", "quote"],
    "symbol": ["url", "email", "hashtag", "math"],
    "orthography": ["diacritic_dense", "compound", "apostrophe"],
}

PUNCT_TYPES = ["declarative", "question", "exclamation", "mixed"]
GENDERS = ["male", "female", "unknown"]
SPLITS = ["train", "dev", "test"]

# --- Column schema ----------------------------------------------------------

FEATURES = Features({
    # identification
    "utt": Value("string"),
    "lang": Value("string"),
    "subset": Value("string"),
    "source": Value("string"),
    "cv_version": Value("string"),

    # prompt: the reference the model clones from
    "prompt_audio": Audio(sampling_rate=16000),
    "prompt_audio_orig": Audio(),
    "prompt_text": Value("string"),
    "prompt_dur": Value("float32"),          # after VAD trimming
    "prompt_sr_orig": Value("int32"),
    "prompt_dur_bin": Value("string"),
    "prompt_cv_path": Value("string"),
    "prompt_split_origin": Value("string"),  # lets users filter a held-out subset

    # prompt speaker
    "speaker_id": Value("string"),
    "speaker_gender": Value("string"),
    "speaker_gender_source": Value("string"),
    "speaker_age": Value("string"),
    "speaker_accent_label": Value("string"),
    "speaker_n_in_subset": Value("int16"),

    # target text
    "text": Value("string"),
    "text_norm": Value("string"),
    "n_words": Value("int16"),
    "n_chars": Value("int16"),
    "len_bin": Value("string"),
    "phones": Value("string"),
    "n_phones": Value("int16"),
    "punct_type": Value("string"),
    "text_cv_path": Value("string"),
    "text_split_origin": Value("string"),

    # ground truth for the WER topline; the speaker is arbitrary by design
    "gt_audio": Audio(sampling_rate=16000),
    "has_gt": Value("bool"),
    "gt_dur": Value("float32"),
    "gt_speaker_id": Value("string"),
    "gt_same_speaker": Value("bool"),
    "gt_gender": Value("string"),

    # SIM topline: a second clip of the prompt speaker, any text
    "sim_ref_audio": Audio(sampling_rate=16000),
    "has_sim_ref": Value("bool"),
    "sim_ref_text": Value("string"),
    "sim_ref_dur": Value("float32"),

    # QC. qc_snr_db and qc_asr_cer are always null in v1 — not computed.
    "qc_dnsmos_ovrl": Value("float32"),
    "qc_dnsmos_sig": Value("float32"),
    "qc_dnsmos_bak": Value("float32"),
    "qc_snr_db": Value("float32"),
    "qc_asr_cer": Value("float32"),
    "qc_vad_speech_ratio": Value("float32"),
    "qc_lead_sil": Value("float32"),
    "qc_trail_sil": Value("float32"),
    "qc_clip_rate": Value("float32"),
    "qc_bandwidth_hz": Value("float32"),
    "qc_spk_centroid_dist": Value("float32"),

    # categories: meaningful only for the hard set
    "category": Value("string"),
    "subcategory": Value("string"),
    "tags": Sequence(Value("string")),
    "difficulty": Value("int8"),
    "has_digit": Value("bool"),
    "has_abbrev": Value("bool"),
    "has_foreign": Value("bool"),
    "notes": Value("string"),
})

# meta.lst field order required by seed-tts-eval
META_LST_FIELDS = ["utt", "prompt_text", "prompt_wav", "infer_text", "infer_wav"]


def meta_lst_line(row, prompt_wav, infer_wav):
    """One meta.lst line. '|' is the separator, so it must not occur in a field."""
    fields = [row["utt"], row["prompt_text"], prompt_wav, row["text"], infer_wav]
    for f in fields:
        if "|" in str(f):
            raise ValueError(f"'|' present in a meta.lst field: {f!r}")
    return "|".join(str(f) for f in fields)


def _selfcheck():
    # Capacity: cap * speakers must cover N. The counts below are speakers that
    # actually received a prompt after QC (S2), not raw slice sizes.
    speakers = {"es-MX": 929, "pt-BR": 247, "nl-NL": 469,
                "en-US": 1736, "es-ES": 722, "ky": 181}
    tight = []
    for code in CODES:
        ceiling = CAP[code] * speakers[code]
        margin = ceiling - N_MAIN[code]
        if margin < 0:
            tight.append(code)
        print(f"  {code:<6} N={N_MAIN[code]:>4}  cap={CAP[code]}  "
              f"ceiling={ceiling:>5}  margin={margin:>+5}"
              f"{'  <-- raise cap' if margin < 0 else ''}")
    assert not tight, f"capacity exceeded for {tight}"

    n_audio = sum(isinstance(v, Audio) for v in FEATURES.values())
    n_sub = sum(len(v) for v in HARD_CATEGORIES.values())
    print(f"\ncolumns: {len(FEATURES)}  (audio: {n_audio})")
    print(f"hard set: {len(HARD_CATEGORIES)} categories, {n_sub} subcategories")


if __name__ == "__main__":
    _selfcheck()
