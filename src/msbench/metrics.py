#!/usr/bin/env python3
"""WER/CER aggregation, separated from inference.

Aggregation lives here so it can be recomputed offline from stored `ref_norm` /
`hyp_norm` without a GPU, and so it can be unit-tested against hand-checked
edit distances. v1's `eval/run_wer.py` does both jobs in one loop, which is why
the two defects below were invisible.

**D-01 — macro versus corpus.** v1 computes `jiwer.wer` per utterance and takes
the mean (`run_wer.py:188`, `report.py:50`). seed-tts-eval — the ecosystem the
dataset card claims compatibility with — sums the edit operations over the whole
corpus and divides by the total reference length:

    wer_corpus = Σ(S + D + I) / Σ N_ref          headline in v2
    wer_macro  = mean_u( (S+D+I)_u / N_ref,u )   v1 legacy, retained

These are different statistics, not different roundings. The corpus form weights
an utterance by its length; the macro form gives a 4-word sentence the same vote
as a 16-word one, so on a corpus with a median of 6-7 words one wrong word in a
short sentence moves the number as much as four wrong words in a long one. Both
are reported, always labelled, so the v1→v2 delta stays auditable.

**D-04 — vanished rows.** v1 returns early when the normalised reference is
empty (`run_wer.py:146`), so the row is neither scored nor counted as missing:
`n` shrinks with no trace. Here `n_empty_ref` is a first-class output.

`catastrophic_rate` (WER > 0.5) is the failure indicator that matters for TTS:
looping, babbling and dropped clauses show up there long before they move the
mean. The full per-utterance distribution is returned too, so users can pick
their own threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

CATASTROPHIC = 0.5


@dataclass
class WERResult:
    """Both aggregations plus the counts needed to re-derive either one."""

    wer_corpus: float
    cer_corpus: float
    wer_macro: float
    cer_macro: float
    ins: int
    dels: int
    subs: int
    hits: int
    n_ref_words: int
    n_ref_chars: int
    exact_match: float
    catastrophic_rate: float
    n_scored: int
    n_missing: int
    n_empty_ref: int
    per_utt: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("per_utt")
        return d


def _counts(refs: list[str], hyps: list[str], *, unit: str):
    """S/D/I/H per pair from jiwer's alignment. Returns (per_pair, totals).

    `unit="word"` tokenises on whitespace, `unit="char"` on characters — and the
    two must go through jiwer's own entry points. Routing characters through
    `process_words` looks equivalent but is not: the space character becomes an
    empty token and disappears, which silently drops every word-boundary error
    from the CER.
    """
    import jiwer

    proc = jiwer.process_words if unit == "word" else jiwer.process_characters
    length = (lambda s: len(s.split())) if unit == "word" else len

    per, tot = [], {"ins": 0, "dels": 0, "subs": 0, "hits": 0, "n_ref": 0}
    for r, h in zip(refs, hyps, strict=True):
        o = proc([r], [h])
        s, d, i, hi = o.substitutions, o.deletions, o.insertions, o.hits
        n = length(r)
        per.append({"subs": s, "dels": d, "ins": i, "hits": hi, "n_ref": n,
                    "err": s + d + i, "rate": (s + d + i) / n if n else np.nan})
        tot["subs"] += s
        tot["dels"] += d
        tot["ins"] += i
        tot["hits"] += hi
        tot["n_ref"] += n
    return per, tot


def score_pairs(pairs: list[tuple[str, str]], *, n_missing: int = 0,
                utts: list[str] | None = None) -> WERResult:
    """Aggregate a list of (normalised reference, normalised hypothesis) pairs.

    Rows whose reference normalises to empty are dropped from the metric — a WER
    with a zero denominator is undefined — but counted in `n_empty_ref` (D-04).
    """
    utts = utts or [""] * len(pairs)
    keep, empty = [], 0
    for u, (r, h) in zip(utts, pairs, strict=True):
        if not r.strip():
            empty += 1
            continue
        keep.append((u, r, h))

    if not keep:
        return WERResult(*([float("nan")] * 4), 0, 0, 0, 0, 0, 0,
                         float("nan"), float("nan"), 0, n_missing, empty, [])

    refs = [r for _, r, _ in keep]
    hyps = [h for _, _, h in keep]
    w_per, w_tot = _counts(refs, hyps, unit="word")
    # The space is a character. `jiwer.cer` — what v1 called — counts it, so word
    # boundaries are part of the CER, and dropping them (the CJK convention)
    # would silently redefine the metric and break continuity with v1. All six
    # subsets are space-delimited scripts, so keeping it is also right on the
    # merits: a model that runs two words together made an error.
    c_per, c_tot = _counts(refs, hyps, unit="char")

    wer_u = np.array([p["rate"] for p in w_per], dtype=float)
    cer_u = np.array([p["rate"] for p in c_per], dtype=float)

    per_utt = [
        # `subs`/`dels`/`ins` and `cer_err` are the raw numerators: any
        # aggregation — corpus, macro, or a user's own slice — is re-derivable
        # from a per-utterance parquet without re-running the ASR.
        {"utt": u, "ref_norm": r, "hyp_norm": h,
         "wer": w["rate"], "cer": c["rate"],
         "subs": w["subs"], "dels": w["dels"], "ins": w["ins"],
         "cer_err": c["err"],
         "n_ref_words": w["n_ref"], "n_ref_chars": c["n_ref"]}
        for (u, r, h), w, c in zip(keep, w_per, c_per, strict=True)
    ]

    return WERResult(
        wer_corpus=(w_tot["subs"] + w_tot["dels"] + w_tot["ins"]) / w_tot["n_ref"],
        cer_corpus=(c_tot["subs"] + c_tot["dels"] + c_tot["ins"]) / c_tot["n_ref"],
        wer_macro=float(np.mean(wer_u)),
        cer_macro=float(np.mean(cer_u)),
        ins=w_tot["ins"], dels=w_tot["dels"], subs=w_tot["subs"],
        hits=w_tot["hits"], n_ref_words=w_tot["n_ref"], n_ref_chars=c_tot["n_ref"],
        exact_match=float(np.mean(wer_u == 0)),
        catastrophic_rate=float(np.mean(wer_u > CATASTROPHIC)),
        n_scored=len(keep), n_missing=n_missing, n_empty_ref=empty,
        per_utt=per_utt,
    )


def score_frame(df, ref_col: str = "ref_norm", hyp_col: str = "hyp_norm",
                utt_col: str = "utt", missing_col: str = "missing") -> WERResult:
    """Re-aggregate an existing per-utterance parquet. No GPU, no ASR."""
    n_missing = int(df[missing_col].sum()) if missing_col in df else 0
    ok = df[~df[missing_col]] if missing_col in df else df
    pairs = list(zip(ok[ref_col].fillna(""), ok[hyp_col].fillna(""), strict=True))
    return score_pairs(pairs, n_missing=n_missing, utts=list(ok[utt_col]))


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
