#!/usr/bin/env python3
"""Gate A — the aggregation is correct, and its relationship to v1 is exact.

The plan's original gate (§11.1) was to re-aggregate the stored `ref_norm` /
`hyp_norm` from `reports/wer_{lang}_gt.parquet` and reproduce the published v1
anchors to four decimals. Those parquets do not exist: they are written inside
`tts-bench-v1/`, which is gitignored, and the published dataset carries only the
three markdown reports. The artifact the gate depends on is itself a v2
deliverable (§6, last row).

So the gate is split. This file is the exact half: it fixes the inputs, so the
only thing under test is the arithmetic, and every assertion is to machine
precision rather than to a tolerance.

  1. `wer_macro` reproduces v1's literal `mean(jiwer.wer(r, h))` exactly. Any
     future v1→v2 delta is then attributable to a deliberate change, not to this
     rewrite.
  2. `wer_corpus` reproduces `Σ(S+D+I) / Σ N_ref` — the seed-tts-eval definition
     — on cases whose edit distances are counted by hand here.
  3. The two disagree in the documented direction, so D-01 is a real effect and
     not a rounding story.
  4. Empty references are counted, not silently dropped (D-04).

The inexact half — re-running v1's decode settings and comparing against
Appendix B — lives in `msbench.cli.replay`, where environment drift is
measured and reported rather than asserted.

    pytest
"""


import numpy as np
import pytest

from msbench.metrics import score_pairs


def v1_macro(pairs):
    """The v1 aggregation, verbatim: `jiwer.wer` per row, then `.mean()`.

    `v1's eval/run_wer.py:149` computes the per-row rate, `v1's eval/report.py:50` takes
    the mean of the column. Reproduced here so the comparison is against the old
    code path itself, not against a description of it.
    """
    import jiwer
    return float(np.mean([jiwer.wer(r, h) for r, h in pairs]))


def v1_macro_cer(pairs):
    import jiwer
    return float(np.mean([jiwer.cer(r, h) for r, h in pairs]))


# Hand-counted cases. `err` is S+D+I against the reference, verified by reading
# the alignment; `n` is the reference word count.
CASES = [
    # (reference, hypothesis, err, n, note)
    ("the cat sat on the mat", "the cat sat on the mat", 0, 6, "identical"),
    ("the cat sat on the mat", "the cat sat on a mat", 1, 6, "one substitution"),
    ("the cat sat", "the cat", 1, 3, "one deletion"),
    ("the cat sat", "the big cat sat", 1, 3, "one insertion"),
    ("yes", "no", 1, 1, "short: one error is 100%"),
    ("a b c d e f g h i j k l", "a b c d e f g h i j k x", 1, 12,
     "long: one error is 8.3%"),
    ("hello world", "", 2, 2, "empty hypothesis — model produced nothing"),
    ("one two three", "one two three four five", 2, 3, "two insertions"),
]


PAIRS = [(r, h) for r, h, *_ in CASES]


@pytest.fixture(scope="module")
def res():
    return score_pairs(PAIRS, utts=[f"u{i}" for i in range(len(PAIRS))])


@pytest.mark.parametrize(("ref", "hyp", "err", "n", "note"), CASES,
                         ids=[c[4] for c in CASES])
def test_edit_counts_match_hand_alignment(ref, hyp, err, n, note):
    """1. Per-utterance S+D+I and reference length, against counts done by hand."""
    row = score_pairs([(ref, hyp)]).per_utt[0]
    assert row["subs"] + row["dels"] + row["ins"] == err
    assert row["n_ref_words"] == n


def test_wer_corpus_is_seed_tts_eval(res):
    """2. wer_corpus == sum(S+D+I) / sum(N_ref)."""
    want = sum(c[2] for c in CASES) / sum(c[3] for c in CASES)
    assert res.wer_corpus == pytest.approx(want, abs=1e-12)
    assert res.n_ref_words == sum(c[3] for c in CASES)


def test_wer_macro_reproduces_v1_bit_for_bit(res):
    """3. Any v1->v2 delta is a deliberate change, not an artifact of the rewrite."""
    assert res.wer_macro == v1_macro(PAIRS)
    assert res.cer_macro == v1_macro_cer(PAIRS)


def test_aggregations_differ_in_the_documented_direction(res):
    """4. D-01 is a real effect: short utterances are over-weighted by the mean."""
    assert res.wer_macro != res.wer_corpus
    assert res.wer_macro > res.wer_corpus


def test_empty_references_are_counted_not_dropped(res):
    """5. D-04."""
    with_empty = PAIRS + [("", "hallucinated text"), ("   ", "more")]
    r2 = score_pairs(with_empty, utts=[f"u{i}" for i in range(len(with_empty))])
    assert r2.n_empty_ref == 2
    assert r2.n_scored == res.n_scored
    assert r2.wer_corpus == pytest.approx(res.wer_corpus, abs=1e-12)
    assert score_pairs(PAIRS, n_missing=7).n_missing == 7


def test_failure_indicators(res):
    """6. exact_match and catastrophic_rate.

    'yes'->'no' (1.0), 'hello world'->'' (1.0) and 'one two three'+2 (0.667)
    are the three rows above the 0.5 threshold.
    """
    assert res.exact_match == pytest.approx(1 / len(CASES), abs=1e-12)
    assert res.catastrophic_rate == pytest.approx(3 / len(CASES), abs=1e-12)


def test_degenerate_corpus_returns_nan_with_counts_intact():
    """7. An all-empty input must not raise."""
    empty = score_pairs([("", "x")])
    assert empty.n_scored == 0
    assert empty.n_empty_ref == 1
    assert np.isnan(empty.wer_corpus)
