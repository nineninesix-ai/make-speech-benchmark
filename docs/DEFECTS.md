# What was wrong with v1, and what it cost

Version 1 of this benchmark was published with defects. This document lists all
of them, says which ones moved a published number and by how much, and records
which are fixed, which are disclosed-but-unfixed, and which remain open.

It exists because a benchmark's credibility rests on what its authors went
looking for and failed to find, not on the numbers they chose to print. Every
entry below was found by auditing v1 after publication — some by external
review, most by re-reading the code against its own card.

**The headline result of the audit:** v1's numbers were *arithmetically correct
for the aggregation it used*. Replaying v1's exact decoding parameters on
current libraries reproduces every published anchor to within 0.0007 (see
[attribution](../reports/attribution.md)). The entire v1 → v2 change in the WER
anchor is **D-01**, a redefinition of the aggregation — not a bug fix.

| | count |
|---|---|
| Defects found | 27 |
| That moved a published anchor value | 1 (D-01) |
| False statements in the v1 card | 5 (D-12, D-13, D-14, D-15, D-26) |
| Fixed in v2.0 | 22 |
| Disclosed, not fixable by construction | 4 (D-11, D-17, D-18, D-19) |
| Open | 1 (D-02, untestable without a synthesis run) |

---

## The one that changed the headline

### D-01 — WER was macro-averaged, not corpus-level

`ok.wer.mean()` averages per-utterance rates. seed-tts-eval — the protocol v1's
card claimed compatibility with — uses `Σ(S+D+I) / Σ N_ref`. On texts with a
median of 6–7 words the two differ materially: one wrong word in a four-word
sentence contributes 25 % to a macro mean regardless of its weight in the
corpus, so short utterances dominate.

**Measured cost**, corpus minus published macro:

| subset | v1 published (macro) | v2 (corpus) | delta |
|---|---|---|---|
| en-US | 0.0806 | 0.0774 | −0.0033 |
| es-ES | 0.0485 | 0.0452 | −0.0032 |
| es-MX | 0.0654 | 0.0613 | −0.0035 |
| nl-NL | 0.0407 | 0.0385 | −0.0023 |
| pt-BR | 0.0823 | 0.0737 | −0.0080 |
| ky | 0.1104 | 0.0998 | −0.0106 |

**Status: fixed.** v2 reports `wer_corpus` as the headline and keeps
`wer_macro` alongside it, so v1 numbers stay comparable. Both are asserted
bit-for-bit in `tests/test_metrics.py`.

---

## Defects in the measurement code

### D-02 — `repetition_penalty=1.1` on the ASR decoder

Non-standard for WER evaluation. It suppresses repeated tokens, which is the
exact signature of the most important autoregressive-TTS failure mode: looping,
babbling, repeated syllables. It was applied identically to the anchor, but
human speech rarely loops, so the anchor barely moves while model failures are
masked. The bias is asymmetric.

**Measured cost on the anchor: ≤ 0.0004 in every subset** — negligible, exactly
as predicted, because the anchor is human speech.

**Status: removed in v2; the effect on synthesis is OPEN.** The claim that it
masked model failures is a claim about synthesis, and this repository contains
no synthesis run. It cannot be settled here. Anyone evaluating a looping model
against v1 numbers should assume they were flattered.

### D-03 — resampling by `np.interp`

Linear interpolation with no anti-aliasing filter. References are natively
16 kHz and never took that path; synthesis at 22.05 / 24 / 44.1 kHz always did,
and arrived at the speaker encoder carrying aliasing artifacts. A 15 kHz tone
that must vanish at 16 kHz survived at −3.1 dBr with 93 % of the residue folded
onto 1 kHz, in the middle of the speech band.

**Every model's SIM was systematically depressed by a technical defect; the
anchor was not.**

**Status: fixed.** All resampling goes through soxr HQ (`msbench.audio`).

### D-04 — empty references silently dropped

`if not r_n: return` — rows whose normalised reference was empty vanished
without being counted. `n` shrank with no trace.

**Status: fixed.** `n_empty_ref` is counted and reported. In this dataset the
count is 0 in every subset, so the defect existed in v1's code but never
affected a published number.

### D-05 — no preprocessing parity for synthesis

The anchors were computed on audio that the build had made 16 kHz,
VAD-trimmed and peak-normalised to −1 dBFS. `run_sim.py` applied none of that
to synthesis. Trailing silence and level differences both shift SIM.

**Status: fixed.** `msbench.audio.prepare()` is the single path, applied by
default in every driver, and is the function [PROTOCOL.md](PROTOCOL.md) tells
third parties to call.

### D-06 — prompt embedding recomputed per row

One prompt clip serves up to 7 examples. Wasteful, not wrong.

**Status: fixed**, cached by `speaker_id`.

### D-10 — `chunk_length_s=30` on 2–6 s clips

Engaged Whisper's long-form chunking algorithm for short-form audio. An
undocumented code path affecting reproducibility.

**Measured cost: exactly 0.0000 in all six subsets**, and hypotheses agree at
100 %. Harmless in practice, as suspected.

**Status: fixed** (short-form path), and now documented either way.

---

## Defects in the dataset schema

| id | defect | status |
|---|---|---|
| **D-07** | `qc_bandwidth_hz` held decibels, not hertz (values ≈ −26.95: energy above 5 kHz in dB) | fixed — renamed `qc_hf_energy_db` |
| **D-08** | all `qc_*` metrics computed on the **untrimmed** clip while `prompt_audio` ships trimmed, which is why `qc_lead_sil` is non-zero on trimmed audio | fixed — documented in the schema, not renamed |
| **D-09** | audio QC ran on prompt candidates only; `gt_audio` and `sim_ref_audio` got the same preprocessing but never passed the rejection filters, so the human WER anchor includes GT recordings that would have been rejected as prompts | fixed — QC now runs on all three streams |
| **D-22** | Kyrgyz `phones` contained espeak dental markers `t[` / `d[` while the card claimed IPA | fixed — `t̪` / `d̪` |
| **D-23** | `tags` had dtype `list<null>`, which cannot hold a value | fixed — `list<string>` |

---

## Defects in the v1 card: claims that did not match the data

Found by checking the prose against the artifacts it described. All five were
wrong in the card, not in the data.

| id | the card said | the data said |
|---|---|---|
| **D-12** | `sim_ref_audio` missing for "26–39 %" | **12–39 %** (nl-NL 12.1, ky 14.4, es-ES 18.5, es-MX 19.9, pt-BR 34.5, en-US 39.4) |
| **D-13** | SIM-vs-duration monotonic in five subsets | monotonic in **three** (en-US, es-MX, nl-NL); es-ES and pt-BR peak in the middle bin |
| **D-14** | every example carries GT audio and a second clip | `has_gt` false for 7 examples; `has_sim_ref` 61–88 % |
| **D-15** | "Speakers: 1,162" (used) next to §5.3's 1,736 (available), unlabelled | renamed "speakers used" |
| **D-26** | GigaAM arXiv id 2607.10371 | did not match the known GigaAM paper |

**Status: all fixed, and structurally prevented.** The v2 card is *generated*
from the artifacts by `msbench.build.card`, so prose and data cannot drift
apart again. That is the real fix; correcting five sentences would not have
been one.

---

## Defects in the statistics

### D-20 — point estimates with no interval, on non-independent observations

One prompt serves up to 7 examples, GT-speaker concentration is uncapped, and
the SIM anchor is constant within a speaker — so the published `n` counted rows,
not measurements. pt-BR's SIM anchor has `n = 983` rows but only **161 distinct
values**.

**Status: fixed.** Every headline carries a 95 % cluster bootstrap over speakers
(2,000 replicates, seed 20260725), and "A beats B" claims use a paired bootstrap
over shared speakers. Every breakdown cell carries its `n` and speaker count and
is flagged below 30 speakers.

### D-21 — the SIM-vs-duration table was confounded

It binned on `prompt_dur` only, while `sim_ref_dur` — down to ~2 s — was
uncontrolled, and confounds the `ky` non-monotonicity.

**Status: fixed.** `sim_ref_dur_bin` ships in the dataset and the breakdown
controls for it.

---

## Disclosed, not fixable by construction

These are properties of what the benchmark measures, not errors in how it
measures. They are recorded so no one mistakes them for oversights.

- **D-11 — the anchor is not a physical ceiling.** Models can and do exceed it.
  Synthesis is conditioned on the prompt and inherits its recording channel,
  whereas the anchor compares two *different* recordings of the same person.
  Asymmetric by construction.
- **D-17 — circularity in the SIM anchor.** v1's QC removed clips beyond μ−2σ
  from their speaker's WavLM-SV centroid, then measured the anchor with the same
  embedding family: the pool was pre-selected for homogeneity in the metric's
  own space. *Mitigated*, not fixed, by adding ECAPA-TDNN — an encoder from
  outside the WavLM family — as a third anchor.
- **D-18 — the WER anchor mixes error sources.** ASR error, reader error and
  unverified crowd transcripts; and conversely Whisper has plausibly seen Common
  Voice in training. Two unquantified biases pointing in opposite directions.
  *Mitigated* by reporting three recognisers instead of one, so at least the
  spread is visible.
- **D-19 — cross-language comparison is not supported.** Anchor differences are
  a property of the recogniser, and for `ky` it is a different recogniser
  entirely. v1's card invited the comparison; v2's forbids it.

---

## Process defects

- **D-16 — the seed-tts-eval compatibility claim was wrong.** v1 advertised
  numbers "consumable by the unmodified seed-tts-eval scripts", but those
  scripts use `wavlm_large_finetune.pth` (scale ≈ 0.70–0.76) while v1's anchors
  came from `microsoft/wavlm-base-plus-sv` (scale ≈ 0.92–0.95). A user of the
  stock scripts would have compared ~0.7 against an anchor of 0.93. There was a
  second, independent incomparability: Seed-TTS uses 3–20 s prompts, this
  dataset 2.5–5 s, and SIM rises with prompt duration.
  **Fixed** — `wavlm_ft` (the stock checkpoint) is now measured and shipped, and
  the compatibility claim is scoped to it explicitly.
- **D-24 — reports were in Russian under an English card**, and the `reports/`
  folder was not mentioned in the card at all. **Fixed.**
- **D-25 — the pipeline was not published**, so v1's determinism claim was a
  promise rather than a recipe. **Fixed by this repository.**
- **D-27 — the `ky` anchor rested on an unvalidated recogniser.** At 11.04 % it
  sat above the 10 % threshold v1's own card named as requiring a cross-check,
  and no cross-check was done. **Fixed**, from three directions: GigaAM's
  MMS-relative ratio (2.43×) is mid-range among the six subsets; ElevenLabs
  Scribe v2, an independent hosted model, loses to GigaAM by +0.0831
  [0.0629, 0.1046]; and a Kyrgyz-specific Whisper fine-tune also loses to it.

---

## Found while fixing the above

Not v1 defects — things the v2 work turned up that a user should know.

- **Kazakh orthography in the Kyrgyz anchor.** GigaAM writes `і ғ қ ұ` — Kazakh
  letters, not Kyrgyz — in 48 of 694 transcripts (6.9 %). These score as errors
  against references that use only the 36 Kyrgyz letters. Folding them would
  move the `ky` anchor from 0.0998 to **0.0892**. The published anchor does not
  fold them, because the references do not; but part of that anchor is a
  spelling convention rather than a mishearing. ElevenLabs Scribe has the same
  problem slightly less (6.2 %, 0.0063 of WER).
- **ElevenLabs Scribe is not deterministic, even pinned.** With `seed=0` and
  `temperature=0`, only 69.5 % of transcripts are identical across three
  end-to-end runs of `ky`; corpus WER lands at 0.1829 / 0.1805 / 0.1809, a range
  of 0.0024 against a bootstrap interval 0.078 wide. Negligible in aggregate,
  real per row: audit a single row and you may not reproduce it.
- **Scribe hallucinates on short clips.** On a 1.1 s pt-BR utterance where
  Whisper scores 0.00 it returned an unrelated sentence, with
  `audio_duration_secs` matching the clip — a decoder failure, not a transport
  one. Read its `catastrophic_rate` alongside its WER.
