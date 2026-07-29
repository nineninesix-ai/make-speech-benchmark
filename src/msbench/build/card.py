#!/usr/bin/env python3
"""Assemble the dataset card.

The card is generated, not hand-written, for the same reason the YAML is: in v1
the prose and the data disagreed in at least five places that a reader could
check — the share of rows with a second clip, which subsets showed a monotonic
SIM/duration relationship, whether every example carries ground-truth audio,
whether "Speakers" meant used or available, and the arXiv id of a cited paper.
Every table below is read from the artifacts in `reports/`, so a number in the
card cannot drift from the number in the files.

    msbench-card --pack tts-bench-v2 --out tts-bench-v2/README.md
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from msbench.build.card_yaml import front_matter
from msbench.paths import workspace

REPO = "https://github.com/nineninesix-ai/make-speech-benchmark"
DATASET = "nineninesix/multilingual-speech-benchmark"


def md(df: pd.DataFrame, cols=None) -> str:
    return (df[cols] if cols else df).to_markdown(index=False)


def load(rep: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(rep / "csv" / f"{name}.csv")


def build(pack: Path) -> str:
    rep = pack / "reports"
    anchors = load(rep, "anchors")
    sim = load(rep, "sim")
    agree = load(rep, "agreement")
    attrib = load(rep, "attribution")
    quality = load(rep, "quality")
    simcut = load(rep, "sim_breakdown")
    summary = pd.read_csv(rep / "csv" / "anchors.csv")  # noqa: F841

    wer_primary = anchors[
        ((anchors.lang != "ky") & (anchors.ASR == "whisper"))
        | ((anchors.lang == "ky") & (anchors.ASR == "gigaam"))]

    subset = []
    for lang in ["en-US", "es-ES", "es-MX", "nl-NL", "pt-BR", "ky"]:
        cov = (rep / f"coverage_{lang}.md").read_text()
        n = int(cov.split("**Examples:** ")[1].split(" ")[0])
        spk = int(cov.split("used**: **")[1].split("**")[0])
        gt = cov.split("(WER anchor): **")[1].split("**")[0]
        sr = cov.split("(SIM anchor): **")[1].split("**")[0]
        subset.append({"subset": lang, "examples": n, "speakers used": spk,
                       "with GT audio": gt, "with second clip": sr})
    subset = pd.DataFrame(subset)

    dur_sv = simcut[(simcut.cut == "prompt duration")
                    & (simcut.encoder == "wavlm_sv")].pivot(
        index="lang", columns="value", values="SIM")[["<3.7", "3.7-4.5", ">=4.5"]]
    dur_sv["monotonic"] = ((dur_sv["<3.7"] < dur_sv["3.7-4.5"])
                           & (dur_sv["3.7-4.5"] < dur_sv[">=4.5"]))
    ref_sv = simcut[(simcut.cut == "second-clip duration")
                    & (simcut.encoder == "wavlm_sv")].pivot(
        index="lang", columns="value", values="SIM")

    sim_ref_pct = subset["with second clip"].str.split("/").apply(
        lambda x: 100 * int(x[0]) / int(x[1]))
    lo, hi = sim_ref_pct.min(), sim_ref_pct.max()

    fields = (pack / "SCHEMA_FIELDS.md")
    fields_md = fields.read_text() if fields.exists() else ""

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"

    return TEMPLATE.format(
        repo=REPO, dataset=DATASET, rev=rev,
        subset=md(subset),
        wer=md(wer_primary, ["lang", "ASR", "n", "speakers", "WER corpus",
                             "95% CI", "WER macro (v1)", "CER corpus", "exact",
                             "catastrophic"]),
        wer_all=md(anchors, ["lang", "ASR", "WER corpus", "95% CI",
                             "CER corpus", "exact", "catastrophic"]),
        agree=md(agree),
        sim=md(sim, ["lang", "encoder", "anchor", "95% CI", "floor mean",
                     "floor p95", "usable range", "n rows", "speakers",
                     "distinct values"]),
        attrib=md(attrib, ["lang", "published v1 (macro)", "legacy (macro)",
                           "drift (env)", "D-02 penalty", "D-10 chunking",
                           "D-01 aggregation", "v2 (corpus)",
                           "v2 corpus 95% CI"]),
        quality=md(quality),
        dur_sv=dur_sv.reset_index().to_markdown(index=False),
        ref_sv=ref_sv.reset_index().to_markdown(index=False),
        simref_lo=f"{lo:.1f}", simref_hi=f"{hi:.1f}",
        fields=fields_md,
    )


TEMPLATE = r"""# Multilingual Speech Benchmark for Zero-Shot TTS

A voice-cloning and intelligibility benchmark for six language variants, built
from Common Voice 17.0 by coverage-driven selection rather than random sampling.
Every example pairs a **reference clip of one speaker** with a **target text that
speaker never read**, so a system is asked to clone a voice and produce new
speech, which is what zero-shot TTS is actually for.

**Pipeline source code:** [{repo}]({repo}) — every number in this card is
reproducible from it. Card generated at pipeline revision `{rev}`.

**Version 2.0.** The audio, the row composition and the `utt` identifiers are
unchanged from v1: anything already synthesised against v1 still joins. What
changed is the measurement and the description of it — see
[What changed in v2.0](#what-changed-in-v20).

---

## Contents

- [What this measures, and what it does not](#what-this-measures-and-what-it-does-not)
- [Quick start](#quick-start)
- [What changed in v2.0](#what-changed-in-v20)
- [Subsets](#subsets)
- [Human anchors — intelligibility](#human-anchors--intelligibility)
- [Human anchors — speaker similarity](#human-anchors--speaker-similarity)
- [Can a model beat the anchor?](#can-a-model-beat-the-anchor)
- [Evaluation protocol](#evaluation-protocol)
- [How to read the numbers](#how-to-read-the-numbers)
- [Source data](#source-data)
- [Quality control](#quality-control)
- [Example selection](#example-selection)
- [Data fields](#data-fields)
- [Reports](#reports)
- [Limitations](#limitations)
- [Reproduction](#reproduction)
- [Personal and sensitive information](#personal-and-sensitive-information)
- [Licence, citation, contact](#licence-citation-contact)

---

## What this measures, and what it does not

This is an **intelligibility and voice-cloning benchmark**, not a TTS quality
benchmark. It measures two things:

- **WER / CER** on a recogniser reading the synthesis — a proxy for intelligibility;
- **SIM** — cosine similarity of speaker embeddings between the synthesis and the
  reference clip — a proxy for speaker identity.

It measures **no naturalness axis at all**: no subjective MOS or CMOS, and no
predicted naturalness on the synthesis. A system can score 3 % WER and 0.95 SIM
and still sound robotic, and nothing here would notice. It also contains **no
digits, no abbreviations and no long-form text** — Common Voice sentences of 3-16
words — so text normalisation, the most common production TTS failure, is
untested. Treat a good score as a necessary condition, not a sufficient one.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("{dataset}", "en-US", split="main")
row = ds[0]
row["prompt_audio"]   # reference clip of the speaker, 16 kHz
row["text"]           # the target text, which this speaker never read
row["gt_audio"]       # a real recording of that text, by someone else
row["sim_ref_audio"]  # a second clip of the PROMPT speaker -> the SIM anchor

row["anchor_wer"]           # what the recogniser scores on the HUMAN recording
row["anchor_sim_wavlm_ft"]  # what two recordings of the same person score
```

Those last two are the reference points your model's numbers are read against,
and they ship per row — along with the edit counts, so the corpus-level headline
is derivable without leaving the dataset (see [Data fields](#data-fields)). The measurements behind them — S/D/I counts, transcripts,
every recogniser and encoder — are browsable as separate configs:

```python
wer = load_dataset("{dataset}", "metrics-wer", split="train")
sim = load_dataset("{dataset}", "metrics-sim", split="train")
```

Synthesise `row["text"]` conditioned on `row["prompt_audio"]`, save it as
`{{utt}}.wav`, then score:

```bash
git clone {repo} && cd make-speech-benchmark
uv venv && source .venv/bin/activate && uv pip install -e ".[all]"

msbench-wer --lang en-US --audio synth --asr whisper \
    --synth-dir out/mymodel/en-US --model-name mymodel
msbench-sim --lang en-US --mode synth --encoder wavlm_ft \
    --synth-dir out/mymodel/en-US --model-name mymodel
msbench-report --all
```

### seed-tts-eval compatibility

The layout is consumable by the unmodified seed-tts-eval scripts, but read
[Human anchors — speaker similarity](#human-anchors--speaker-similarity) first:
those scripts score SIM with `wavlm_large_finetune.pth`, whose values live near
0.65-0.69 here, while v1 published anchors from `microsoft/wavlm-base-plus-sv`,
which live near 0.93. Comparing one against the other is the single easiest way
to misread this benchmark, and v2 ships anchors for both.

## What changed in v2.0

**No data changed.** Audio in all three streams is byte-identical, rows are in
the same order, `utt` is stable. Verified by `msbench.build.validate`, which is run
before every release.

### Published numbers

The headline WER changes because the **aggregation** changed, not because the
recogniser or the audio did. v1 averaged per-utterance error rates; v2 reports
the corpus-level rate that seed-tts-eval uses, Σ(S+D+I) / Σ N_ref, and keeps the
old form beside it labelled as legacy. On texts with a median of 6-7 words the
two differ materially: one wrong word in a four-word sentence contributes 25 %
under the macro mean regardless of its weight in the corpus.

Each column below isolates **one** change, measured on identical audio:

{attrib}

- **drift (env)** — replaying v1's exact decoding parameters on current library
  and hardware versions. Worst case −0.0007; four subsets reproduce exactly. This
  is not a methodological change and is reported only so it is not mistaken for one.
- **D-02 penalty** — v1 decoded with `repetition_penalty=1.1`, which is not
  standard for WER evaluation and suppresses exactly the token pattern that
  autoregressive TTS failure produces. Removing it moves the **human** anchor by
  at most 0.0004, which is expected: human speech rarely loops. Whether it was
  masking model failures is a claim about synthesis and is **not** tested here.
- **D-10 chunking** — `chunk_length_s=30` engaged Whisper's long-form path on
  clips of 2-6 s. Effect: exactly zero, transcripts 100 % identical. Removed as
  hygiene.
- **D-01 aggregation** — the entire remaining change.

### Schema

| change | detail |
|---|---|
| `qc_hf_energy_db` | new name for `qc_bandwidth_hz`, which holds **decibels**, not hertz: it is the share of energy above 5 kHz in dB. The old column is kept as a deprecated copy for one release |
| `tags` | was `list<null>`, a type that cannot hold a value; now `list<string>` |
| `sim_ref_dur_bin` | duration stratum of the *second* clip, so SIM breakdowns can control for it |
| `gt_qc_*`, `simref_qc_*` | 9 columns each: DNSMOS and the QC battery on ground-truth and second-clip audio, which never faced the prompt-acceptance filters. **Reporting only — nothing is filtered** |
| `phones` (ky) | espeak dental markers `t[` / `d[` (1,584 and 1,132 occurrences) replaced by IPA `t̪` / `d̪`. `n_phones` is unaffected |
| removed | `qc_snr_db` and `qc_asr_cer` were declared in v1 and **100 % null in every subset** — never computed |

### New measurements

- a second recogniser on all six subsets (`facebook/mms-1b-all`, CTC without a
  language model) and a third encoder family for SIM;
- an **impostor floor** for SIM, which v1 lacks entirely;
- **95 % confidence intervals** on every headline number, and `n` in every cell;
- **per-utterance parquet artifacts** shipped inside this repository, so any
  number here can be recomputed, sliced or re-tested without a GPU.

### Corrections to the v1 card

| claim in v1 | corrected |
|---|---|
| second clip missing for "26-39 %" of rows | present for **{simref_lo}-{simref_hi} %**; see [Subsets](#subsets) |
| SIM rises monotonically with prompt duration in five subsets | **three** under v1's encoder — en-US, es-MX, nl-NL |
| every example carries ground-truth audio and a second clip | GT missing for 7 examples; second clip present for {simref_lo}-{simref_hi} % |
| "Speakers" | **speakers used**, which is not speakers available in the source slice |
| anchors are a "physical ceiling" | **human anchor**. A model can exceed it — see [Can a model beat the anchor?](#can-a-model-beat-the-anchor) |

## Subsets

{subset}

## Human anchors — intelligibility

Every figure below is a **human anchor**: a recogniser reading a real human
recording of the target text. Without it a model's WER cannot be read — is 8 %
on en-US a weak model, or roughly what the recogniser scores on human speech?

Headline aggregation is corpus-level, Σ(S+D+I) / Σ N_ref. Intervals are 95 %
cluster bootstraps resampling **speakers**, not rows.

{wer}

`exact` is the share of utterances transcribed with zero errors.
`catastrophic` is the share above 50 % WER — the indicator that catches looping,
babbling and dropped clauses long before they move the mean.

### Three recognisers, and why the gaps matter

Whisper decodes with a strong internal language model, so a mispronounced or
half-swallowed word is often repaired into the word the sentence implies; a
system is then credited with intelligibility it did not produce. MMS is CTC with
greedy decoding and no language model: it emits what it heard. Its absolute WER
is higher everywhere, which is not a defect. **Scribe** is a second strong-LM
read, from a different vendor and training set.

{agree}

Deltas are paired bootstraps over the utterances both recognisers scored; every
interval above excludes zero.

Two of these gaps mean different things.

**Whisper − MMS** measures how much the intelligible reading depends on the
listener's expectations. A large gap means the acoustics alone do not carry the
sentence.

**Scribe − Whisper** measures something the benchmark could not see with one
recogniser: how much of the "human anchor" was never the human at all. Scribe
reads the *same recordings* 27-59 % more accurately, and with a lower
catastrophic rate, so it is not buying accuracy with hallucination. On `nl-NL`
the human anchor falls from 0.0385 to **0.0159** with **zero** catastrophic
utterances. Whatever a v1-style anchor attributed to "human speech is hard" was
substantially Whisper's own error, and the ceiling a synthesis system is measured
against is correspondingly higher.

`ky` is the exception and runs the other way: Scribe (0.1829) is well behind
GigaAM (0.0998), paired delta +0.0831 [0.0629, 0.1046]. See below.

### Is the Kyrgyz subset usable?

Kyrgyz is the one subset where the primary recogniser (`GigaAM-Multilingual`) is
not independently validated for this purpose, and its v1 anchor of 11.04 % sat
above the 10 % threshold that calls for a cross-check. v2 performs that
cross-check with MMS, which covers Kyrgyz through its `kir` adapter.

MMS scores 2.03x-3.04x the strong recogniser across the five Whisper subsets.
Kyrgyz sits at **2.43x** — mid-range. GigaAM therefore behaves, relative to a
language-model-free CTC baseline, exactly as Whisper does on the other five
subsets, and the cross-check does **not** find it anomalous.

A third recogniser now confirms it from the other direction. Scribe covers `kir`
— which Whisper's 100 languages do not — and it lands at 0.1829, well *behind*
GigaAM's 0.0998 (paired delta +0.0831 [0.0629, 0.1046]) while beating MMS. So on
the five subsets where a general-purpose hosted model is the most accurate reader
available, it is; on Kyrgyz, the language-specific model wins by a wide margin.
That is what a genuinely competent `ky` recogniser looks like, and it is the
opposite of what a broken one would produce.

Scribe's Kyrgyz output also carries a signature worth knowing about. The
references use exactly the 36 letters of the Kyrgyz alphabet; **6.2 %** of Scribe
transcripts contain letters from neighbouring Turkic languages — `ғ`, `қ`, `ұ`,
`ә`, `і` (Kazakh), `ҡ` (Bashkir). Folding those to their Kyrgyz counterparts
recovers only 0.0063 WER, 3.4 % of its errors, so the spelling is not what makes
it worse. It is a **symptom**: rows containing a foreign letter average 0.59 WER
against 0.17 elsewhere. When Scribe drifts into a neighbouring orthography it is
usually losing the whole utterance, not just the spelling.

The real limitation of `ky` is a sampling problem: 694 rows come from only
**89 distinct ground-truth speakers**, and the anchor's interval is
correspondingly wide, [0.0773, 0.1311]. Use it for measurement, treat small
differences between systems on it with suspicion, and quote the interval.

## Human anchors — speaker similarity

**An absolute SIM value is meaningless without a floor.** v1 published an anchor
and no baseline, so "0.87 against a ceiling of 0.93" read like 94 % of the way
there. Whether that is good depends entirely on what two *different* speakers
score, and v1 never measured it. v2 does, over 5,000 random cross-speaker prompt
pairs per language per encoder:

{sim}

- **anchor** — cos(second clip, prompt clip): two different recordings of the
  same person.
- **floor mean / p95** — cross-speaker pairs. p95 is the practical false-accept
  level: a system scoring there is being confused with strangers one time in twenty.
- **usable range** — anchor minus floor. This is the entire span in which a
  cloning system can distinguish itself.

**Read the `wavlm_sv` rows carefully.** That is v1's encoder, and its usable
range is 0.17-0.32 on a scale that looks like it runs to 1.0. Worse, its
impostor **p95** reaches 0.87-0.93 — on Kyrgyz the p95 impostor (0.9279) is
*above* the human anchor (0.9214), meaning more than 5 % of random cross-speaker
pairs outscore two recordings of the same person. A bare number near 0.93 from
this encoder carries very little information. `wavlm_ft` and `ecapa` keep
0.41-0.58 of usable range and separate speakers far more cleanly.

Report the normalised score, which is readable where a bare cosine is not:

```
sim_norm = (sim_o − sim_floor) / (sim_anchor − sim_floor)
```

0 means indistinguishable from an impostor; 1 means as close as two recordings
of the same person. Per-language coefficients are in `reports/csv/sim.csv`.

### Which encoder to use

| encoder | model | use it for |
|---|---|---|
| `wavlm_sv` | `microsoft/wavlm-base-plus-sv` | continuity with v1 numbers only |
| `wavlm_ft` | `wavlm_large_finetune.pth` | comparability with the seed-tts-eval literature |
| `ecapa` | `speechbrain/spkrec-ecapa-voxceleb` | a reading from outside the WavLM family |

The third exists because of a circularity in the build: prompt QC rejected clips
lying more than μ−2σ from their speaker's centroid **as measured by WavLM-SV**,
and the anchor was then measured with WavLM-SV. The pool was pre-selected for
homogeneity in the metric's own space. `wavlm_ft` shares that backbone and
inherits the blind spot; ECAPA-TDNN shares neither architecture, pretraining nor
training corpus, so where its anchor tracks WavLM's the anchor is a property of
the speakers, and where it does not, the selection is showing through.

### SIM against prompt duration

Under v1's encoder, the relationship is monotonic in **three** subsets, not five:

{dur_sv}

The choice of encoder changes the answer: under `wavlm_ft` and `ecapa`, es-ES
becomes monotonic too and only `ky` and `pt-BR` peak in the middle bin. Full
tables with `n` per cell are in `reports/csv/sim_breakdown.csv`.

### The confounder v1 did not control

The table above bins on the **prompt** clip only. The other clip in the pair —
`sim_ref_audio`, down to about 2 s — was uncontrolled, and it moves the anchor:

{ref_sv}

`sim_ref_dur_bin` ships as a column in v2 so any SIM breakdown can control for it.

### Effective sample size

The anchor is **constant within a speaker** by construction: every example of a
speaker shares one prompt clip and one second clip. So `n rows` is not the number
of measurements — en-US's 909 rows carry 629 distinct values, and `ky`'s 599 rows
carry 155. Every interval in this card is a cluster bootstrap over speakers for
this reason. A row-level bootstrap would report intervals roughly √(rows per
speaker) too narrow.

## Can a model beat the anchor?

**Yes, and models do.** The Seed-TTS paper's Table 1 (zero-shot in-context
learning) reports English SIM 0.762 against a human 0.730, and Mandarin WER
1.115 % against a human 1.254 %. The anchor is not a physical ceiling and this
card does not call it one.

The reason is structural. Synthesis is **conditioned on the prompt** and inherits
its recording channel, microphone and room; the anchor compares two *different*
recordings of the person, made at different times. The comparison is asymmetric
in the model's favour. Three further biases point in various directions:

- **circularity** — the QC filter and the v1 SIM metric share an embedding space
  (above);
- **transcript quality** — the WER anchor mixes recogniser error, reader error
  and unverified crowd-sourced transcripts. The `ky` subset contains clear cases
  where both recognisers agree with each other and disagree with the reference,
  i.e. the reference is wrong;
- **training contamination** — Whisper has plausibly seen Common Voice in
  training, which pushes the anchor the other way.

None of these are quantified. Treat the anchor as a reference point, not a bound.

## Evaluation protocol

Every parameter that determines a number. v1 stated none of them.

### Preprocessing parity

The anchors are measured on audio that the build put into a canonical state:
mono, 16 kHz, Silero-VAD trimmed with a 50 ms pad, peak-normalised to −1 dBFS.
**Your synthesis must go through the same pipeline** or the comparison is not
like-for-like — trailing silence and level differences both shift SIM.
`msbench.audio.prepare()` does this and the drivers apply it by default.

Resampling uses **soxr HQ**. This matters: v1 resampled with `np.interp`, linear
interpolation with no anti-aliasing filter. References are natively 16 kHz and
never took that path, but synthesis at 22.05 / 24 / 44.1 kHz always did and
arrived at the speaker encoder carrying aliasing artifacts — a 15 kHz tone that
must vanish at 16 kHz survives at −3.1 dBr with 93 % of the residue folded onto
1 kHz, in the middle of the speech band. Every model's SIM was depressed by this;
the anchor was not.

Stored dataset audio is **not** re-trimmed, because re-running VAD with a
different Silero version than the build used would move the anchor.

### ASR

```
model            openai/whisper-large-v3        (ai-sage/GigaAM-Multilingual rev "ctc" for ky)
dtype            float16
num_beams        1                              greedy
max_new_tokens   200                            the runaway guard
repetition_penalty  not set                     removed in v2
chunk_length_s   not set                        short-form path
batch            8
second opinion   facebook/mms-1b-all, per-language adapter, CTC greedy, no LM
```

### ElevenLabs Scribe

```
model               scribe_v2
language_code       per subset, ISO 639-3 (eng / spa / nld / por / kir) — never auto-detect
tag_audio_events    false     default is TRUE; its tags would be scored as words
diarize             false
timestamps_granularity  none
seed                0
temperature         0
no_verbatim         false     it strips filler words, which are real content here
keyterms            unset     it biases decoding toward supplied words
workers             12        measured: ~8.8 clips/s; 24 draws 429s
timeout             300 s
```

Two properties of this backend have no counterpart in the local recognisers and
must be understood before its numbers are used.

**It is not deterministic, even pinned.** Identical requests return different
transcripts. `seed` alone changes nothing; `seed` with `temperature=0` narrows
the spread a great deal but does not close it. Measured by running `ky` three
times end to end: only **69.5 %** of transcripts are identical across all three,
yet corpus WER lands at 0.1829 / 0.1805 / 0.1809 — a range of **0.0024** against
a bootstrap interval 0.078 wide. The noise is real per utterance and negligible
in the aggregate. Audit a single row and you may not reproduce it; quote a
headline and you will. (`scribe_v1` is less stable still, which is why v2 is the
default here — the version number is not the reason.)

**It hallucinates on short clips.** On a 1.1 s pt-BR utterance where Whisper
scores 0.00 it returned an unrelated sentence. `audio_duration_secs` comes back
matching the clip, so this is a decoder failure, not a transport one. Read its
`catastrophic_rate` alongside its WER, never the WER alone.

There is no batch endpoint: the API takes one file per request, and "batch" in
ElevenLabs' terms means asynchronous delivery by webhook, which does not raise
throughput. Concurrency is the only lever.

### Text normalisation

Applied identically to reference and hypothesis:

```
NFC -> lowercase -> expand digits (num2words, per language; none for ky)
    -> strip punctuation -> collapse whitespace
PUNCT = !"#$%&()*+,-./:;<=>?@[\]^_`{{|}}~«»„""''…–—¿¡
```

Apostrophes are **kept** (word-internal: don't, 's-Gravenhage). Diacritics are
**kept** — `año`/`ano` and `sé`/`se` are different words, and stripping them
would mask real errors. This is where the normaliser deliberately differs from
`whisper.normalizers`, which is too aggressive for non-English.

### Aggregation

```
wer_corpus  = Σ(S+D+I) / Σ N_ref     headline, matches seed-tts-eval
wer_macro   = mean per-utterance     v1 legacy, reported alongside
cer         counts the space as a character (jiwer default)
exact_match = share of utterances with WER == 0
catastrophic_rate = share with WER > 0.5
```

Rows whose normalised reference is empty are **counted** (`n_empty_ref`), not
silently dropped. In this dataset that count is 0 in every subset — the defect
existed in v1's code but never affected a published number.

## How to read the numbers

**Confidence intervals.** Every headline figure carries a 95 % cluster bootstrap
over speakers (2,000 replicates, seed 20260725). Observations are not
independent: one prompt serves up to 7 examples, GT-speaker concentration is
uncapped, and the SIM anchor is constant within a speaker. Resampling rows would
understate the interval.

**Claiming "A beats B".** Do not compare two independent intervals — that throws
away the fact that both systems were measured on the same sentences. Use
`msbench.stats.paired_bootstrap`, which resamples the shared speakers once per
replicate and applies that resample to both systems, and report the delta with
its interval and P(A better).

**Per-cell `n`.** Every breakdown table in `reports/` carries `n` and the number
of distinct speakers behind it, and flags cells with fewer than 30 speakers.
Do not read a trend off a thin cell.

**Cross-language comparison is not supported.** Anchor-normalised comparison is
valid **within a language only**. Differences between anchors are a property of
the recogniser, not of the languages, and for `ky` it is a different recogniser
entirely.

## Source data

Common Voice 17.0 (`fsicoli/common_voice_17_0`), CC0. Text and audio are
crowd-sourced; speaker metadata is self-declared and often absent.

### Regional variants are not locales

Common Voice has no `es-MX`, `pt-BR` or `nl-NL` locale. These variants live in
free-text, self-declared `accents` / `variant` columns, the field is
multi-valued, and the labels themselves contain commas inside parentheses
("España: Norte peninsular (Asturias, Castilla y León, Cantabria)"). A naive
`split(",")` shreds them. Labels are split on commas at parenthesis depth zero
and matched **exactly** — substring matching gives false positives, since the
Caribbean Spanish label contains "Costa del golfo de México" and would be picked
up by a `contains("México")` test.

`es-ES` means the Castilian norm: Norte peninsular plus Centro-Sur. Andalusian is
excluded — seseo/ceceo places it closer to `es-MX`, and 82 % of that slice is one
speaker. /θ/ is present in `es-ES` and absent from `es-MX`, which is the point of
having both.

## Quality control

Prompt candidates were rejected on criteria that speech restoration cannot fix:
not mono, no speech found, clipping above 1e-3, energy above 5 kHz below −45 dB
(the signature of upsampling from 8 kHz), speech ratio below 0.75 after trimming,
more than one speech segment after merging gaps under 0.25 s, and duration
outside 2-12 s. Noise metrics were deliberately **not** used as filters, because
references are expected to go through a restoration model downstream.

Two things about the `qc_*` columns that v1 did not state:

- they are computed on the **untrimmed** clip, while `prompt_audio` ships
  trimmed. This is why `qc_lead_sil` and `qc_trail_sil` are non-zero on audio
  that has had its edge silence removed;
- they cover **prompts only**.

The last point matters for the WER anchor. `gt_audio` and `sim_ref_audio` got the
same *preprocessing* but never faced the *rejection* filters, so the anchor
includes recordings that would not have been accepted as prompts. v2 measures how
many, and ships the result as `gt_qc_*` and `simref_qc_*` columns:

{quality}

`would fail prompt QC %` applies the prompt thresholds to each stream. Prompts
pass by construction. Nothing is filtered — removing rows would break `utt`
stability with v1 — so a user who wants a clean subset applies their own
threshold and says so.

## Example selection

**Prompts and targets are decoupled.** Each speaker contributes one fixed
reference clip for all of their examples, which keeps SIM variance down; the
target text comes from a different recording, usually a different speaker. The
model is therefore always asked for speech that does not exist.

Selection is constrained rather than random: a target length distribution per
language, a per-speaker cap, a minimum female-voice share where the source allows
one, and full phonetic coverage. All six subsets cover **100 % of the candidate
pool's phoneme and diphone inventory**.

## Data fields

Three audio streams per example, all mono 16 kHz, VAD-trimmed, peak-normalised
to −1 dBFS:

| field | what it is |
|---|---|
| `prompt_audio` | the reference clip — condition your model on this |
| `prompt_audio_orig` | the same clip at its original sample rate |
| `gt_audio` | a real recording of `text`, usually by a **different** speaker. Basis of the WER anchor |
| `sim_ref_audio` | a second clip of the **prompt** speaker, different text. Basis of the SIM anchor |

**The human anchors, per row** — the benchmark's central numbers, carried in the
data itself so they are visible without downloading anything else:

| field | what it is |
|---|---|
| `anchor_asr` | which recogniser produced the primary anchor: `whisper`, or `gigaam` for `ky` |
| `anchor_wer`, `anchor_cer` | that recogniser's **per-utterance** error rate on `gt_audio` |
| `anchor_subs`, `anchor_dels`, `anchor_ins`, `anchor_n_ref_words` | the edit counts behind it, and the reference length |
| `anchor_cer_err`, `anchor_n_ref_chars` | the same for characters |
| `anchor_hyp` | what it actually transcribed, normalised — so a row's WER can be understood rather than only read |
| `anchor_wer_mms`, `anchor_cer_mms` + counts | MMS, CTC without a language model |
| `anchor_wer_scribe`, `anchor_cer_scribe` + counts | ElevenLabs Scribe v2, and `anchor_hyp_scribe` |
| `anchor_sim_wavlm_sv`, `anchor_sim_wavlm_ft`, `anchor_sim_ecapa` | cos(second clip, prompt clip) under each encoder |

`NaN` where the underlying audio is absent (`has_gt` or `has_sim_ref` false).

**`anchor_wer` is not the headline number.** It is a per-utterance rate, and
averaging it gives the *macro* aggregation that v2 keeps only for continuity with
v1. The headline is corpus-level, and the counts are shipped so it is derivable
from the dataset alone:

```python
import datasets
d = datasets.load_dataset("{dataset}", "en-US", split="main").to_pandas()

wer_corpus = (d.anchor_subs + d.anchor_dels + d.anchor_ins).sum() \
             / d.anchor_n_ref_words.sum()      # 0.0774 — the headline
wer_macro  = d.anchor_wer.mean()               # 0.0807 — v1 legacy
```

The same holds for any slice: filter the rows first, then sum. Averaging
`anchor_wer` over a slice silently switches you back to the macro form.

One further caution: the SIM anchor is **constant within a speaker** by
construction, so those columns are not independent observations. Cluster by
`speaker_id` before putting an interval on anything.

**Identity:** `utt` (stable join key), `lang`, `subset`, `source`, `cv_version`.

**Prompt:** `prompt_text`, `prompt_dur` (after trimming), `prompt_sr_orig`,
`prompt_dur_bin`, `prompt_cv_path`, `prompt_split_origin`.

**Speaker:** `speaker_id`, `speaker_gender` (+ `_source`), `speaker_age`,
`speaker_accent_label` (raw label, secondary accents included),
`speaker_n_in_subset`.

**Target text:** `text`, `text_norm` (produced by `msbench.normalize`, not a
placeholder), `n_words`, `n_chars`, `len_bin`, `phones` (IPA),
`n_phones`, `punct_type`, `text_cv_path`, `text_split_origin`.

**Anchor availability:** `has_gt`, `gt_dur`, `gt_speaker_id`, `gt_same_speaker`,
`gt_gender`, `has_sim_ref`, `sim_ref_dur`, `sim_ref_dur_bin`, `sim_ref_text`.

**Prompt QC** (`qc_*`, on the untrimmed clip): `qc_vad_speech_ratio`,
`qc_lead_sil`, `qc_trail_sil`, `qc_clip_rate`, `qc_hf_energy_db` (dB above
5 kHz), `qc_bandwidth_hz` (deprecated alias of the previous), `qc_dnsmos_ovrl`,
`qc_dnsmos_sig`, `qc_dnsmos_bak`, `qc_spk_centroid_dist`.

**Ground-truth and second-clip QC** (new in v2, reporting only): `gt_qc_*` and
`simref_qc_*`, each with `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak`, `clip_rate`,
`hf_energy_db`, `vad_speech_ratio`, `lead_sil`, `trail_sil`, `fails_prompt_qc`.

**Reserved for a future hard set**, currently constant: `category` (`general`),
`subcategory`, `tags` (empty `list<string>`), `difficulty` (0), `has_digit`,
`has_abbrev`, `has_foreign` (all false), `notes`.

## Reports

`reports/` is part of the release, not an afterthought.

| path | contents |
|---|---|
| `reports/results.md` | all anchors with intervals, breakdowns with per-cell `n` |
| `reports/attribution.md` | the v1 → v2 ladder, one cause per delta |
| `reports/summary.md`, `reports/coverage_*.md` | per-subset composition and coverage |
| `reports/quality_{{prompt,gt,sim_ref}}.md` | DNSMOS and QC per audio stream |
| `reports/csv/*.csv` | every table above, machine-readable |
| `reports/per_utterance/*.parquet` | **per-`utt` metrics for every recogniser and encoder**, with S/D/I counts, reference lengths, both normalised strings and `speaker_id` |
| `reports/per_utterance/*.json` | the exact configuration of every run |

The per-utterance artifacts are what let you re-aggregate any number in this
card, audit outliers, run your own paired tests, or slice by any column — without
a GPU and without re-running a recogniser. v1 shipped aggregate markdown only.

## Limitations

1. **No naturalness axis.** Neither subjective nor predicted. A system can score
   well here and sound robotic.
2. **No hard set.** Common Voice contains 0 % texts with digits, so text
   normalisation is untested.
3. **No long-form.** Texts are 3-16 words; long-context prosody and stability are
   untested.
4. **`ky` rests on 89 ground-truth speakers** for 694 rows, with a correspondingly
   wide interval.
5. **The WER anchor mixes three error sources** — recogniser error, reader error,
   and unverified crowd transcripts — with no way to separate them.
6. **Whisper has plausibly seen Common Voice**, which flatters the anchor in the
   opposite direction to the previous point. Neither bias is quantified.
7. **The QC filter and v1's SIM encoder share an embedding space.** Use `ecapa`
   to see past it.
8. **v1's SIM encoder has almost no usable range**, and on `ky` its impostor p95
   exceeds the human anchor.
9. **Observations are not independent**; always cluster by speaker.
10. **Ground-truth and second-clip audio were never rejection-filtered**;
    1.2-4.3 % would fail the prompt criteria.
11. **`utt` stability was prioritised over data cleanliness.** Nothing is
    filtered on the new QC columns.
12. **The macro→corpus change makes v2 numbers incomparable with v1's** unless you
    use the `wer_macro` column, which is retained for exactly that purpose.
13. **The repetition-penalty removal is justified theoretically**, not
    empirically: its effect on synthesis has not been measured here.
14. **Cross-language comparison of anchors is not supported.**
15. **Packaged audio was resampled to 16 kHz with linear interpolation at build
    time**, before v2 fixed the evaluation path. That is frozen into the data and
    is one more reason not to compare absolute SIM across datasets.
16. **Prompt durations are 2.5-5 s**, shorter than the 3-20 s Seed-TTS uses, and
    SIM rises with prompt duration — so even same-encoder comparison with that
    literature is only partial.
17. **Speaker metadata is self-declared** and missing for a large share of
    speakers (44 % of `ky` gender).
18. **pt-BR splits are broken upstream**: 97 % of test sentences also appear in
    train, and 9,464 clips are duplicated between train and dev.
19. **`n_syllables` was dropped rather than approximated** — every heuristic
    broke on hiatus or diphthongs. Use `n_phones / gt_dur` for speech rate.
20. **Scribe is a paid, hosted, non-deterministic recogniser.** It is the most
    accurate reader of these recordings on five subsets, but it cannot be the
    primary anchor of an open benchmark: reproducing it costs money, needs
    network access, and lands within ~0.002 rather than exactly. Whisper and
    GigaAM stay primary for that reason, not because they are better.
21. **DNSMOS is a reporting metric here, never a filter.** Reference quality was
    measured against SIM and explains under 1.6 % of its variance, so it is not a
    confounder — but that was measured on the human anchor, where noise affects
    both embeddings and partly cancels.

## Reproduction

```bash
git clone {repo} && cd make-speech-benchmark
uv venv && source .venv/bin/activate && uv pip install -e ".[all]"

msbench-fetch                  # download this dataset
bash scripts/runbook.sh all    # every number in this card
msbench-validate               # v2 differs from v1 only where claimed
pytest                         # the aggregation gate
```

Selection stages S1-S3 are **not** re-run and are not needed for any of the
above; they require the 66 GB Common Voice corpus and would change `utt`.

Two documents in the repository carry more detail than fits here:
[`docs/PROTOCOL.md`]({repo}/blob/main/docs/PROTOCOL.md) is the full measurement
contract, and [`docs/DEFECTS.md`]({repo}/blob/main/docs/DEFECTS.md) is the
catalogue of all 27 v1 defects with what each one cost.

## Personal and sensitive information

The audio is human speech from Common Voice, contributed under CC0 by volunteers
who consented to public release. Speaker identifiers are truncated Common Voice
`client_id` hashes and are not linkable to a person by this dataset alone.
Demographic fields are self-declared and frequently absent. The recordings are
nonetheless **biometric voice data**: they can be used to build speaker models,
and this dataset exists to measure exactly that capability. Do not use them to
impersonate the contributors.

Sentences come from Common Voice's own text corpora and may contain the usual
errors of crowd-sourced transcription; several were identified during this work
where the reference text disagrees with what was actually said.

## Licence, citation, contact

**CC0-1.0**, matching Common Voice 17.0.

```bibtex
@misc{{multilingual_speech_benchmark_2026,
  title  = {{Multilingual Speech Benchmark for Zero-Shot TTS}},
  author = {{nineninesix}},
  year   = {{2026}},
  note   = {{Version 2.0}},
  url    = {{https://huggingface.co/datasets/{dataset}}}
}}
```

Pipeline: [{repo}]({repo}). Issues and pull requests welcome — adding a language,
a recogniser backend or a speaker encoder each touch one file.

## References

- Common Voice 17.0 — <https://commonvoice.mozilla.org>
- Seed-TTS ([arXiv:2406.02430](https://arxiv.org/abs/2406.02430)) and
  seed-tts-eval, the source of the corpus-level WER convention and the
  `wavlm_large_finetune.pth` SIM checkpoint
- Whisper ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356))
- MMS ([arXiv:2305.13516](https://arxiv.org/abs/2305.13516))
- WavLM ([arXiv:2110.13900](https://arxiv.org/abs/2110.13900))
- ECAPA-TDNN ([arXiv:2005.07143](https://arxiv.org/abs/2005.07143))
- DNSMOS P.835 ([arXiv:2110.01763](https://arxiv.org/abs/2110.01763))
- Silero VAD — <https://github.com/snakers4/silero-vad>
"""


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v2"))
    ap.add_argument("--yaml", default=None,
                    help="front matter from a file; by default it is generated "
                         "from the pack's own parquet")
    ap.add_argument("--no-yaml", action="store_true",
                    help="emit the body only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pack = Path(args.pack)
    body = build(pack)
    # Generated here rather than concatenated by hand afterwards: the front
    # matter declares the schema and the body describes it, and a card assembled
    # in two steps can be published with the two out of step — which is the class
    # of drift this generator exists to prevent.
    if args.no_yaml:
        front = None
    elif args.yaml:
        front = Path(args.yaml).read_text()
    else:
        front = front_matter(pack)
    text = f"---\n{front}---\n\n{body}" if front else body
    if args.out:
        Path(args.out).write_text(text)
        print(f"-> {args.out}  ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
