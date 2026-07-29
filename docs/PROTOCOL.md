# Evaluation protocol

Every parameter that determines a number. If you intend to compare a synthesis
run against the published anchors, this document is the contract — a number
produced any other way is not comparable to them, however reasonable the method.

v1 stated none of this, which is why v2 exists.

---

## The short version

```python
from msbench.audio import prepare
from msbench.normalize import normalize
from msbench.metrics import score_pairs
```

Those three functions are the contract. `prepare` puts audio into the state the
anchors were measured in, `normalize` puts text into the state they were scored
in, and `score_pairs` aggregates the way seed-tts-eval does. Reimplementing any
of them from the prose below will produce numbers that are close but not
comparable, and the difference will be invisible.

---

## 1. Preprocessing parity

The anchors are measured on audio the build put into a canonical state:

```
mono -> 16 kHz (soxr HQ) -> Silero VAD trim, 50 ms pad -> peak-normalise to -1 dBFS
```

**Your synthesis must go through the same pipeline** or the comparison is not
like-for-like. Trailing silence and level differences both shift SIM, and
neither is visible in a spectrogram at a glance. `msbench.audio.prepare()` does
this, and every driver applies it by default.

**Resampling must be band-limited.** v1 used `np.interp` — linear interpolation
with no anti-aliasing filter. References are natively 16 kHz and never took that
path, but synthesis at 22.05 / 24 / 44.1 kHz always did and arrived at the
speaker encoder carrying aliasing artifacts: a 15 kHz tone that must vanish at
16 kHz survived at −3.1 dBr, with 93 % of the residue folded onto 1 kHz — the
middle of the speech band. Every model's SIM was depressed by this; the anchor
was not. See [DEFECTS.md](DEFECTS.md#d-03--resampling-by-npinterp).

**Stored dataset audio is not re-trimmed.** Re-running VAD with a different
Silero version than the build used would move the anchor. `read_cell(...,
stored=True)` decodes without re-trimming; `prepare()` is for your synthesis.

---

## 2. Speech recognition

### Local recognisers

```
model            openai/whisper-large-v3     (ai-sage/GigaAM-Multilingual rev "ctc" for ky)
dtype            float16
num_beams        1                           greedy
max_new_tokens   200                         the runaway guard
repetition_penalty  not set                  removed in v2 (D-02)
chunk_length_s   not set                     short-form path (D-10)
batch            8
second opinion   facebook/mms-1b-all, per-language adapter, CTC greedy, no LM
```

`batch=8` throughout, including in the replay ladder, because Whisper's output
depends on batch composition through padding.

### ElevenLabs Scribe

```
model               scribe_v2
language_code       per subset, ISO 639-3 (eng / spa / nld / por / kir) — never auto-detect
tag_audio_events    false     the default is TRUE; its tags would be scored as words
diarize             false
timestamps_granularity  none
seed                0
temperature         0
no_verbatim         false     it strips filler words, which are real content here
keyterms            unset     it biases decoding toward supplied words
workers             12        measured ~8.8 clips/s; 24 draws 429s
timeout             300 s
```

Two properties have no counterpart in the local recognisers.

**It is not deterministic, even pinned.** Identical requests return different
transcripts. `seed` alone changes nothing; `seed` with `temperature=0` narrows
the spread but does not close it. Measured by running `ky` three times end to
end: **69.5 %** of transcripts identical across all three, yet corpus WER lands
at 0.1829 / 0.1805 / 0.1809 — a range of **0.0024** against a bootstrap interval
0.078 wide. The noise is real per utterance and negligible in aggregate. Audit a
single row and you may not reproduce it; quote a headline and you will.
(`scribe_v1` is less stable still. The version number is not the reason v2 is
the default here.)

**It hallucinates on short clips.** On a 1.1 s pt-BR utterance where Whisper
scores 0.00 it returned an unrelated sentence, with `audio_duration_secs`
matching the clip — a decoder failure, not a transport one. Read its
`catastrophic_rate` alongside its WER, never the WER alone.

There is no batch endpoint. The API takes one file per request; "batch" in
ElevenLabs' terms means asynchronous delivery by webhook, which does not raise
throughput. Concurrency is the only lever.

### Why three recognisers

A single ASR conflates the model's errors with the recogniser's idiosyncrasies.
Whisper's internal language model repairs mispronounced synthesis; MMS is CTC
with no LM and repairs nothing; Scribe is a second strong-LM system trained
independently. The gap between them is information, not noise — a model that
scores well on Whisper and badly on MMS is mispronouncing words that a language
model can guess back.

`ky` has no Whisper coverage (Whisper's 100 languages exclude Kyrgyz), so
GigaAM-Multilingual is the primary there. That makes cross-language comparison
invalid — see §6.

---

## 3. Text normalisation

Applied identically to reference and hypothesis:

```
NFC -> lowercase -> expand digits (num2words, per language; none for ky)
    -> strip punctuation -> collapse whitespace

PUNCT = !"#$%&()*+,-./:;<=>?@[\]^_`{|}~«»„""''…–—¿¡
```

Apostrophes are **kept** (word-internal: don't, 's-Gravenhage). Diacritics are
**kept** — `año`/`ano` and `sé`/`se` are different words, and stripping them
would mask real errors. This is where the normaliser deliberately differs from
`whisper.normalizers`, which is too aggressive for non-English.

There is no num2words locale for Kyrgyz, so digits pass through unexpanded in
`ky`. Both sides get the same treatment, so it is fair, but a model that spells
out a numeral will be penalised there.

---

## 4. Speaker similarity

Three encoders, because the choice changes the scale by more than most model
differences:

| key | checkpoint | scale | why it is here |
|---|---|---|---|
| `wavlm_sv` | `microsoft/wavlm-base-plus-sv` | ≈ 0.92–0.95 | v1's encoder, kept for continuity |
| `wavlm_ft` | `wavlm_large_finetune.pth` (UniSpeech) | ≈ 0.70–0.76 | what the stock seed-tts-eval scripts use |
| `ecapa` | `speechbrain/spkrec-ecapa-voxceleb` | — | outside the WavLM family; breaks the QC circularity (D-17) |

**Do not compare a `wavlm_ft` score against a `wavlm_sv` anchor.** They are
different scales, and this was one of v1's defects (D-16).

### The impostor floor

A raw cosine of 0.9 means nothing without knowing what two *different* speakers
score. For each language and encoder, 5,000 cross-speaker pairs give a floor,
and the anchor-normalised score is

```
sim_norm = (sim_observed - floor) / (anchor - floor)
```

so 0 is "indistinguishable from a stranger" and 1 is "as similar as two
recordings of the same person". Report `sim_norm` when comparing across
encoders; report the raw cosine when comparing against published literature.

---

## 5. Aggregation

```
wer_corpus  = Σ(S+D+I) / Σ N_ref     headline, matches seed-tts-eval
wer_macro   = mean per-utterance     v1 legacy, reported alongside
cer         counts the space as a character (jiwer default)
exact_match = share of utterances with WER == 0
catastrophic_rate = share with WER > 0.5
```

Rows whose normalised reference is empty are **counted** (`n_empty_ref`), never
silently dropped (D-04). In this dataset that count is 0 in every subset.

---

## 6. How to read the numbers

**Confidence intervals.** Every headline carries a 95 % cluster bootstrap over
speakers (2,000 replicates, seed 20260725). Observations are not independent:
one prompt serves up to 7 examples, GT-speaker concentration is uncapped, and
the SIM anchor is constant within a speaker. Resampling rows would understate
the interval — pt-BR's SIM anchor has 983 rows but 161 distinct values.

**Claiming "A beats B".** Do not compare two independent intervals; that throws
away the fact that both systems were measured on the same sentences. Use
`msbench.stats.paired_bootstrap`, which resamples the shared speakers once per
replicate and applies that resample to both systems, and report the delta with
its interval and P(A better).

**Per-cell `n`.** Every breakdown table in `reports/` carries `n` and the number
of distinct speakers behind it, and flags cells with fewer than 30 speakers. Do
not read a trend off a thin cell.

**Cross-language comparison is not supported.** Anchor-normalised comparison is
valid **within a language only**. Differences between anchors are a property of
the recogniser, and for `ky` it is a different recogniser entirely.

**The anchor is not a ceiling.** Models can exceed it. Synthesis is conditioned
on the prompt and inherits its recording channel; the anchor compares two
different recordings of the same person. The asymmetry favours the model.

---

## 7. Environment

Reproducibility has a version dimension. The Silero VAD version affects
trimming, and therefore the anchor; `jiwer` computes the edit distances; the
`transformers` version has already broken one Kyrgyz tokenizer in this
project's lifetime.

`uv.lock` pins a resolvable set. The versions present when v2.0 was packaged:

```
python 3.12          torch 2.12.0+cu130   torchaudio 2.11.0
transformers 5.14.1  numpy 2.4.6          pandas 3.0.5
jiwer 4.0.0          rapidfuzz 3.14.5     silero-vad 6.2.1
soxr 1.1.0           soundfile 0.14.0     onnxruntime 1.28.0
s3prl 0.4.18         speechbrain 1.1.0    datasets 5.0.1
```

**Honest caveat:** `jiwer`, `pandas`, `silero-vad` and `torchaudio` were
reinstalled during the packaging work that produced this repository layout, so
the list above is the environment *as packaged*, not provably the one that
computed the published anchors. The aggregation is pinned independently of the
library set by `tests/test_metrics.py`, which asserts the arithmetic against
hand-counted alignments and against v1's literal `jiwer.wer` call. The way to
confirm an anchor is to re-run it, not to trust this list.

---

## 8. What this protocol does not cover

- **Prosody, naturalness, expressiveness.** A model can hit every phoneme and
  match the speaker embedding and still sound robotic; nothing here notices.
- **Whether `repetition_penalty` masked looping in v1's model numbers.** That is
  a claim about synthesis, and this repository contains no synthesis run (D-02).
- **Long-form synthesis.** Targets are single utterances with a median of 6–7
  words.
- **Whether the recogniser has seen the source corpus.** Whisper plausibly has
  seen Common Voice; the anchor is optimistic by an unquantified amount (D-18).
