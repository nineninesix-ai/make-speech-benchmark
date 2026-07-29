# S2 pilot — results

**Date:** 2026-07-25
**Sample:** 40 speakers x up to 8 clips per language = 320 clips x 6 languages
= 1,920 clips. Speakers with >=3 clips of 3-8 s inside the target slice.
**Runtime:** ~30 s per language on an RTX 4080 SUPER.
**Raw metrics:** `reports/pilot_qc.parquet`.

## QC scope

Deliberately narrowed: the selected references are meant to be passed through an
external speech-restoration model after the benchmark is assembled. Noise
metrics (DNSMOS, WADA-SNR) and ASR verification of prompts are therefore not
used for filtering, and hypotheses **H1 and H2 are closed as inapplicable**. What
is checked is what restoration cannot fix: decode sanity, VAD trimming and
speaker consistency.

## Results

| language | clips passed | **speakers surviving** | H3 median cos | H3 p05 |
|---|---|---|---|---|
| es-MX | 263/320 (82%) | **40/40 (100%)** | 0.981 | 0.952 |
| pt-BR | 208/320 (65%) | **39/40 (98%)** | 0.969 | 0.889 |
| nl-NL | 268/320 (84%) | **39/40 (98%)** | 0.977 | 0.934 |
| en-US | 293/320 (92%) | **40/40 (100%)** | 0.979 | 0.941 |
| es-ES | 281/320 (88%) | **40/40 (100%)** | 0.981 | 0.938 |
| ky | 257/320 (80%) | **39/40 (98%)** | 0.979 | 0.942 |

### Headline: the capacity risk is cleared

The estimate in `build/schema.py` assumed 75% speaker survival, at which point
pt-BR, nl-NL and ky would fail to reach N. **Measured survival is 98-100%.**
Speaker survival, not clip survival, is the relevant quantity: each speaker
contributes **one fixed prompt** to all of their examples, so a single usable
clip is enough to supply all `cap` of them.

The S3 ceilings hold; no cap needs raising on this evidence. (pt-BR was later
raised from 6 to 7 for a different reason — only 247 of 287 speakers received a
prompt in the full run, and 247 x 6 < 1500.)

### Rejection reasons

| reason | es-MX | pt-BR | nl-NL | en-US | es-ES | ky |
|---|---|---|---|---|---|---|
| recording break (pause > 0.25 s) | 45 | 31 | 17 | 20 | 23 | 45 |
| duration outside 2-12 s | 5 | **80** | 18 | 7 | 10 | 7 |
| narrowband (upsampled from 8 kHz) | 8 | — | 17 | — | 7 | 14 |
| little speech after trimming | 1 | — | — | — | — | 3 |

* **pt-BR is the outlier** — 80 clips fall below 2 s after trimming. Expected:
  its median duration is 3.64 s against 4.4-5.3 s elsewhere, and trimming pushes
  the tail below the threshold. Not a data defect, just a shorter corpus; it does
  not endanger a set of 1,500 (ceiling 1,920).
* **"Recording break" is the most common reason everywhere.** A clip with an
  internal pause > 0.25 s makes a worse cloning reference: the pause eats into
  the usable signal.

### Distributions (p05 / median / p95)

| language | speech_ratio | trimmed off the edges, s | hf_db, dB | duration after trimming, s |
|---|---|---|---|---|
| es-MX | 0.88 / 0.97 / 0.98 | 0.80 + 0.63 | −40 / −24 / −9 | 2.40 / 4.13 / 5.92 |
| pt-BR | 0.85 / 0.95 / 0.98 | 0.83 + 0.96 | −39 / −22 / −6 | 1.12 / 2.83 / 5.12 |
| nl-NL | 0.90 / 0.97 / 0.98 | 0.77 + 0.89 | −47 / −25 / −11 | 1.98 / 3.60 / 5.34 |
| en-US | 0.91 / 0.97 / 0.98 | 0.75 + 0.82 | −32 / −20 / −8 | 2.21 / 3.94 / 5.86 |
| es-ES | 0.91 / 0.97 / 0.98 | 0.77 + 0.92 | −42 / −25 / −10 | 2.17 / 3.98 / 5.89 |
| ky | 0.87 / 0.96 / 0.98 | 0.80 + 0.72 | −42 / −23 / −11 | 2.14 / 3.58 / 5.41 |

**VAD trimming removes ~1.6 s of ~5.2 s — nearly a third of the clip.** Hence
the S3 consequence: the `prompt_dur_bin` stratum must be computed **after**
trimming, or the "6-8 s" bin ends up half empty. (The bin boundaries were later
redefined entirely, to the terciles of the post-trim distribution.)

Sample rates in the corpus are not uniform: **32 000, 44 100 and 48 000 Hz**,
contrary to the early assumption that "Common Voice is 32 kHz". The
`prompt_sr_orig` column is therefore mandatory.

## H3 — `client_id` is not necessarily one person

Cosine to the speaker centroid: median 0.97-0.98 across all languages, p05 no
lower than 0.889. **There is no systematic problem** — the hypothesis does not
hold in its original form. Individual outliers do exist and are worth cutting:
the worst speakers score 0.71 (pt-BR), 0.75 (es-ES), 0.83 (nl-NL). The
within-speaker μ−2σ threshold stays as it is; it affects a handful of clips, not
the population.

## Three threshold bugs the pilot exposed

The first run rejected **100%** of the sample. All three causes were defects in
the metrics, not in the data:

1. **Edge silence and `speech_ratio` were measured before trimming.** The VAD
   removes that silence — it is not a property of the clip. A median leading
   silence of 0.8 s against a 0.3 s threshold rejected 148 clips out of 150.
2. **Bandwidth was taken as an energy percentile (99.5%).** Almost all speech
   energy sits below 4 kHz, so "narrowband" fired on perfectly good clips —
   95 of 150. Replaced by the share of energy above 5 kHz; that distribution is
   unimodal with a median near −24 dB, and genuine upsampling is the tail below
   −45 dB, some 1-2% of clips.
3. **A pause between words counted as a recording break.** Segments are now
   merged when the gap is under 0.25 s.

The lesson carried forward: thresholds are read off a histogram during the
pilot, never assigned in advance — exactly what hypothesis H2 prescribed for
noise before it was dropped.
