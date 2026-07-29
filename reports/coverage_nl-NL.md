# Coverage: nl-NL

**Examples:** 1500 (target 1500)  
**Source:** Common Voice 17.0, espeak voice `nl`

## Phonetics

* phonemes: **44** (100% of the candidate pool, verified at build time)
* diphones: **1083** (100% of the candidate pool, verified at build time)
* no uncovered diphones

## Target text length

| bin | selected | target | delta |
|---|---|---|---|
| 3-5 | 225 | 225 | +0 |
| 6-9 | 450 | 450 | +0 |
| 10-12 | 450 | 450 | +0 |
| 13-16 | 345 | 345 | +0 |
| 17+ | 30 | 30 | +0 |

words: median 10, p10-p90 5-14

## Speakers

* unique speakers **used**: **469** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **4** (cap 4), median 3
* gender (by example): male 994 (66%), female 272 (18%), unknown 234 (16%)
* age, top 5: twenties 556, thirties 307, unknown 223, fourties 176, fifties 117

## Reference audio

* duration after VAD trimming: median **3.52 s**, p05-p95 2.46-5.12
* strata: {'<3.7': 862, '3.7-4.5': 421, '>=4.5': 217}
* original sample rates: [32000, 44100, 48000]
* second clip (`sim_ref_audio`) duration: median **2.46 s**, p05-p95 2.05-3.62

## Human anchors and provenance

* with GT audio (WER anchor): **1500/1500**
* with sim_ref (SIM anchor): **1319/1500** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 3
* held-out (dev/test): texts **94%**, prompts **95%**
* punctuation: {'declarative': 1256, 'question': 210, 'exclamation': 34}
* unique texts: 1500/1500
