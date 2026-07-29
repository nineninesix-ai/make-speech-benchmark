# Coverage: es-MX

**Examples:** 1500 (target 1500)  
**Source:** Common Voice 17.0, espeak voice `es-419`

## Phonetics

* phonemes: **36** (100% of the candidate pool, verified at build time)
* diphones: **785** (100% of the candidate pool, verified at build time)
* no uncovered diphones

## Target text length

| bin | selected | target | delta |
|---|---|---|---|
| 3-5 | 225 | 225 | +0 |
| 6-9 | 450 | 450 | +0 |
| 10-12 | 450 | 450 | +0 |
| 13-16 | 375 | 375 | +0 |
| 17+ | 0 | 0 | +0 |

words: median 10, p10-p90 5-14

## Speakers

* unique speakers **used**: **929** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **2** (cap 2), median 2
* gender (by example): male 835 (56%), female 562 (37%), unknown 103 (7%)
* age, top 5: twenties 795, thirties 267, fourties 135, teens 135, unknown 88

## Reference audio

* duration after VAD trimming: median **4.38 s**, p05-p95 2.81-5.70
* strata: {'>=4.5': 654, '3.7-4.5': 503, '<3.7': 343}
* original sample rates: [32000, 44100, 48000]
* second clip (`sim_ref_audio`) duration: median **2.88 s**, p05-p95 2.08-4.45

## Human anchors and provenance

* with GT audio (WER anchor): **1500/1500**
* with sim_ref (SIM anchor): **1201/1500** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 1
* held-out (dev/test): texts **73%**, prompts **51%**
* punctuation: {'declarative': 1346, 'question': 76, 'exclamation': 75, 'mixed': 3}
* unique texts: 1500/1500
