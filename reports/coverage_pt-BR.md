# Coverage: pt-BR

**Examples:** 1500 (target 1500)  
**Source:** Common Voice 17.0, espeak voice `pt-br`

## Phonetics

* phonemes: **43** (100% of the candidate pool, verified at build time)
* diphones: **1080** (100% of the candidate pool, verified at build time)
* no uncovered diphones

## Target text length

| bin | selected | target | delta |
|---|---|---|---|
| 3-5 | 525 | 525 | +0 |
| 6-9 | 525 | 525 | +0 |
| 10-12 | 300 | 300 | +0 |
| 13-16 | 150 | 150 | +0 |
| 17+ | 0 | 0 | +0 |

words: median 7, p10-p90 3-12

## Speakers

* unique speakers **used**: **247** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **7** (cap 7), median 6
* gender (by example): male 774 (52%), unknown 594 (40%), female 132 (9%)
* age, top 5: unknown 576, twenties 366, thirties 267, fourties 150, teens 80

## Reference audio

* duration after VAD trimming: median **3.46 s**, p05-p95 2.21-5.25
* strata: {'<3.7': 873, '3.7-4.5': 322, '>=4.5': 305}
* original sample rates: [32000, 48000]
* second clip (`sim_ref_audio`) duration: median **2.50 s**, p05-p95 2.05-4.29

## Human anchors and provenance

* with GT audio (WER anchor): **1500/1500**
* with sim_ref (SIM anchor): **983/1500** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 4
* held-out (dev/test): texts **13%**, prompts **58%**
* punctuation: {'declarative': 1317, 'question': 144, 'exclamation': 38, 'mixed': 1}
* unique texts: 1500/1500
