# Coverage: en-US

**Examples:** 1500 (target 1500)  
**Source:** Common Voice 17.0, espeak voice `en-us`

## Phonetics

* phonemes: **49** (100% of the candidate pool, verified at build time)
* diphones: **1489** (100% of the candidate pool, verified at build time)
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

* unique speakers **used**: **1162** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **2** (cap 2), median 1
* gender (by example): female 676 (45%), male 442 (29%), unknown 382 (25%)
* age, top 5: twenties 425, unknown 380, thirties 219, teens 190, fourties 105

## Reference audio

* duration after VAD trimming: median **4.26 s**, p05-p95 2.43-5.63
* strata: {'>=4.5': 623, '<3.7': 456, '3.7-4.5': 421}
* original sample rates: [32000, 44100, 48000]
* second clip (`sim_ref_audio`) duration: median **3.01 s**, p05-p95 2.08-4.67

## Human anchors and provenance

* with GT audio (WER anchor): **1500/1500**
* with sim_ref (SIM anchor): **909/1500** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 0
* held-out (dev/test): texts **69%**, prompts **59%**
* punctuation: {'declarative': 1342, 'question': 108, 'exclamation': 50}
* unique texts: 1500/1500
