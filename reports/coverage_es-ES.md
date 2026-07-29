# Coverage: es-ES

**Examples:** 1500 (target 1500)  
**Source:** Common Voice 17.0, espeak voice `es`

## Phonetics

* phonemes: **37** (100% of the candidate pool, verified at build time)
* diphones: **808** (100% of the candidate pool, verified at build time)
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

* unique speakers **used**: **722** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **3** (cap 3), median 2
* gender (by example): male 815 (54%), female 570 (38%), unknown 115 (8%)
* age, top 5: twenties 438, thirties 327, fourties 308, fifties 160, teens 102

## Reference audio

* duration after VAD trimming: median **4.22 s**, p05-p95 2.69-5.60
* strata: {'3.7-4.5': 562, '>=4.5': 538, '<3.7': 400}
* original sample rates: [32000, 44100, 48000]
* second clip (`sim_ref_audio`) duration: median **2.87 s**, p05-p95 2.08-4.29

## Human anchors and provenance

* with GT audio (WER anchor): **1499/1500**
* with sim_ref (SIM anchor): **1222/1500** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 2
* held-out (dev/test): texts **57%**, prompts **44%**
* punctuation: {'declarative': 1342, 'exclamation': 95, 'question': 62, 'mixed': 1}
* unique texts: 1500/1500
