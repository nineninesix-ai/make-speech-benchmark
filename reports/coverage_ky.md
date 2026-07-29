# Coverage: ky

**Examples:** 700 (target 700)  
**Source:** Common Voice 17.0, espeak voice `ky`

## Phonetics

* phonemes: **32** (100% of the candidate pool, verified at build time)
* diphones: **741** (100% of the candidate pool, verified at build time)
* no uncovered diphones
* dental consonants are written `t̪` / `d̪`; v1 shipped the espeak markers `t[` / `d[` (D-22)

## Target text length

| bin | selected | target | delta |
|---|---|---|---|
| 3-5 | 245 | 245 | +0 |
| 6-9 | 385 | 385 | +0 |
| 10-12 | 70 | 70 | +0 |
| 13-16 | 0 | 0 | +0 |
| 17+ | 0 | 0 | +0 |

words: median 7, p10-p90 4-9

## Speakers

* unique speakers **used**: **181** (this is speakers used, not speakers available in the source slice)
* examples per speaker: max **4** (cap 4), median 4
* gender (by example): unknown 305 (44%), male 275 (39%), female 120 (17%)
* age, top 5: unknown 287, twenties 275, teens 72, thirties 51, fourties 11

## Reference audio

* duration after VAD trimming: median **3.68 s**, p05-p95 2.49-4.99
* strata: {'<3.7': 353, '3.7-4.5': 232, '>=4.5': 115}
* original sample rates: [32000, 48000]
* second clip (`sim_ref_audio`) duration: median **2.55 s**, p05-p95 2.05-3.98

## Human anchors and provenance

* with GT audio (WER anchor): **694/700**
* with sim_ref (SIM anchor): **599/700** — the rest had no second usable clip from that speaker
* GT from the same speaker as the prompt: 6
* held-out (dev/test): texts **96%**, prompts **83%**
* punctuation: {'declarative': 630, 'question': 65, 'exclamation': 5}
* unique texts: 700/700
