# make-speech-benchmark

Build and evaluation pipeline for a **multilingual zero-shot TTS (voice cloning)
benchmark** derived from Common Voice, following the
[seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) protocol and
extending it with controlled length distributions, guaranteed phonetic coverage,
and **human anchors for every metric**.

**Dataset:** <https://huggingface.co/datasets/nineninesix/multilingual-speech-benchmark>

> ### Status: working prototype
>
> This was built to evaluate [nineninesix/gepard-1.0](https://huggingface.co/nineninesix/gepard-1.0)
> and models like it. It is a working instrument, not a settled standard: it is
> going to be reviewed with full scientific rigour, corrected where that review
> finds it wanting, and extended.
>
> It is published in that state deliberately. Version 1 of this benchmark was
> published with 27 defects, [all of them catalogued](docs/DEFECTS.md) — including
> a headline aggregation that did not match the protocol it claimed compatibility
> with. That list is the reason to trust version 2, not a reason to distrust it.
> Numbers here carry confidence intervals, every anchor is re-derivable from the
> shipped data, and where something is unresolved this repository says so.
>
> Read [docs/DEFECTS.md](docs/DEFECTS.md) before citing anything.

---

## Why human anchors

A WER of 8 % means nothing until you know what the same recogniser scores on
*real human speech* reading the same sentences.

| subset | examples | speakers used | WER anchor (corpus) | 95 % CI | recogniser |
|---|---|---|---|---|---|
| `en-US` | 1,500 | 1,162 | **0.0774** | [0.0703, 0.0842] | whisper-large-v3 |
| `es-ES` | 1,500 | 722 | **0.0452** | [0.0397, 0.0512] | whisper-large-v3 |
| `es-MX` | 1,500 | 929 | **0.0613** | [0.0541, 0.0691] | whisper-large-v3 |
| `nl-NL` | 1,500 | 469 | **0.0385** | [0.0331, 0.0453] | whisper-large-v3 |
| `pt-BR` | 1,500 | 247 | **0.0737** | [0.0608, 0.0884] | whisper-large-v3 |
| `ky` | 700 | 181 | **0.0998** | [0.0773, 0.1311] | GigaAM-Multilingual |

A model at 8 % WER is near-human in Dutch and twice as bad as human in US
English — the same number, two different verdicts. Likewise for similarity: two
recordings of the *same living person* score 0.92–0.95 under `wavlm_sv`, so a
model at 0.90 is close to the metric's ceiling, not mediocre.

All six subsets cover 100 % of their candidate pool's phoneme and diphone
inventory.

**These anchors are not comparable across languages.** The differences are a
property of the recogniser, and `ky` uses a different recogniser entirely. See
[docs/PROTOCOL.md](docs/PROTOCOL.md#6-how-to-read-the-numbers).

---

## Install

```bash
git clone https://github.com/nineninesix-ai/make-speech-benchmark
cd make-speech-benchmark
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"        # or pick extras: asr, ky, sim, quality, build

# System dependency, not installable from PyPI (only needed for the `build` extra):
apt-get install -y espeak-ng      # phonemizer backend, needs the ky voice
```

Requires Python ≥ 3.11. `uv.lock` pins a resolvable set; see
[PROTOCOL.md §7](docs/PROTOCOL.md#7-environment) for what versions actually
produced the published numbers.

A full corpus rebuild needs ~150 GB of disk (66 GB of tar shards plus working
files) and a 16 GB GPU. Evaluation alone needs far less.

## Evaluate your own model

```bash
export MSBENCH_HOME=$PWD          # where the data lives
msbench-fetch                     # download the benchmark (~4 GB)

# synthesise into out/<model>/<lang>/<utt>.wav, then:
msbench-wer --lang en-US --audio synth --asr whisper \
            --synth-dir out/mymodel/en-US --model-name mymodel
msbench-sim --lang en-US --mode synth --encoder wavlm_ft \
            --synth-dir out/mymodel/en-US
msbench-report --all
```

**Before you do:** your synthesis has to go through the same audio path the
anchors did, or the comparison is not like-for-like.

```python
from msbench.audio import prepare   # 16 kHz soxr HQ, VAD trim, -1 dBFS
```

That single call is the difference between a comparable number and a plausible
one. [docs/PROTOCOL.md](docs/PROTOCOL.md) is the full contract.

## Reproduce the published numbers

```bash
msbench-fetch
bash scripts/runbook.sh all       # every anchor in the dataset card
msbench-validate                  # v2 differs from v1 only where it claims to
pytest                            # the aggregation gate
```

Selection stages S1–S3 are not re-run by any of the above and are not needed for
it: they require the 66 GB Common Voice corpus and would change `utt`.

---

## Layout

```
src/msbench/
    audio.py normalize.py metrics.py stats.py    the measurement contract
    paths.py languages.py schema.py
    asr/      whisper mms gigaam elevenlabs      recogniser backends
    sim/      wavlm_sv wavlm_ft ecapa            speaker encoders
    corpus/   inventory phonemes qc index        Common Voice analysis
    build/    pools prompts select assemble      S1-S4: corpus -> pack
              repack_v2 reports backfill         schema v2, coverage reports
              validate card card_yaml publish    gate, card generation, upload
    cli/      wer sim quality report replay fetch
docs/         PROTOCOL.md  DEFECTS.md
reports/      generated tables and reports
scripts/      corpus download, runbook
tests/
```

The pipeline runs `pools -> prompts -> select -> assemble` (S1–S4) to build a
pack from Common Voice, then `repack_v2` to migrate it to schema v2.

Adding a language, a recogniser or a speaker encoder each touch one file — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## What this does not measure

Prosody, naturalness and expressiveness. A model can hit every phoneme, match
the speaker embedding, and still sound robotic; nothing here would notice.
Targets are single utterances with a median of 6–7 words, so long-form behaviour
is untested. [PROTOCOL.md §8](docs/PROTOCOL.md#8-what-this-protocol-does-not-cover)
is the full list.

## Licence

Code Apache-2.0 (see `LICENSE`). The dataset is CC0-1.0, matching Common Voice
17.0. One vendored file is MIT. Downloaded models carry their own terms, and one
of them — MMS-1B-all — is **non-commercial**. See [`NOTICE`](NOTICE).

## Citation

```bibtex
@misc{multilingual_speech_benchmark_2026,
  title  = {Multilingual Speech Benchmark for Zero-Shot TTS},
  author = {nineninesix},
  year   = {2026},
  note   = {Version 2.0},
  url    = {https://huggingface.co/datasets/nineninesix/multilingual-speech-benchmark}
}
```

Anchors are comparable only within a version. Cite the version you used.
