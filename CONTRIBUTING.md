# Contributing

The three extensions this benchmark is designed for — a language, a recogniser,
a speaker encoder — each touch one file. That is a design constraint, not a
coincidence: anything that requires editing a driver is a sign the abstraction
is wrong, and is worth reporting as a bug on its own.

Before anything else, read [docs/PROTOCOL.md](docs/PROTOCOL.md). Most changes
that look like improvements will silently move a published anchor, and the
question "does this change a number that is already out there?" has to be
answered before the change, not after.

```bash
uv pip install -e ".[all]"
pytest && ruff check src tests scripts     # both must pass
```

---

## Adding a recogniser

One new file in `src/msbench/asr/`. A backend is any object with `name`,
`supports(lang)` and `transcribe(wavs, lang) -> list[str]`; 16 kHz float32 mono
in, plain text out — no speaker tags, no timestamps.

```python
from . import register

class MyASR:
    name = "myasr"
    def supports(self, lang): return lang in {"en-US", "nl-NL"}
    def transcribe(self, wavs, lang): ...

@register("myasr")
def _build(**kw):
    return MyASR(**kw)
```

Then add it to `_MODULE` in `src/msbench/asr/__init__.py` — the registry imports
backends lazily, so `--asr mms` never imports Whisper and a broken optional
dependency only breaks the backend that needs it.

**Rules that are not negotiable:**

- **Never return empty strings on failure.** A failed transcription that comes
  back as `""` scores as a 100 % error and looks like a catastrophically bad
  model. Raise. `elevenlabs.py` retries and then raises; copy that shape.
- **Pin decoding.** Greedy, no repetition penalty, an explicit token cap. If the
  backend is a hosted API, pin every parameter that exists and say in the
  docstring which ones do not actually pin anything (Scribe's `seed` does not).
- **State the language mapping explicitly.** Never rely on auto-detection.

## Adding a speaker encoder

One new file in `src/msbench/sim/`, same shape: `embed(wavs) -> np.ndarray`,
registered with `@register("name")`.

An encoder is only useful here if its **impostor floor** is measured too —
`msbench-sim --mode floor --pairs 5000`. A cosine of 0.9 means nothing until you
know what two different speakers score, and the floors differ enormously between
encoders (`wavlm_sv` ≈ 0.61–0.74, `ecapa` ≈ 0.08–0.17).

Prefer an encoder from outside the WavLM family. v1's QC pre-selected clips for
homogeneity in WavLM-SV space and then measured the anchor with WavLM-SV
(**D-17**); every additional WavLM-family encoder inherits that circularity.

## Adding a language

One entry in `LANGUAGES` in `src/msbench/languages.py`. Nothing else in the
pipeline hard-codes a language. Run `python -m msbench.languages` for a registry
self-check, and `python -m msbench.corpus.check_phonemes --lang xx` before
trusting the espeak voice.

The fields that go wrong:

- **`espeak`** — verify the voice on real text first. `es` has /θ/ and `es-419`
  does not; using the wrong one silently corrupts the phonetic coverage
  objective. Some voices emit X-SAMPA rather than IPA and fall back to English on
  unreadable tokens (Kyrgyz does both).
- **`match`** — Common Voice has no `es-MX` / `pt-BR` / `nl-NL` locales.
  Regional variants live in free-text, self-declared, multi-valued `accents` /
  `variant` columns whose labels contain commas inside parentheses. Split with
  `split_labels` and match exactly; substring matching gives false positives.
- **`cap`** and **`n_main`** — the per-speaker cap has to satisfy
  `speakers_surviving_QC * cap >= n_main`, and you do not know the survivor
  count until S2 has run.
- **`num2words`** — `None` is legitimate (there is no Kyrgyz locale) but means
  digits pass through unexpanded on both sides.

A new language needs a full S1–S4 build, which needs the 66 GB corpus. It also
produces a new subset with **no human anchor** until the anchor runs are done —
do not publish one without it, because the anchor is the entire point.

## Changing anything in `msbench.audio` or `msbench.normalize`

These two modules *are* the published numbers. A change to either invalidates
every anchor in the current dataset version.

If a change is genuinely warranted, it is a new dataset version: bump the
version, re-run the anchors, and extend [docs/DEFECTS.md](docs/DEFECTS.md) with
what was wrong and what it cost, measured. `msbench-replay` exists to attribute
a v(n) → v(n+1) delta to individual causes on identical audio — use it, and
publish the ladder. One lump labelled "improved methodology" is not acceptable
here; it is exactly what v1 did wrong.

## Reporting a defect

Defects get an ID and stay in the record permanently, including after they are
fixed. Open an issue with:

1. what the code does, with a file and line;
2. what the card or docs claim it does;
3. whether it moves a published number, and by how much if you measured it.

Point (3) is the valuable part and the one usually missing. "This looks wrong"
is a useful report; "this looks wrong and it is worth 0.003 WER in pt-BR" is a
finding.

## Style

`ruff check` with the repo config, 90 columns. Comments explain *why*, not what
— the existing code is dense with rationale for choices that look arbitrary
(`batch=8` is not arbitrary; neither is `chunk_length_s` being unset), and that
rationale is what makes the benchmark auditable. Match it.

Never commit `.env`, credentials, or anything under the data directories in
`.gitignore`.
