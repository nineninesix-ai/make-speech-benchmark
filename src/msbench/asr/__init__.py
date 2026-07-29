#!/usr/bin/env python3
"""ASR backend registry.

v1 reads every subset with one ASR (Whisper-large-v3, GigaAM for Kyrgyz), which
conflates two different things: what a TTS model got wrong, and what that one
recogniser happens to be bad at (**G-03**). Whisper in particular carries a
strong internal language model that *repairs* mispronounced synthesis, so it
systematically understates unintelligibility — the failure mode a TTS benchmark
most needs to catch.

v2 therefore reads each subset with more than one family and reports the
agreement:

    whisper   openai/whisper-large-v3   encoder-decoder + strong internal LM
    mms       facebook/mms-1b-all       CTC, no LM — the cleanest intelligibility probe
    gigaam    ai-sage/GigaAM-Multilingual   v1 reference for Kyrgyz

A backend is a small object with three members; anything satisfying `ASRBackend`
can be registered without touching the drivers.

The `repetition_penalty` question (**D-02**) is settled in `whisper.py`, not here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRBackend(Protocol):
    """16 kHz float32 mono in, plain text out — no speaker tags, no timestamps."""

    name: str

    def supports(self, lang: str) -> bool: ...

    def transcribe(self, wavs: list[np.ndarray], lang: str) -> list[str]: ...


_BUILDERS: dict[str, callable] = {}


def register(name: str):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco


def build(name: str, **kw) -> ASRBackend:
    """Instantiate a backend by registry name. Import is lazy: loading the
    registry must not drag in every model's dependencies."""
    if name not in _BUILDERS:
        # Import on demand so `--asr mms` never imports Whisper, and a broken
        # optional dependency only breaks the backend that needs it.
        __import__(f"{__name__}.{_MODULE.get(name, name)}", fromlist=["_"])
    if name not in _BUILDERS:
        raise KeyError(f"unknown ASR backend {name!r}; have {sorted(_BUILDERS)}")
    return _BUILDERS[name](**kw)


def available() -> list[str]:
    return sorted(set(_MODULE) | set(_BUILDERS))


# registry name -> module file, where they differ
_MODULE = {"whisper": "whisper", "mms": "mms", "gigaam": "gigaam",
           "elevenlabs": "elevenlabs"}
