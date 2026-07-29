#!/usr/bin/env python3
"""Whisper-large-v3 — primary backend, and the home of D-02.

**D-02 — the repetition penalty.** v1 decodes with `repetition_penalty=1.1`
(v1's `eval/run_wer.py:65`). That is not a standard WER-evaluation setting, and its
effect is not neutral. The penalty suppresses repeated tokens, which is exactly
the signature of the most important autoregressive-TTS failure mode: looping,
babbling, repeated syllables. It is applied identically to the human anchor and
to synthesis, but human recordings almost never loop, so the anchor barely moves
while a model's worst failures are smoothed away. The bias is asymmetric, and it
flatters models. v2 removes it: a looping model must surface as a catastrophic
WER, not be quietly repaired by the decoder.

`max_new_tokens=200` stays. The original comment attributed the OOM guard to the
combination of greedy decoding and the token cap; the cap is what actually
bounds the generation, and greedy decoding is standard for WER anyway.

**D-10 — `chunk_length_s`.** v1 passes `chunk_length_s=30`, which engages
Whisper's long-form chunking algorithm on clips of 2-6 seconds. It is a
different code path from the short-form one — padding, stride and the
timestamp logic all differ — and it was never documented. v2 defaults to the
short-form path (`chunk_length_s=None`) since no clip in the benchmark comes
near 30 s, and keeps the parameter exposed so the v1 setting can be replayed
exactly.

Both parameters are constructor arguments precisely so that
`msbench.cli.replay` can attribute the v1→v2 delta to each of them
separately, on identical audio, instead of reporting one lump.
"""

from __future__ import annotations

import numpy as np

from . import register

MODEL = "openai/whisper-large-v3"
SR = 16000

# The subset code -> Whisper language token. Kyrgyz is absent on purpose:
# Whisper is not usable for it (that is why v1 uses GigaAM).
WHISPER_LANG = {"en-US": "en", "es-ES": "es", "es-MX": "es",
                "nl-NL": "nl", "pt-BR": "pt"}


class WhisperASR:
    name = "whisper"

    def __init__(self, model: str = MODEL, batch: int = 16,
                 repetition_penalty: float | None = None,
                 chunk_length_s: float | None = None,
                 num_beams: int = 1, max_new_tokens: int = 200,
                 dtype: str = "float16"):
        import torch
        from transformers import pipeline

        self.batch = batch
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.chunk_length_s = chunk_length_s
        kw = {"chunk_length_s": chunk_length_s} if chunk_length_s else {}
        self.pipe = pipeline("automatic-speech-recognition", model=model,
                             dtype=getattr(torch, dtype), device="cuda", **kw)

    def supports(self, lang: str) -> bool:
        return lang in WHISPER_LANG

    def config(self) -> dict:
        """Every decoding parameter that determines a number, for the report."""
        return {"model": MODEL, "num_beams": self.num_beams,
                "max_new_tokens": self.max_new_tokens,
                "repetition_penalty": self.repetition_penalty,
                "chunk_length_s": self.chunk_length_s, "batch": self.batch}

    def transcribe(self, wavs: list[np.ndarray], lang: str) -> list[str]:
        gk = {"language": WHISPER_LANG[lang], "task": "transcribe",
              "num_beams": self.num_beams, "max_new_tokens": self.max_new_tokens}
        if self.repetition_penalty is not None:
            gk["repetition_penalty"] = self.repetition_penalty
        out = self.pipe([{"raw": w, "sampling_rate": SR} for w in wavs],
                        batch_size=self.batch, generate_kwargs=gk)
        return [o["text"].strip() for o in out]


@register("whisper")
def _build(**kw):
    return WhisperASR(**kw)
