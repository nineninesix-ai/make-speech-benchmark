#!/usr/bin/env python3
"""GigaAM-Multilingual (revision `ctc`) — the v1 reference for Kyrgyz.

Kept in v2 for one reason: it produced the published `ky` anchor of 11.04 %, and
that number cannot be revised or defended without re-measuring it with the same
instrument. It is not promoted to a general backend — it is the control in the
MMS↔GigaAM cross-check that **D-27** requires.

The model ships as `trust_remote_code` and expects a filesystem path rather than
an array, so this backend materialises each batch to a temporary WAV. That is
the one place in `eval/` where audio goes back through the filesystem; it is the
model's calling convention, not a choice.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import register

MODEL = "ai-sage/GigaAM-Multilingual"
REVISION = "ctc"
SR = 16000


class GigaAMASR:
    name = "gigaam"

    # `batch` is accepted and ignored: the model transcribes one file at a time,
    # but the drivers pass a uniform keyword to every backend.
    def __init__(self, model: str = MODEL, revision: str = REVISION,
                 batch: int = 1):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(model, revision=revision,
                                               trust_remote_code=True)
        if hasattr(self.model, "cuda"):
            self.model = self.model.cuda()
        self.model.eval()

    def supports(self, lang: str) -> bool:
        return lang == "ky"

    def config(self) -> dict:
        return {"model": MODEL, "revision": REVISION, "decoding": "ctc"}

    def transcribe(self, wavs: list[np.ndarray], lang: str) -> list[str]:
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, w in enumerate(wavs):
                p = Path(tmp) / f"{i}.wav"
                sf.write(p, np.asarray(w, dtype="float32"), SR)
                with self.torch.no_grad():
                    t = self.model.transcribe(str(p))
                if isinstance(t, (list, tuple)):
                    t = t[0] if t else ""
                out.append(str(t).strip())
        return out


@register("gigaam")
def _build(**kw):
    return GigaAMASR(**kw)
