#!/usr/bin/env python3
"""Speaker-encoder registry.

v1 measures SIM with one checkpoint, and that single choice causes two separate
problems.

**D-16 — the scale is not the literature's.** The dataset card advertises output
"consumable by the unmodified seed-tts-eval scripts". Those scripts score with
`wavlm_large_finetune.pth`, whose cosine similarities live around 0.70-0.76.
v1's anchors come from `microsoft/wavlm-base-plus-sv`, which lives around
0.92-0.95. A user who follows the card gets ~0.7 for their model and compares it
against an anchor of 0.93 — the same quantity measured on two incompatible
rulers. v2 reports both, so each number can be read against its own anchor.

**D-17 — circularity.** Prompt QC dropped clips further than mu−2sigma from
their speaker's centroid *in WavLM-SV space* (`msbench.build.prompts:96`), and the
anchor is then measured in that same space. The pool was pre-selected for
homogeneity under the very metric that later scores it. Disclosure is not enough
here, because the effect is a number and not a caveat: v2 measures the anchor
with ECAPA-TDNN as well, which is a different model family trained on different
data, so the part of the anchor that is an artifact of the selection becomes
visible as a gap between the encoders.

    wavlm_sv   microsoft/wavlm-base-plus-sv      v1 continuity
    wavlm_ft   wavlm_large_finetune.pth          seed-tts-eval canon
    ecapa      speechbrain/spkrec-ecapa-voxceleb non-WavLM family

Every encoder returns an L2-normalised embedding, so similarity is a dot product.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SpeakerEncoder(Protocol):
    """16 kHz float32 mono in, one L2-normalised embedding per clip out."""

    name: str
    scale: str          # human-readable note on the value range, for reports

    def embed(self, wavs: list[np.ndarray]) -> np.ndarray: ...


_BUILDERS: dict[str, callable] = {}


def register(name: str):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco


def build(name: str, **kw) -> SpeakerEncoder:
    if name not in _BUILDERS:
        __import__(f"{__name__}.{name}", fromlist=["_"])
    if name not in _BUILDERS:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(_BUILDERS)}")
    return _BUILDERS[name](**kw)


ENCODERS = ["wavlm_sv", "wavlm_ft", "ecapa"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two already-normalised embeddings."""
    return float(np.dot(a, b))
