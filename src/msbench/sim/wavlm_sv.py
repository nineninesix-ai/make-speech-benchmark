#!/usr/bin/env python3
"""microsoft/wavlm-base-plus-sv — the v1 encoder, kept for continuity.

This is `WavLMForXVector` from transformers: a base-plus WavLM with an x-vector
head. Its cosine similarities sit near 0.92-0.95 for same-speaker pairs, far
above the seed-tts-eval checkpoint's 0.70-0.76, because the head and the training
objective differ — not because the speakers are more similar. Numbers from this
encoder are comparable only with other numbers from this encoder.

It is also the encoder that prompt QC filtered with, which is what makes the
anchor it produces circular (**D-17**). It stays in v2 so the v1→v2 migration
table has a like-for-like row, not because it is the best instrument.
"""

from __future__ import annotations

import numpy as np

from . import register

MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000


class WavLMSV:
    name = "wavlm_sv"
    scale = "same-speaker ~0.92-0.95; not comparable with wavlm_ft"

    # batch=1 by default, deliberately. v1 (`eval/run_sim.py:63`) embedded one
    # clip at a time, so nothing was padded. Batching pads to the longest clip
    # and the convolutional front end still convolves across that padding, which
    # moves the embedding by a small amount that depends on batch composition —
    # enough to break the four-decimal reproduction of the v1 anchor. The model
    # is small and the corpus is 8,200 clips; the sequential path costs minutes.
    def __init__(self, model: str = MODEL, batch: int = 1):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.batch = batch
        self.fe = AutoFeatureExtractor.from_pretrained(model)
        self.m = WavLMForXVector.from_pretrained(model).eval().cuda()

    def embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(wavs), self.batch):
            chunk = [np.asarray(w, dtype="float32") for w in wavs[i:i + self.batch]]
            inp = self.fe(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
            kw = {}
            if "attention_mask" in inp:
                kw["attention_mask"] = inp.attention_mask.cuda()
            with torch.no_grad():
                e = self.m(inp.input_values.cuda(), **kw).embeddings
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


@register("wavlm_sv")
def _build(**kw):
    return WavLMSV(**kw)
