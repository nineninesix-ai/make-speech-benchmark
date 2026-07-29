#!/usr/bin/env python3
"""speechbrain/spkrec-ecapa-voxceleb — the encoder from outside the WavLM family.

This is the one that answers **D-17**. Prompt QC rejected clips lying more than
mu−2sigma from their speaker's centroid as measured by WavLM-SV, and the human
anchor was then computed with WavLM-SV. Whatever part of that anchor is an
artifact of having filtered in the metric's own space cannot be seen from inside
the family — `wavlm_ft` shares the same backbone and would inherit the same
blind spot.

ECAPA-TDNN trained on VoxCeleb shares neither the architecture, the
self-supervised pretraining, nor the training corpus. Where its anchor tracks
WavLM's, the anchor is a property of the speakers; where it does not, the
selection is showing through.

It is not a drop-in replacement for the other two: VoxCeleb is interview and
in-the-wild audio, this benchmark is read speech from Common Voice, and the
absolute scale differs again. Only within-encoder comparisons are meaningful —
which is exactly why `sim_norm` (score minus impostor floor, over anchor minus
floor) is the quantity the reports lead with.
"""

from __future__ import annotations

import numpy as np

from . import register

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SR = 16000


class EcapaVox:
    name = "ecapa"
    scale = "VoxCeleb ECAPA cosine; own scale, read against its own floor"

    def __init__(self, model: str = MODEL, batch: int = 1, cache: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.batch = batch
        self.m = EncoderClassifier.from_hparams(
            source=model, savedir=cache or f"/tmp/sb-{model.split('/')[-1]}",
            run_opts={"device": "cuda:0"})
        self.m.mods.eval()

    def embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(wavs), self.batch):
            chunk = [torch.from_numpy(np.asarray(w, dtype="float32"))
                     for w in wavs[i:i + self.batch]]
            lens = torch.tensor([len(c) for c in chunk], dtype=torch.float32)
            x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True).cuda()
            # speechbrain wants relative lengths so padding is excluded from the
            # statistics pooling rather than averaged into the embedding.
            rel = (lens / lens.max()).cuda()
            with torch.no_grad():
                e = self.m.encode_batch(x, rel).squeeze(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


@register("ecapa")
def _build(**kw):
    return EcapaVox(**kw)
