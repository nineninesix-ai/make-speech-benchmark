#!/usr/bin/env python3
"""wavlm_large_finetune.pth — the checkpoint the stock seed-tts-eval scripts use.

This exists to close **D-16**. The dataset card claims its layout is consumable
by the unmodified seed-tts-eval scripts, and it is; but those scripts score
similarity with this checkpoint, on a scale where same-speaker pairs land near
0.70-0.76, while v1's published anchors come from `wavlm-base-plus-sv` at
0.92-0.95. Anyone who took the card at its word compared their model's ~0.7
against an anchor of 0.93 and concluded their cloning was catastrophic. Shipping
an anchor measured with this checkpoint is what makes the claimed compatibility
real.

Architecture is `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')` —
a WavLM-Large backbone with a learned softmax weighting over its 25 hidden
layers, feeding an ECAPA-TDNN head that outputs 256 dimensions. The definition is
vendored verbatim in `_vendor/ecapa_tdnn.py`; see that file's header for the one
substituted import.

Note that comparability with the literature is still only partial, and the
remaining gap is not fixable here: Seed-TTS conditions on 3-20 s prompts while
this benchmark uses 2.5-5 s, and the card's own table shows SIM rising with
prompt duration. Same ruler, different measurement conditions.
"""

from __future__ import annotations

import contextlib

import numpy as np

from . import register

REPO = "subatomicseer/wavlm-large-sv-ckpts"
BACKBONE = "wavlm_large.pt"
CHECKPOINT = "wavlm_large_finetune.pth"
SR = 16000


@contextlib.contextmanager
def _allow_full_unpickle():
    """Load these two checkpoints with `weights_only=False`.

    Both files predate PyTorch 2.6, which flipped `torch.load`'s default, and
    both carry a config object rather than a bare tensor dict — s3prl reads
    `checkpoint["cfg"]` to build the model. The loading code lives inside s3prl
    and the vendored module, so the flag cannot be passed down; patching the
    default for the duration of construction is the narrowest available fix.
    These are pinned files from a known repository, downloaded over HTTPS.
    """
    import torch
    orig = torch.load

    def patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return orig(*a, **kw)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


class WavLMFinetuned:
    name = "wavlm_ft"
    scale = "same-speaker ~0.70-0.76; the seed-tts-eval scale"

    def __init__(self, repo: str = REPO, batch: int = 1):
        import torch
        from huggingface_hub import hf_hub_download

        from ._vendor.ecapa_tdnn import ECAPA_TDNN_SMALL

        self.torch = torch
        self.batch = batch
        backbone = hf_hub_download(repo, BACKBONE)
        ckpt = hf_hub_download(repo, CHECKPOINT)

        with _allow_full_unpickle():
            self.m = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large",
                                      config_path=backbone)
            state = torch.load(ckpt, map_location="cpu")["model"]
        missing, unexpected = self.m.load_state_dict(state, strict=False)
        # `loss_calculator.*` belongs to the training objective and has no
        # forward-time role; anything else missing means the architecture and
        # the checkpoint disagree, and the embeddings would be silently wrong.
        bad = [k for k in unexpected if not k.startswith("loss_calculator")]
        if missing or bad:
            raise RuntimeError(
                f"checkpoint does not match the model: missing={missing[:5]} "
                f"unexpected={bad[:5]}")
        self.m = self.m.eval().cuda()

    def embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(wavs), self.batch):
            chunk = [torch.from_numpy(np.asarray(w, dtype="float32")).cuda()
                     for w in wavs[i:i + self.batch]]
            x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
            with torch.no_grad():
                e = self.m(x)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


@register("wavlm_ft")
def _build(**kw):
    return WavLMFinetuned(**kw)
