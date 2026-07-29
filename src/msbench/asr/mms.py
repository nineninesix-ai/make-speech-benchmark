#!/usr/bin/env python3
"""MMS-1B-all — CTC, no language model, and the only non-GigaAM read of Kyrgyz.

The point of this backend is what it *lacks*. Whisper decodes with a strong
internal language model, so a mispronounced or half-swallowed word is often
repaired into the word the sentence implies — a TTS system is then credited with
intelligibility it did not produce (**G-03**). A CTC model with greedy decoding
and no LM has nothing to repair with: it emits what it heard. Its absolute WER is
higher on every subset, which is not a defect. The comparison that carries
information is Whisper-minus-MMS, per system: a large gap means the intelligible
reading depends on the listener's expectations.

It is also the only recogniser besides GigaAM that covers Kyrgyz, which is what
makes **D-27** answerable — the v1 `ky` anchor of 11.04 % rests on a single
unvalidated ASR, above the 10 % threshold the card itself names as needing a
cross-check.

One adapter per language, loaded into the shared 1B encoder by
`model.load_adapter(<iso3>)`; the tokeniser has to be switched in step, which is
the part that silently produces garbage if forgotten.
"""

from __future__ import annotations

import numpy as np

from . import register

MODEL = "facebook/mms-1b-all"
SR = 16000

# subset code -> MMS adapter (ISO 639-3). All five confirmed present in the repo,
# along with `kir`, which is the reason this backend exists.
ADAPTER = {"en-US": "eng", "es-ES": "spa", "es-MX": "spa",
           "nl-NL": "nld", "pt-BR": "por", "ky": "kir"}


class MMSASR:
    name = "mms"

    def __init__(self, model: str = MODEL, batch: int = 8, dtype: str = "float32"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.torch = torch
        self.batch = batch
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model, dtype=getattr(torch, dtype)).eval().cuda()
        self._adapter = None

    def supports(self, lang: str) -> bool:
        return lang in ADAPTER

    def config(self) -> dict:
        return {"model": MODEL, "decoding": "ctc-greedy", "lm": None,
                "batch": self.batch}

    def _set_lang(self, lang: str):
        code = ADAPTER[lang]
        if code == self._adapter:
            return
        # Both halves are required. Swapping the adapter without swapping the
        # tokeniser leaves the CTC ids being decoded through the previous
        # language's vocabulary, which yields fluent-looking nonsense rather
        # than an error.
        self.processor.tokenizer.set_target_lang(code)
        self.model.load_adapter(code)
        self._adapter = code

    def transcribe(self, wavs: list[np.ndarray], lang: str) -> list[str]:
        self._set_lang(lang)
        out = []
        for i in range(0, len(wavs), self.batch):
            chunk = [np.asarray(w, dtype="float32") for w in wavs[i:i + self.batch]]
            inputs = self.processor(chunk, sampling_rate=SR, return_tensors="pt",
                                    padding=True)
            iv = inputs.input_values.to("cuda", dtype=self.model.dtype)
            mask = inputs.get("attention_mask")
            kw = {"attention_mask": mask.cuda()} if mask is not None else {}
            with self.torch.no_grad():
                logits = self.model(iv, **kw).logits
            ids = logits.argmax(dim=-1).cpu()
            out += [t.strip() for t in self.processor.batch_decode(ids)]
        return out


@register("mms")
def _build(**kw):
    return MMSASR(**kw)
