#!/usr/bin/env python3
"""ElevenLabs Scribe v1 — a hosted recogniser, and the only one here covering
all six subsets with one model.

It joins Whisper and MMS rather than replacing either. The three answer different
questions:

    whisper     strong internal LM, repairs what it half-hears
    mms         CTC, no LM at all — what was actually said acoustically
    elevenlabs  a second strong-LM read, from a different vendor and training set

Adding it does two useful things. It gives **Kyrgyz** a third opinion —
`kir` is supported, which Whisper's 100 languages do not include — so the `ky`
anchor no longer rests on GigaAM cross-checked by a single CTC baseline. And on
the other five it separates "Whisper is right" from "Whisper's language model
happens to agree with the reference", which one strong recogniser cannot do
alone.

Four things to know before reading its numbers.

**It is not deterministic, even pinned.** Identical requests return different
transcripts. The API exposes `seed` and `temperature`, and both are set here —
`seed` alone changes nothing, and the pair helps a great deal — but they do not
close the gap: repeating three runs over the same clips gives 10/12 identical for
`en-US`, 11/12 for `pt-BR` and **4/12 for `ky`**. `scribe_v1` is worse still, and
that, not the version number, is why `scribe_v2` is the default here.

The spread was then measured rather than left as a worry, by running `ky` — the
worst subset — three times end to end. Only **69.5 %** of its transcripts are
identical across all three runs, but corpus WER lands at 0.1829 / 0.1805 /
0.1809: a range of **0.0024**, against a bootstrap interval 0.078 wide. So the
nondeterminism is real per utterance and negligible in the aggregate. Audit a
single row and you may not reproduce it; quote a headline and you will.

**Its Kyrgyz output is orthographically contaminated.** The subset's references
use exactly the 36 letters of the Kyrgyz alphabet and nothing else, but 8-10 % of
Scribe transcripts contain letters from neighbouring Turkic languages — `ғ`, `қ`,
`ұ`, `ә`, `і` (Kazakh) and `ҡ` (Bashkir). Those are scored as substitutions, so
part of its `ky` error rate is a spelling convention rather than a mishearing.
`nonstandard_chars` counts it per row so the two can be separated.

**It hallucinates on short clips.** On a 1.1 s pt-BR utterance where Whisper
scores 0.00 it returned an unrelated sentence. This is not a transport problem —
`audio_duration_secs` comes back matching the clip to the hundredth — it is the
failure mode of a generative decoder given too little to condition on. Watch
`catastrophic_rate`, which is what that failure produces, and do not read a
Scribe WER without it.

**Audio events are suppressed.** Scribe tags laughter and similar by default;
those tags would be scored as words. `tag_audio_events=false` is not optional
here.

There is no batch endpoint: the API takes one file per request (passing two
`file` parts returns a single result, and a `files` field is rejected), so
throughput comes from concurrency. Measured on this account, 12 workers sustain
~8.8 clips/s cleanly and 24 starts returning 429.

Needs `ELLBS_TOKEN`, read from the environment or `.env` at the repo root.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import register

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL = "scribe_v2"
SR = 16000
SEED = 0

# The Kyrgyz alphabet. Scribe reaches for neighbouring Turkic letters that do not
# occur in it, and none of these appear anywhere in the subset's references.
KY_ALPHABET = set("абвгдеёжзийклмнңоөпрстуүфхцчшщъыьэюя")

# subset code -> Scribe language code (ISO 639-3). All six confirmed against the
# live API, each detected with probability 1.0.
LANG = {"en-US": "eng", "es-ES": "spa", "es-MX": "spa",
        "nl-NL": "nld", "pt-BR": "por", "ky": "kir"}


def _token() -> str:
    tok = os.environ.get("ELLBS_TOKEN") or os.environ.get("ELEVENLABS_API_KEY")
    if tok:
        return tok.strip()
    for base in (Path.cwd(), *Path(__file__).resolve().parents[:4]):
        f = base / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip().startswith("ELLBS_TOKEN"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ELLBS_TOKEN not found in the environment or .env")


def _wav_bytes(wav: np.ndarray) -> bytes:
    """float32 [-1, 1] -> 16-bit PCM WAV. The API takes a container, not raw
    samples, and 16-bit is what the rest of the pipeline already assumes."""
    x = np.clip(np.asarray(wav, dtype="float32"), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def _multipart(audio: bytes, fields: dict[str, str]) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{b}\r\nContent-Disposition: form-data; "
            f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f"--{b}\r\nContent-Disposition: form-data; "
        f'name="file"; filename="clip.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode() + audio + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


class ElevenLabsASR:
    name = "elevenlabs"

    def __init__(self, batch: int = 16, workers: int = 12, timeout: int = 300,
                 retries: int = 5, model: str = MODEL, seed: int = SEED,
                 temperature: float = 0.0):
        # `batch` is the driver's chunk size and means nothing to a per-clip HTTP
        # API; `workers` is the parameter that actually sets throughput. 12 is
        # measured: it sustains ~8.8 clips/s on this account, and 24 draws 429s.
        self.batch = batch
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.token = _token()
        self.n_calls = 0
        self.n_retried = 0
        self.detected: dict[str, int] = {}

    def supports(self, lang: str) -> bool:
        return lang in LANG

    def config(self) -> dict:
        return {"model": self.model, "endpoint": ENDPOINT,
                "language_code": "per-subset ISO 639-3",
                "tag_audio_events": False, "diarize": False,
                "timestamps_granularity": "none",
                "seed": self.seed, "temperature": self.temperature,
                # Explicitly off: `no_verbatim` strips filler words, which would
                # delete real content from a WER reference comparison, and
                # `keyterms` biases decoding toward supplied words — a benchmark
                # that hints at its own answers measures nothing.
                "no_verbatim": False, "keyterms": None,
                "timeout_s": self.timeout, "workers": self.workers,
                "retries": self.retries}

    def _one(self, wav: np.ndarray, lang: str) -> tuple[str, str | None]:
        fields = {
            "model_id": self.model,
            # Pinning the language is the point of having a per-subset code:
            # letting it auto-detect would make `es-ES` and `es-MX` answerable
            # by different models of Spanish, and would occasionally read a
            # short Kyrgyz clip as Kazakh.
            "language_code": LANG[lang],
            "tag_audio_events": "false",
            "diarize": "false",
            "timestamps_granularity": "none",
            # Neither pins the output fully (see the module docstring), but the
            # pair narrows the spread substantially and costs nothing.
            "seed": str(self.seed),
            "temperature": str(self.temperature),
        }
        body, ctype = _multipart(_wav_bytes(wav), fields)
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                ENDPOINT, data=body,
                headers={"xi-api-key": self.token, "Content-Type": ctype})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                self.n_calls += 1
                code = d.get("language_code")
                self.detected[code] = self.detected.get(code, 0) + 1
                return d.get("text", "").strip(), None
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                # 4xx other than rate-limiting will not fix themselves.
                if e.code not in (408, 429, 500, 502, 503, 504):
                    return "", f"{last}: {e.read()[:200].decode(errors='replace')}"
            except Exception as e:  # timeouts, resets, DNS
                last = f"{type(e).__name__}: {e}"
            self.n_retried += 1
            time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
        return "", last

    def transcribe(self, wavs: list[np.ndarray], lang: str) -> list[str]:
        """Concurrency lives here, not in the driver: this backend is bound by
        round-trip latency, so the useful parallelism is requests in flight."""
        if not wavs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            out = list(ex.map(lambda w: self._one(w, lang), wavs))
        failed = [e for _, e in out if e]
        if failed:
            # Loud, and counted: a silently empty hypothesis scores as a total
            # miss and would be indistinguishable from a real ASR failure.
            raise RuntimeError(
                f"{len(failed)}/{len(wavs)} Scribe calls failed after "
                f"{self.retries} attempts; first: {failed[0]}")
        return [t for t, _ in out]


@register("elevenlabs")
def _build(**kw):
    return ElevenLabsASR(**kw)
