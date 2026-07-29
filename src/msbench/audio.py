#!/usr/bin/env python3
"""Shared audio handling — the single source of truth for eval-side preprocessing.

Two v1 defects live here (see V2_IMPLEMENTATION_PLAN §3):

**D-03 — resampling.** v1's `eval/run_sim.py:50` and `msbench.cli.quality:68` resample
with `np.interp`: linear interpolation, no anti-aliasing filter. The stored
references are already 16 kHz and never take that path, but synthesis at 22.05 /
24 / 44.1 kHz always does and reaches the speaker encoder carrying aliasing
artifacts. Every model's SIM is depressed by a purely technical defect while the
human anchor is untouched. `resample()` below uses soxr HQ — a proper polyphase
resampler with a stopband — and `np.interp` must not be used for audio anywhere
in `eval/`.

**D-05 — preprocessing parity.** The anchors were computed on audio that
`msbench.build.assemble:load_trimmed` had already put into a canonical state: mono,
16 kHz, Silero-VAD trimmed with a 50 ms pad, peak-normalised to −1 dBFS.
`run_sim.py` applied none of that to synthesis, so trailing silence and level
differences shifted SIM. `prepare()` reproduces the build-time state exactly, and
the v2 drivers route synthesis through it.

**What parity does *not* mean.** The stored pack audio is already in the canonical
state, so it must be fed to the encoders as-is. Running `prepare()` over it again
would re-trim with a *different* Silero version than the build used and quietly
move the anchor. Preprocess synthesis; leave stored references alone. That
asymmetry is deliberate — `prepare_stored()` documents it in code.

**A note on the build.** `msbench.corpus.qc:113` resamples with `np.interp` too,
so the packaged 16 kHz audio was produced through linear interpolation from the
Common Voice mp3s. That is frozen into the published data and is not fixable
without re-running S1–S4, which v2 deliberately does not do (§2.1). It is
recorded in METHODOLOGY as a known property of the corpus, and it is one more
reason not to compare absolute SIM across datasets.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 16000
PEAK_DBFS = -1.0
VAD_PAD_S = 0.05          # msbench.build.assemble keeps 50 ms either side of speech

_VAD = None


def _vad():
    """Silero VAD, loaded once per process.

    `build/` loads it through `torch.hub`; here it comes from the pinned
    `silero-vad` wheel so the eval path needs no network. The model weights are
    the same, but the wrapper version is not — hence `prepare_stored()`.
    """
    global _VAD
    if _VAD is None:
        from silero_vad import get_speech_timestamps, load_silero_vad
        _VAD = (load_silero_vad(), get_speech_timestamps)
    return _VAD


def to_mono(wav: np.ndarray) -> np.ndarray:
    return wav.mean(axis=1) if wav.ndim > 1 else wav


def resample(wav: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited resampling. NEVER `np.interp` — see D-03."""
    if sr_in == sr_out:
        return wav.astype("float32", copy=False)
    return soxr.resample(wav.astype("float32", copy=False), sr_in, sr_out,
                         quality="HQ").astype("float32")


def peak_normalize(wav: np.ndarray, dbfs: float = PEAK_DBFS) -> np.ndarray:
    """Scale so the sample peak sits at `dbfs`. Mirrors msbench.build.assemble:peak_norm."""
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak <= 0:
        return wav.astype("float32", copy=False)
    return (wav / peak * (10 ** (dbfs / 20))).astype("float32")


def vad_trim(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Drop edge silence, keeping `VAD_PAD_S` either side of the speech span.

    Same boundaries as `msbench.corpus.qc:analyse` — first timestamp start
    minus the pad, last timestamp end plus the pad. Interior pauses are kept:
    trimming them would change the prosody the encoder sees.
    """
    model, get_ts = _vad()
    import torch
    ts = get_ts(torch.from_numpy(np.asarray(wav, dtype="float32")), model,
                sampling_rate=sr)
    if not ts:
        return wav.astype("float32", copy=False)
    pad = int(VAD_PAD_S * sr)
    a = max(0, ts[0]["start"] - pad)
    b = min(len(wav), ts[-1]["end"] + pad)
    return wav[a:b].astype("float32") if b > a else wav.astype("float32", copy=False)


def prepare(wav: np.ndarray, sr: int, *, trim: bool = True,
            peak: bool = True) -> np.ndarray:
    """Put arbitrary audio into the state the anchors were measured in.

    mono → 16 kHz (soxr HQ) → VAD trim → peak-normalise to −1 dBFS.
    The default is full parity; `--no-preprocess` on the drivers turns it off for
    users who want raw model output scored.
    """
    x = resample(to_mono(np.asarray(wav)), sr, SR)
    if trim:
        x = vad_trim(x, SR)
    if peak:
        x = peak_normalize(x)
    return x


def prepare_stored(wav: np.ndarray, sr: int) -> np.ndarray:
    """Pack audio (`prompt_audio`, `gt_audio`, `sim_ref_audio`) — already canonical.

    `msbench.build.assemble` wrote these mono, at 16 kHz, VAD-trimmed and
    peak-normalised. Only mono folding and a defensive resample remain; do not
    trim again (see the module docstring).
    """
    return resample(to_mono(np.asarray(wav)), sr, SR)


def read_cell(cell, *, stored: bool = True, trim: bool = True,
              peak: bool = True) -> np.ndarray | None:
    """Decode a parquet audio cell or a filesystem path to 16 kHz float32 mono.

    `stored=True` for pack columns, `stored=False` for synthesis, which gets the
    full parity pipeline.
    """
    if cell is None:
        return None
    if isinstance(cell, (str, Path)):
        p = Path(cell)
        if not p.exists():
            return None
        wav, sr = sf.read(p, dtype="float32")
    elif isinstance(cell, dict):
        wav, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    else:
        wav, sr = sf.read(io.BytesIO(bytes(cell)), dtype="float32")
    if stored:
        return prepare_stored(wav, sr)
    return prepare(wav, sr, trim=trim, peak=peak)


def dbfs_peak(wav: np.ndarray) -> float:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def _demo_aliasing():
    """Quantify D-03 on a signal whose correct answer is known exactly.

    A 15 kHz tone sampled at 44.1 kHz is above the 8 kHz Nyquist limit of 16 kHz
    audio, so a correct resampler must reject it and output near-silence. Linear
    interpolation has no stopband: the tone folds down to |16000 − 15000| = 1 kHz
    and lands in the middle of the speech band, where a speaker encoder reads it
    as timbre. That is the mechanism behind every model's depressed SIM.
    """
    sr_hi, n = 44100, 44100
    t = np.arange(n) / sr_hi
    tone = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype("float32")

    def band_energy(x, lo, hi, sr=SR):
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return float(spec[(f >= lo) & (f < hi)].sum())

    good = resample(tone, sr_hi, SR)
    idx = np.linspace(0, len(tone) - 1, int(round(len(tone) * SR / sr_hi)))
    bad = np.interp(idx, np.arange(len(tone)), tone).astype("float32")

    rms_in = float(np.sqrt((tone ** 2).mean()))
    print("15 kHz tone @44.1 kHz -> 16 kHz.  Correct output: silence.")
    for name, y in (("soxr HQ", good), ("np.interp", bad)):
        rms = float(np.sqrt((y ** 2).mean()))
        alias = 100 * band_energy(y, 900, 1100) / (band_energy(y, 0, 8000) + 1e-20)
        print(f"  {name:<10} residual {20 * np.log10(rms / rms_in + 1e-20):7.1f} dBr"
              f"   energy in the 1 kHz alias band: {alias:5.1f} %")
    return good, bad


if __name__ == "__main__":
    _demo_aliasing()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, SR).astype("float32")
    print(f"\npeak_normalize -> {dbfs_peak(peak_normalize(x)):.3f} dBFS (want -1.000)")
