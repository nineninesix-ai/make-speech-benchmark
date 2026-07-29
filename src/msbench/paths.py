"""Where the data lives.

Every driver needs to find the benchmark pack, the corpus and the reports. The
flat layout derived those from `Path(__file__).parent.parent`, which worked only
because the code sat next to the data. An installed package does not: it can
live in site-packages on a different filesystem from the audio.

So the root is resolved at call time, in this order:

1. ``$MSBENCH_HOME`` if set — the explicit answer, and the one to use in a
   runbook or a container;
2. the nearest ancestor of the current directory that looks like a workspace
   (it holds ``tts-bench-v1``, ``tts-bench-v2``, ``cv`` or ``work``);
3. the current directory.

Nothing here creates directories or fails on a missing one. A driver that needs
a path that is not there should say so itself, with the path in the message.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Data directories that mark a workspace root, in the order they are looked for.
MARKERS = ("tts-bench-v2", "tts-bench-v1", "cv", "work")

PACK_V1 = "tts-bench-v1"
PACK_V2 = "tts-bench-v2"


def workspace() -> Path:
    """The directory the data hangs off. See the module docstring for the order."""
    env = os.environ.get("MSBENCH_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if any((base / m).is_dir() for m in MARKERS):
            return base
    return cwd


def pack(version: str = "v2") -> Path:
    """The benchmark package directory for a schema version."""
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown pack version: {version!r}")
    return workspace() / (PACK_V1 if version == "v1" else PACK_V2)


def corpus() -> Path:
    """The Common Voice download root (``cv/``), populated by scripts/fetch_cv.sh."""
    return workspace() / "cv"


def work() -> Path:
    """S1-S3 intermediates (``work/``): candidate pools, QC results, selections."""
    return workspace() / "work"


def reports() -> Path:
    """Generated reports, tables and per-utterance metric artifacts."""
    return workspace() / "reports"
