#!/usr/bin/env bash
# S0 - download Common Voice 17 transcripts from the fsicoli mirror.
#
#   ./fetch_cv.sh                # all locales
#   ./fetch_cv.sh es pt nl       # selected locales
#
# Audio is NOT fetched here: anonymous HTTP from the Hub is throttled per
# connection and silently degrades to a few MB/s. Use scripts/fetch_audio.py,
# which goes through huggingface_hub's xet protocol (~30x faster on the same
# link). Transcripts are small enough that it does not matter.
set -uo pipefail

REPO=${REPO:-fsicoli/common_voice_17_0}
BASE=https://huggingface.co/datasets/$REPO/resolve/main
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)/cv}
LANGS_DEFAULT="es pt nl en ky"

fetch_one() { # url dest
  local url=$1 dest=$2
  [ -s "$dest" ] && { echo "  = $(basename "$dest") (already present)"; return 0; }
  mkdir -p "$(dirname "$dest")"
  curl -L --fail --silent --show-error \
       --retry 10 --retry-delay 3 --retry-all-errors \
       -o "$dest.part" "$url" \
    && mv "$dest.part" "$dest" \
    && echo "  + $(basename "$dest") $(du -h "$dest" | cut -f1)" \
    || { echo "  ! FAILED $(basename "$dest")"; rm -f "$dest.part"; return 1; }
}

for l in ${*:-$LANGS_DEFAULT}; do
  echo "[$l] transcripts"
  for f in train dev test clip_durations; do
    fetch_one "$BASE/transcript/$l/$f.tsv" "$ROOT/$l/$f.tsv"
  done
done
