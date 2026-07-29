#!/usr/bin/env bash
# v2 recomputation runbook. Run from tts-benchmark/ with the venv active.
#
#   bash scripts/runbook.sh all          # everything, in order
#   bash scripts/runbook.sh replay wer   # selected phases
#
# Phase R0 of the original plan (offline re-aggregation of the stored v1
# per-utterance parquets) is absent because those artifacts do not exist: they
# are written inside tts-bench-v1/, which is gitignored, and the published
# dataset carries only markdown. Shipping them is itself a v2 deliverable.
# Everything below therefore starts from audio.
set -uo pipefail

PY=${PY:-python}
LANGS_WHISPER="en-US es-ES es-MX nl-NL pt-BR"
LANGS_ALL="en-US es-ES es-MX nl-NL pt-BR ky"
ENCODERS="wavlm_sv wavlm_ft ecapa"
LOG=${LOG:-logs}
mkdir -p "$LOG"

run() {   # run <logfile> <command...>
  local name=$1; shift
  echo "--- $name" | tee -a "$LOG/$name.log"
  "$@" >>"$LOG/$name.log" 2>&1 || echo "!! FAILED: $name :: $*" | tee -a "$LOG/failures.log"
}

phase_replay() {
  # The attribution ladder. Each pass changes exactly one thing on identical
  # audio, so the v1 -> v2 delta gets a cause per step instead of one lump.
  # batch=8 throughout because v1 used 8 and Whisper's output depends on batch
  # composition through padding.
  for L in $LANGS_WHISPER; do
    run "replay_legacy" $PY msbench.cli.wer --lang "$L" --audio gt --asr whisper \
        --batch 8 --repetition-penalty 1.1 --chunk-length-s 30 --tag legacy
    run "replay_nopen"  $PY msbench.cli.wer --lang "$L" --audio gt --asr whisper \
        --batch 8 --chunk-length-s 30 --tag nopen
  done
}

phase_wer() {
  # R1 - Whisper, v2 settings: no repetition penalty (D-02), short-form path (D-10).
  for L in $LANGS_WHISPER; do
    run "r1_whisper" $PY msbench.cli.wer --lang "$L" --audio gt --asr whisper --batch 8
  done
  # R2 - MMS: CTC without a language model, all six subsets including ky.
  for L in $LANGS_ALL; do
    run "r2_mms" $PY msbench.cli.wer --lang "$L" --audio gt --asr mms --batch 8
  done
  # R2b - GigaAM: the v1 reference for ky, the control in the D-27 cross-check.
  run "r2b_gigaam" $PY msbench.cli.wer --lang ky --audio gt --asr gigaam
}

phase_sim() {
  # R4 - three encoders, human anchor and impostor floor.
  for L in $LANGS_ALL; do
    for E in $ENCODERS; do
      run "r4_sim" $PY msbench.cli.sim --lang "$L" --mode anchor --encoder "$E"
      run "r4_sim" $PY msbench.cli.sim --lang "$L" --mode floor  --encoder "$E" --pairs 5000
    done
  done
}

phase_quality() {
  # R5 - DNSMOS and the QC battery on all three audio streams (D-09).
  # 12 workers, not the 48 in the plan: this machine has 16 cores and 31 GB.
  for A in prompt gt sim_ref; do
    run "r5_quality" $PY msbench.cli.quality --audio "$A" --workers 12
  done
}

phase_report() {
  run "r6_report" $PY msbench.cli.replay
  run "r6_report" $PY eval/report2.py --all
}

for phase in "$@"; do
  case "$phase" in
    all)      phase_replay; phase_wer; phase_sim; phase_quality; phase_report ;;
    replay)   phase_replay ;;
    wer)      phase_wer ;;
    sim)      phase_sim ;;
    quality)  phase_quality ;;
    report)   phase_report ;;
    *) echo "unknown phase: $phase" >&2; exit 2 ;;
  esac
done
echo "done."
