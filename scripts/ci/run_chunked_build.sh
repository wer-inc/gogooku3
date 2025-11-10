#!/usr/bin/env bash
#
# CI-friendly orchestrator for the gogooku5 chunked dataset pipeline.
# Runs quarterly chunk builds followed by merge + status validation so
# long date ranges can execute safely inside automation (Jenkins/GitHub).

set -euo pipefail

echoerr() { printf "%s\n" "$*" >&2; }
log() { printf "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] %s\n" "$*"; }

require_var() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echoerr "Missing required environment variable: ${name}"
        exit 1
    fi
}

CHUNK_START=${CHUNK_START:-}
CHUNK_END=${CHUNK_END:-}
CHUNK_RESUME=${CHUNK_RESUME:-1}
CHUNK_FORCE=${CHUNK_FORCE:-0}
CHUNK_LATEST_ONLY=${CHUNK_LATEST_ONLY:-0}
CHUNK_JOBS=${CHUNK_JOBS:-1}
CHUNK_ALLOW_PARTIAL=${CHUNK_ALLOW_PARTIAL:-0}
CHUNK_MONTHS=${CHUNK_MONTHS:-}
CHUNK_CHUNKS_DIR=${CHUNK_CHUNKS_DIR:-output/chunks}
CHUNK_OUTPUT_DIR=${CHUNK_OUTPUT_DIR:-}
CHUNK_DRY_RUN=${CHUNK_DRY_RUN:-0}

require_var CHUNK_START
require_var CHUNK_END

run_make() {
    log "Running: $*"
    eval "$@"
}

build_chunks() {
    local cmd="make build-chunks START=${CHUNK_START} END=${CHUNK_END}"
    [[ "${CHUNK_RESUME}" == "1" ]] && cmd+=" RESUME=1"
    [[ "${CHUNK_FORCE}" == "1" ]] && cmd+=" FORCE=1"
    [[ "${CHUNK_LATEST_ONLY}" == "1" ]] && cmd+=" LATEST=1"
    [[ -n "${CHUNK_JOBS}" ]] && cmd+=" JOBS=${CHUNK_JOBS}"
    [[ -n "${CHUNK_MONTHS}" ]] && cmd+=" CHUNK_MONTHS=${CHUNK_MONTHS}"
    [[ "${CHUNK_DRY_RUN}" == "1" ]] && cmd+=" DRY_RUN=1"
    run_make "${cmd}"
}

merge_chunks() {
    local cmd="make merge-chunks"
    [[ -n "${CHUNK_CHUNKS_DIR}" ]] && cmd+=" CHUNKS_DIR=${CHUNK_CHUNKS_DIR}"
    [[ -n "${CHUNK_OUTPUT_DIR}" ]] && cmd+=" OUTPUT_DIR=${CHUNK_OUTPUT_DIR}"
    [[ "${CHUNK_ALLOW_PARTIAL}" == "1" ]] && cmd+=" ALLOW_PARTIAL=1"
    run_make "${cmd}"
}

validate_chunk_status() {
    if [[ ! -d "${CHUNK_CHUNKS_DIR}" ]]; then
        echoerr "Chunks directory not found: ${CHUNK_CHUNKS_DIR}"
        exit 1
    fi
    python3 - "$CHUNK_CHUNKS_DIR" "$CHUNK_ALLOW_PARTIAL" <<'PY'
import json
import sys
from pathlib import Path

chunks_dir = Path(sys.argv[1])
allow_partial = sys.argv[2] == "1"
status_files = sorted(chunks_dir.glob("*/status.json"))

if not status_files:
    print(f"[ERROR] No status.json found under {chunks_dir}", file=sys.stderr)
    sys.exit(1)

incomplete = []
completed = []
for status_path in status_files:
    data = json.loads(status_path.read_text(encoding="utf-8"))
    state = data.get("state", "unknown")
    chunk_id = data.get("chunk_id") or status_path.parent.name
    if state != "completed":
        incomplete.append((chunk_id, state))
    else:
        completed.append(chunk_id)

print(f"[INFO] Completed chunks: {len(completed)}")
for cid in completed:
    print(f"  ✓ {cid}")

if incomplete:
    print(f"[WARN] Incomplete chunks: {len(incomplete)}", file=sys.stderr)
    for chunk_id, state in incomplete:
        print(f"  ✗ {chunk_id}: state={state}", file=sys.stderr)
    if not allow_partial:
        print("[ERROR] Incomplete chunks detected but ALLOW_PARTIAL=0", file=sys.stderr)
        sys.exit(1)
PY
}

log "Chunk build start=${CHUNK_START} end=${CHUNK_END}"
build_chunks

if [[ "${CHUNK_DRY_RUN}" == "1" ]]; then
    log "Chunk planner dry-run completed – skipping merge/validation."
    exit 0
fi

validate_chunk_status
merge_chunks
log "Chunk merge finished successfully."
