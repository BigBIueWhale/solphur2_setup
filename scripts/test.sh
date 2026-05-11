#!/usr/bin/env bash
# scripts/test.sh — automated smoke + headline tests against the live stack.
#
# Assumes scripts/up.sh has already brought everything up and `/healthz`
# returns ok. Runs:
#   1. SMOKE: 720p × 5s, fast mode, no enhancer (deterministic seed).
#      Validates the workflow plumbing end-to-end. Expected: HTTP 200 + a
#      ≥1 MiB MP4 in ~2 minutes on RTX 5090.
#   2. HEADLINE: 1080p × 20s, quality mode, no enhancer (deterministic seed).
#      The user-facing target. Expected: HTTP 200 + a ≥5 MiB MP4 in
#      ~25–35 min on RTX 5090.
#
# Each request includes a fixed seed for reproducibility. Output MP4s land in
# ./outputs/ and are also copied to ./test_artifacts/ with descriptive names.
#
# Flags:
#     --smoke-only          run only the smoke test
#     --headline-only       run only the headline test
#     --enhance             include the prompt enhancer in both tests
#     --mode=fast|quality   override the default mode (smoke=fast, headline=quality)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

API_URL="${API_URL:-http://127.0.0.1:8000}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/test_artifacts}"
mkdir -p "$ARTIFACTS_DIR"

log()  { printf '[\033[36msolphur2-test\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[\033[31msolphur2-test\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; exit 1; }

DO_SMOKE=1
DO_HEADLINE=1
ENHANCE="false"
SMOKE_MODE="fast"
HEADLINE_MODE="quality"

for arg in "$@"; do
    case "$arg" in
        --smoke-only)     DO_HEADLINE=0 ;;
        --headline-only)  DO_SMOKE=0 ;;
        --enhance)        ENHANCE="true" ;;
        --mode=fast)      SMOKE_MODE="fast"; HEADLINE_MODE="fast" ;;
        --mode=quality)   SMOKE_MODE="quality"; HEADLINE_MODE="quality" ;;
        *) echo "scripts/test.sh: unknown flag: $arg" >&2; exit 64 ;;
    esac
done

PROMPT="a cinematic close-up of a foggy cobblestone alley at dawn, warm amber lamplight reflecting on wet stones, slow tracking shot, shallow depth of field"

# Pre-flight: API must be reachable.
log "pre-flight: GET ${API_URL}/healthz"
hz="$(curl -fsS --max-time 5 "${API_URL}/healthz" || true)"
[[ -z "$hz" ]] && fail "API not reachable at ${API_URL}; run scripts/up.sh first"
echo "$hz" | grep -q '"ok":true' || fail "API /healthz reports unhealthy: $hz"

run_test() {
    local name="$1" mode="$2" width="$3" height="$4" duration="$5" seed="$6" timeout_s="$7"
    local out_mp4="${ARTIFACTS_DIR}/${name}_${width}x${height}_${duration}s_${mode}_seed${seed}.mp4"
    local out_meta="${out_mp4%.mp4}.headers.txt"
    rm -f "$out_mp4" "$out_meta"
    log "$name: ${width}x${height} × ${duration}s × 24fps, mode=$mode, seed=$seed, max ${timeout_s}s"
    local t0
    t0=$(date +%s)
    local body
    body=$(cat <<JSON
{
  "prompt": $(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
  "duration_seconds": ${duration},
  "fps": 24,
  "width": ${width},
  "height": ${height},
  "mode": "${mode}",
  "seed": ${seed},
  "enhance_prompt": ${ENHANCE}
}
JSON
)
    local status size elapsed
    status=$(curl -sS -X POST "${API_URL}/generate" \
        -H 'Content-Type: application/json' \
        -d "$body" \
        -D "$out_meta" \
        -o "$out_mp4" \
        --max-time "$timeout_s" \
        -w '%{http_code}')
    elapsed=$(( $(date +%s) - t0 ))
    size=$(stat -c %s "$out_mp4" 2>/dev/null || echo 0)
    if [[ "$status" != "200" ]]; then
        log "    FAIL: http=$status elapsed=${elapsed}s size=${size} bytes"
        log "    response head:"
        head -c 600 "$out_mp4" || true
        echo
        return 1
    fi
    # Validate the output really is an MP4
    if ! file -b "$out_mp4" | grep -qi "MP4\|ISO Media"; then
        log "    FAIL: HTTP 200 but file is not MP4: $(file -b "$out_mp4")"
        return 1
    fi
    log "    OK   http=$status elapsed=${elapsed}s size=$((size/1024/1024)) MiB → $out_mp4"
}

if [[ "$DO_SMOKE" -eq 1 ]]; then
    run_test smoke "$SMOKE_MODE" 1280 704 5 42 1800
fi

if [[ "$DO_HEADLINE" -eq 1 ]]; then
    run_test headline "$HEADLINE_MODE" 1920 1088 20 42 3000
fi

log "all requested tests passed."
log "artifacts in: $ARTIFACTS_DIR"
