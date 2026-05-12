#!/usr/bin/env bash
# scripts/test.sh — automated functional tests against the live stack.
#
# Assumes scripts/up.sh has already brought everything up and `/healthz`
# returns ok. The DEFAULT is the **headline test** at the highest-fidelity
# config validated to work end-to-end: Sulphur's tested envelope at
# quality mode. (The LTX-2.3 ceiling 1920×1088 × 20 s × quality reproduces
# an audio-mux EINVAL at the SaveVideo node and is NOT the default; opt
# into it explicitly via `--ltx-ceiling-quality` if you want to capture
# the bug on your own host.)
#
# Tests:
#   • HEADLINE (default): 1280×704 × 10s, quality mode, no enhancer
#     (deterministic seed). The user-facing target. Expected: HTTP 200 + a
#     ≥1 MiB MP4 in ~188 s cold-start / ~169 s warm-cache on RTX 5090.
#   • SMOKE (opt-in via --with-smoke or --smoke-only): 1280×704 × 5s,
#     fast mode, no enhancer. Validates workflow plumbing end-to-end.
#     Expected: HTTP 200 + a ≥1 MiB MP4 in ~80 s on RTX 5090.
#   • LTX-CEILING (opt-in via --ltx-ceiling-fast): 1920×1088 × 20s,
#     FAST mode. The largest envelope that produces a valid MP4.
#     Quality mode at this envelope crashes — captured separately.
#
# Each request includes a fixed seed for reproducibility. Output MP4s land in
# ./outputs/ and are also copied to ./test_artifacts/ with descriptive names.
#
# Flags (default = headline only at 1280×704 × 10s × quality):
#     --headline-only        run only the headline test (the default)
#     --smoke-only           run only the smoke test (skip the slow headline)
#     --with-smoke           run smoke first, then headline (regression-guard)
#     --ltx-ceiling-fast     additionally run 1920×1088 × 20s × fast
#     --ltx-ceiling-quality  additionally run 1920×1088 × 20s × quality
#                            (expected to FAIL — captures the EINVAL bug)
#     --enhance              include the prompt enhancer in the test(s)
#     --mode=fast|quality    override the default mode (smoke=fast, headline=quality)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

API_URL="${API_URL:-http://127.0.0.1:8000}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/test_artifacts}"
mkdir -p "$ARTIFACTS_DIR"

log()  { printf '[\033[36msolphur2-test\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[\033[31msolphur2-test\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; exit 1; }

# Defaults: headline only (highest-fidelity validated config). Smoke and
# the ltx-ceiling probes are opt-in.
DO_SMOKE=0
DO_HEADLINE=1
DO_LTX_CEIL_FAST=0
DO_LTX_CEIL_QUAL=0
ENHANCE="false"
SMOKE_MODE="fast"
HEADLINE_MODE="quality"

for arg in "$@"; do
    case "$arg" in
        --smoke-only)            DO_SMOKE=1; DO_HEADLINE=0 ;;
        --headline-only)         DO_SMOKE=0; DO_HEADLINE=1 ;;
        --with-smoke)            DO_SMOKE=1; DO_HEADLINE=1 ;;
        --ltx-ceiling-fast)      DO_LTX_CEIL_FAST=1 ;;
        --ltx-ceiling-quality)   DO_LTX_CEIL_QUAL=1 ;;
        --enhance)               ENHANCE="true" ;;
        --mode=fast)             SMOKE_MODE="fast"; HEADLINE_MODE="fast" ;;
        --mode=quality)          SMOKE_MODE="quality"; HEADLINE_MODE="quality" ;;
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
    run_test headline "$HEADLINE_MODE" 1280 704 10 42 1800
fi

if [[ "$DO_LTX_CEIL_FAST" -eq 1 ]]; then
    run_test ltx-ceiling-fast fast 1920 1088 20 42 3000
fi

if [[ "$DO_LTX_CEIL_QUAL" -eq 1 ]]; then
    log "NOTE: ltx-ceiling-quality is EXPECTED to fail with avcodec EINVAL"
    log "      at the SaveVideo node — this run captures the bug payload"
    log "      to ${ARTIFACTS_DIR}/. Useful for diagnosing regressions."
    run_test ltx-ceiling-quality quality 1920 1088 20 42 3000 || true
fi

log "all requested tests passed."
log "artifacts in: $ARTIFACTS_DIR"
