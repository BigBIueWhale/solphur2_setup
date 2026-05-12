#!/usr/bin/env bash
# scripts/measure.sh — instrument ONE /generate request end-to-end:
# rebuild if needed, sample nvidia-smi + cgroup v2 + /proc at 1 Hz, write
# the resulting MP4 + headers + CSVs + a human-readable summary under
# bench_runs/.
#
# DEFAULT BEHAVIOUR is the **highest-fidelity config that's empirically
# validated to work end-to-end** — i.e. the Sulphur-tested envelope at
# quality mode: `1280 × 704 × 10 s × 24 fps × mode=quality`. This is what
# the README's headline wall-clock + VRAM numbers come from.
#
# Why not default to the LTX-2.3 ceiling (1920×1088 × 20 s × quality)?
# Empirically that envelope reproduces an `avcodec_send_frame() returned
# 22` (EINVAL) failure at the SaveVideo audio mux. The bug is independent
# of stage-2 distill LoRA stacking (control runs with both wirings fail
# identically) and is presumed to be audio-VAE amplitude drift at the
# longer / higher-resolution latent. Until that's fixed, the highest-
# quality validated envelope is the Sulphur-tested 1280×704 × 10 s.
# Override flags let you measure other envelopes — use
# `--ltx-ceiling-fast` for 1080p × 20 s in fast mode (the largest known-
# working envelope) or `--ltx-ceiling-quality` to reproduce the EINVAL
# bug capture locally.
#
# Why this exists separately from scripts/test.sh and scripts/bench.py:
#   • scripts/test.sh runs functional pass/fail smoke + headline tests; no
#     telemetry capture.
#   • scripts/bench.py sweeps the (resolution × duration × mode) envelope
#     across multiple runs; only samples VRAM (no RAM, no per-phase split,
#     enhancer disabled).
#   • This script measures EXACTLY ONE config — with the prompt enhancer
#     active, per-phase wall-clock split, all GPU telemetry, and per-
#     container CPU+RAM — and writes everything to a timestamped subdir.
#
# Steps (each delegated — no duplicated docker/compose commands here):
#   1. scripts/build.sh                  → rebuild images (layer cache; the
#                                           API image rebuilds quickly when
#                                           api/server.py changes).
#   2. docker compose up -d              → apply any new image.
#   3. wait for /healthz                 → don't measure during startup.
#   4. sample nvidia-smi + cgroup v2     → 1 s cadence, background.
#   5. POST /generate                    → at the requested envelope.
#   6. aggregate peaks                   → from CSVs.
#   7. print summary                     → including the per-phase X-Solphur2
#                                           headers the API returned.
#
# Usage:
#     bash scripts/measure.sh "<prompt>"                          # 1280×704 × 10 s × quality (highest validated)
#     bash scripts/measure.sh "<prompt>" --skip-build             # same, reuse current images
#     bash scripts/measure.sh "<prompt>" --ltx-ceiling-fast       # 1920×1088 × 20 s × fast (largest known-working envelope)
#     bash scripts/measure.sh "<prompt>" --ltx-ceiling-quality    # 1920×1088 × 20 s × quality (KNOWN FAILURE; captures the EINVAL)
#     bash scripts/measure.sh "<prompt>" --width 1280 --height 704 --duration 10 --mode quality
#     bash scripts/measure.sh "<prompt>" --seed 42 --no-enhance   # deterministic, skip enhancer
#
# The prompt is a REQUIRED positional argument — there is intentionally no
# default. Defaulting would either ship a SFW prompt (silently bypassing
# Sulphur's uncensored capability and giving misleading timing data) or ship
# an NSFW prompt that nobody asked for. Make the caller think about what
# they're measuring.
#
# Example:
#     bash scripts/measure.sh \
#         "a beautiful nude woman lying on satin sheets, soft golden light, slow cinematic tracking camera, 35mm"
#
# Output: ./bench_runs/measure_<W>x<H>_<DUR>s_<MODE>_YYYYMMDD-HHMMSS/
#   ├── run.mp4                           → the generated video
#   ├── headers.txt                       → all X-Solphur2-* response headers
#   ├── gpu.csv                           → 1s-cadence nvidia-smi
#   ├── ram.csv                           → 1s-cadence cgroup v2 (memory.current + cpu.stat)
#   ├── host.csv                          → 1s-cadence host CPU + RAM via /proc
#   ├── api.log                           → solphur2-api docker logs (phase lines)
#   └── summary.txt                       → human-readable peak/timing table

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

API_URL="http://127.0.0.1:8000"

log()  { printf '[\033[36msolphur2-measure\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[\033[31msolphur2-measure\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; exit 1; }

usage() {
    cat >&2 <<EOF
usage: bash scripts/measure.sh "<prompt>" [options]

  <prompt>            REQUIRED positional argument. Free-text prompt for /generate.
                      No default — see scripts/measure.sh header for why.

Build control:
  --skip-build        Reuse the current Docker images. Default rebuilds (incremental).
  --no-cache          Full rebuild from scratch (rare; use when upstream wheels
                      changed without a version pin bumping).

Envelope (default = highest validated quality = 1280×704 × 10 s × quality):
  --ltx-ceiling-fast    Shortcut for 1920×1088 × 20 s × fast (largest envelope
                        that produces a valid MP4 at the LTX-2.3 ceiling).
  --ltx-ceiling-quality Shortcut for 1920×1088 × 20 s × quality (KNOWN FAILURE
                        — reproduces avcodec_send_frame() EINVAL at SaveVideo;
                        useful only for capturing the bug to bench_runs/).
  --width N             Override width (server default: 1280).
  --height N            Override height (server default: 704).
  --duration N          Override duration in seconds (server default: 10.0).
  --fps N               Override fps (server default: 24).
  --mode quality|fast   Override mode (server default: quality).
  --seed N              Override the per-request random seed (for reproducibility).
  --no-enhance          Skip the Sulphur prompt enhancer (saves ~25-35 s).
EOF
    exit 64
}

PROMPT=""
DO_BUILD=1
BUILD_FLAGS=()

# Default envelope = highest validated quality = Sulphur's tested envelope
# at quality mode. The LTX-2.3 ceiling (1920×1088 × 20 s) currently
# reproduces an audio-mux EINVAL at quality mode — see the script header
# for details. `--ltx-ceiling-fast` opts into the largest known-working
# config.
WIDTH=1280
HEIGHT=704
DURATION=10
FPS=24
MODE="quality"
SEED=""
ENHANCE_FLAG=""

while (( $# > 0 )); do
    case "$1" in
        --skip-build)            DO_BUILD=0 ;;
        --no-cache)              BUILD_FLAGS+=("--no-cache") ;;
        --ltx-ceiling-fast)      WIDTH=1920; HEIGHT=1088; DURATION=20; MODE="fast" ;;
        --ltx-ceiling-quality)   WIDTH=1920; HEIGHT=1088; DURATION=20; MODE="quality" ;;
        --width)                 shift; WIDTH="$1" ;;
        --height)                shift; HEIGHT="$1" ;;
        --duration)              shift; DURATION="$1" ;;
        --fps)                   shift; FPS="$1" ;;
        --mode)                  shift; MODE="$1" ;;
        --seed)                  shift; SEED="$1" ;;
        --no-enhance)            ENHANCE_FLAG="--no-enhance" ;;
        -h|--help)               usage ;;
        --*)                     echo "unknown flag: $1" >&2; usage ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
            else
                echo "unexpected extra positional argument: $1" >&2
                usage
            fi
            ;;
    esac
    shift
done

[[ -n "$PROMPT" ]] || { echo "error: <prompt> is required." >&2; usage; }

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$REPO_ROOT/bench_runs/measure_${WIDTH}x${HEIGHT}_${DURATION}s_${MODE}_${STAMP}"
mkdir -p "$OUT_DIR"

log "envelope: ${WIDTH}x${HEIGHT} × ${DURATION}s × ${FPS}fps × ${MODE}${SEED:+ × seed=${SEED}}${ENHANCE_FLAG:+ × enhance=off}"
log "out-dir : $OUT_DIR"

# --- 1. Rebuild images ---------------------------------------------------
if [[ "$DO_BUILD" -eq 1 ]]; then
    log "rebuilding images via scripts/build.sh ${BUILD_FLAGS[*]:-}"
    bash scripts/build.sh "${BUILD_FLAGS[@]}"
else
    log "skipping rebuild (--skip-build)"
fi

# --- 2. Apply new image + wait healthy -----------------------------------
log "applying new image: docker compose up -d (recreates containers if image changed)"
docker compose --env-file versions.env up -d

log "waiting for /healthz (≤5 min)…"
deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < deadline )); do
    if curl -fsS --max-time 3 "$API_URL/healthz" 2>/dev/null | grep -q '"ok":true'; then
        log "healthz ok"
        break
    fi
    sleep 3
done
curl -fsS --max-time 3 "$API_URL/healthz" 2>/dev/null | grep -q '"ok":true' \
    || fail "/healthz did not report ok within 5 min; check 'docker compose logs'"

# --- 3. Start the samplers (background) ----------------------------------
GPU_CSV="$OUT_DIR/gpu.csv"
RAM_CSV="$OUT_DIR/ram.csv"
HOST_CSV="$OUT_DIR/host.csv"

log "starting samplers (1 s cadence) → gpu.csv, ram.csv, host.csv"

echo "timestamp_iso,gpu_used_mib,gpu_free_mib,util_gpu_pct,util_mem_pct,power_w,clock_sm_mhz,temp_c" > "$GPU_CSV"
nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 1000 >> "$GPU_CSV" 2>&1 &
GPU_PID=$!

COMFY_CID="$(docker inspect --format '{{.Id}}' solphur2-comfyui)"
ENH_CID="$(  docker inspect --format '{{.Id}}' solphur2-enhancer)"
API_CID="$(  docker inspect --format '{{.Id}}' solphur2-api)"
CG_BASE="/sys/fs/cgroup/system.slice"
COMFY_MEM="$CG_BASE/docker-$COMFY_CID.scope/memory.current"
COMFY_CPU="$CG_BASE/docker-$COMFY_CID.scope/cpu.stat"
ENH_MEM="$CG_BASE/docker-$ENH_CID.scope/memory.current"
ENH_CPU="$CG_BASE/docker-$ENH_CID.scope/cpu.stat"
API_MEM="$CG_BASE/docker-$API_CID.scope/memory.current"
API_CPU="$CG_BASE/docker-$API_CID.scope/cpu.stat"
for f in "$COMFY_MEM" "$COMFY_CPU" "$ENH_MEM" "$ENH_CPU" "$API_MEM" "$API_CPU"; do
    [[ -r "$f" ]] || fail "cgroup file not readable: $f"
done

echo "timestamp_iso,comfyui_mem_mib,comfyui_cpu_pct,enhancer_mem_mib,enhancer_cpu_pct,api_mem_mib,api_cpu_pct" > "$RAM_CSV"
(
    read_cpu_usec() { awk '/^usage_usec/ {print $2; exit}' "$1"; }
    prev_comfy_us=$(read_cpu_usec "$COMFY_CPU")
    prev_enh_us=$(  read_cpu_usec "$ENH_CPU")
    prev_api_us=$(  read_cpu_usec "$API_CPU")
    prev_ns=$(date +%s%N)
    sleep 1
    while true; do
        ts="$(date -Iseconds)"
        now_ns=$(date +%s%N)
        d_ns=$(( now_ns - prev_ns ))
        prev_ns=$now_ns

        comfy_mib=$(( $(<"$COMFY_MEM") / 1048576 ))
        enh_mib=$((   $(<"$ENH_MEM")   / 1048576 ))
        api_mib=$((   $(<"$API_MEM")   / 1048576 ))

        cur_comfy_us=$(read_cpu_usec "$COMFY_CPU")
        cur_enh_us=$(  read_cpu_usec "$ENH_CPU")
        cur_api_us=$(  read_cpu_usec "$API_CPU")
        d_comfy_us=$(( cur_comfy_us - prev_comfy_us ))
        d_enh_us=$((   cur_enh_us   - prev_enh_us   ))
        d_api_us=$((   cur_api_us   - prev_api_us   ))
        prev_comfy_us=$cur_comfy_us
        prev_enh_us=$cur_enh_us
        prev_api_us=$cur_api_us
        cpus_pcts=$(awk -v c="$d_comfy_us" -v e="$d_enh_us" -v a="$d_api_us" -v dn="$d_ns" \
            'BEGIN {
                if (dn <= 0) { print "0.0,0.0,0.0"; exit }
                printf "%.1f,%.1f,%.1f", c*100000/dn, e*100000/dn, a*100000/dn
            }')
        IFS=',' read -r comfy_pct enh_pct api_pct <<< "$cpus_pcts"

        printf '%s,%d,%s,%d,%s,%d,%s\n' "$ts" \
            "$comfy_mib" "$comfy_pct" \
            "$enh_mib"   "$enh_pct" \
            "$api_mib"   "$api_pct" \
            >> "$RAM_CSV"
        sleep 1
    done
) &
RAM_PID=$!

echo "timestamp_iso,host_cpu_pct,host_mem_used_mib,host_mem_avail_mib,host_swap_used_mib" > "$HOST_CSV"
(
    prev_idle=0; prev_total=0
    while true; do
        ts="$(date -Iseconds)"
        read cpu user nice system idle iowait irq softirq steal _ < /proc/stat
        total=$((user + nice + system + idle + iowait + irq + softirq + steal))
        d_total=$(( total - prev_total ))
        d_idle=$(( idle  - prev_idle  ))
        if (( prev_total > 0 && d_total > 0 )); then
            cpu_pct=$(awk -v dt=$d_total -v di=$d_idle 'BEGIN{ printf "%.1f", 100*(dt-di)/dt }')
        else
            cpu_pct="0.0"
        fi
        prev_idle=$idle
        prev_total=$total
        mem_used_kb=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t-a}' /proc/meminfo)
        mem_avail_kb=$(awk '/MemAvailable/{print $2; exit}' /proc/meminfo)
        swap_used_kb=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
        printf '%s,%s,%d,%d,%d\n' "$ts" "$cpu_pct" \
            "$((mem_used_kb/1024))" "$((mem_avail_kb/1024))" "$((swap_used_kb/1024))" \
            >> "$HOST_CSV"
        sleep 1
    done
) &
HOST_PID=$!

cleanup() {
    for pid in "$GPU_PID" "$RAM_PID" "$HOST_PID"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "$GPU_PID" "$RAM_PID" "$HOST_PID"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- 4. Fire one POST /generate at the requested envelope ----------------
log "POST $API_URL/generate via scripts/_measure_client.py (envelope: ${WIDTH}x${HEIGHT} × ${DURATION}s × ${MODE})"

T0="$(date +%s)"
CLIENT_RC=0
CLIENT_ARGS=(
    --prompt          "$PROMPT"
    --out-dir         "$OUT_DIR"
    --api-url         "$API_URL"
    --timeout-seconds 1800
    --width           "$WIDTH"
    --height          "$HEIGHT"
    --duration-seconds "$DURATION"
    --fps             "$FPS"
    --mode            "$MODE"
)
if [[ -n "$SEED" ]];          then CLIENT_ARGS+=(--seed "$SEED"); fi
if [[ -n "$ENHANCE_FLAG" ]];  then CLIENT_ARGS+=("$ENHANCE_FLAG"); fi
python3 scripts/_measure_client.py "${CLIENT_ARGS[@]}" || CLIENT_RC=$?
T1="$(date +%s)"
WALL=$(( T1 - T0 ))

cleanup
trap - EXIT

if (( CLIENT_RC != 0 )); then
    log "FAIL: _measure_client.py exited $CLIENT_RC after ${WALL}s; first 600 bytes of body:"
    head -c 600 "$OUT_DIR/run.mp4" 2>/dev/null || true; echo
    exit 1
fi

HTTP=$(awk 'NR==1 {print $2; exit}' "$OUT_DIR/headers.txt")
[[ "$HTTP" == "200" ]] || fail "expected HTTP 200 in headers; got: $HTTP"

# --- 5. Capture API logs for the same window (phase lines) ---------------
docker logs --since "$((WALL + 10))s" solphur2-api > "$OUT_DIR/api.log" 2>&1 || true

# --- 6. Aggregate peaks --------------------------------------------------
GPU_PEAK=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$GPU_CSV")
GPU_BASE=$(awk -F, 'NR==2 {print $2+0; exit}' "$GPU_CSV")
GPU_UTIL_PEAK=$(awk -F, 'NR>1 && $4+0 > max {max=$4+0} END {print max+0}' "$GPU_CSV")
GPU_UTIL_AVG=$( awk -F, 'NR>1 {s+=$4+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$GPU_CSV")
POWER_PEAK=$(  awk -F, 'NR>1 && $6+0 > max {max=$6+0} END {printf "%.1f", max+0}' "$GPU_CSV")
POWER_AVG=$(   awk -F, 'NR>1 {s+=$6+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$GPU_CSV")
CLOCK_PEAK=$(  awk -F, 'NR>1 && $7+0 > max {max=$7+0} END {print max+0}' "$GPU_CSV")
TEMP_PEAK=$(   awk -F, 'NR>1 && $8+0 > max {max=$8+0} END {print max+0}' "$GPU_CSV")

COMFY_PEAK=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$RAM_CSV")
ENH_PEAK=$(  awk -F, 'NR>1 && $4+0 > max {max=$4+0} END {print max+0}' "$RAM_CSV")
API_PEAK=$(  awk -F, 'NR>1 && $6+0 > max {max=$6+0} END {print max+0}' "$RAM_CSV")
COMFY_CPU_PEAK=$(awk -F, 'NR>1 && $3+0 > max {max=$3+0} END {printf "%.1f", max+0}' "$RAM_CSV")
ENH_CPU_PEAK=$(  awk -F, 'NR>1 && $5+0 > max {max=$5+0} END {printf "%.1f", max+0}' "$RAM_CSV")
API_CPU_PEAK=$(  awk -F, 'NR>1 && $7+0 > max {max=$7+0} END {printf "%.1f", max+0}' "$RAM_CSV")
RAM_STACK_PEAK=$( awk -F, 'NR>1 {s=$2+$4+$6; if (s>max) max=s} END {print max+0}' "$RAM_CSV")
CPU_STACK_PEAK=$( awk -F, 'NR>1 {s=$3+$5+$7; if (s>max) max=s} END {printf "%.1f", max+0}' "$RAM_CSV")

HOST_CPU_PEAK=$(   awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {printf "%.1f", max+0}' "$HOST_CSV")
HOST_CPU_AVG=$(    awk -F, 'NR>1 {s+=$2+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$HOST_CSV")
HOST_MEM_PEAK=$(   awk -F, 'NR>1 && $3+0 > max {max=$3+0} END {print max+0}' "$HOST_CSV")
HOST_MEM_AVAIL_MIN=$( awk -F, 'NR>1 {if (NR==2 || $4+0 < min) min=$4+0} END {print min+0}' "$HOST_CSV")
HOST_SWAP_PEAK=$(  awk -F, 'NR>1 && $5+0 > max {max=$5+0} END {print max+0}' "$HOST_CSV")

PHASE_ENHANCE=$(awk -F': ' 'tolower($1)=="x-solphur2-phaseenhanceseconds"  {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
PHASE_SUBMIT=$( awk -F': ' 'tolower($1)=="x-solphur2-phasesubmitseconds"   {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
PHASE_COMFY=$(  awk -F': ' 'tolower($1)=="x-solphur2-phasecomfyrunseconds" {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
ELAPSED=$(      awk -F': ' 'tolower($1)=="x-solphur2-elapsedseconds"       {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
SEED_HDR=$(     awk -F': ' 'tolower($1)=="x-solphur2-seed"                 {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
RES=$(          awk -F': ' 'tolower($1)=="x-solphur2-resolution"           {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
FRAMES=$(       awk -F': ' 'tolower($1)=="x-solphur2-frames"               {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
MODE_HDR=$(     awk -F': ' 'tolower($1)=="x-solphur2-mode"                 {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')

MP4_SIZE=$(stat -c %s "$OUT_DIR/run.mp4")

# --- 7. Print + persist summary ------------------------------------------
SUMMARY="$OUT_DIR/summary.txt"
{
    echo "solphur2 measurement — ${STAMP}"
    echo "==========================================================="
    echo "Config:"
    echo "  resolution  : $RES"
    echo "  frames      : $FRAMES"
    echo "  mode        : $MODE_HDR"
    echo "  seed        : $SEED_HDR"
    echo "  prompt      : $PROMPT"
    echo
    echo "Wall-clock breakdown (from API response headers):"
    printf "  enhance     : %6s s   (Sulphur Qwen3.5-9B enhancer, CPU only)\n" "$PHASE_ENHANCE"
    printf "  submit      : %6s s   (workflow POST to ComfyUI)\n"            "$PHASE_SUBMIT"
    printf "  comfy_run   : %6s s   (sampling + upsample + VAE + mux)\n"     "$PHASE_COMFY"
    printf "  ── total    : %6s s   (== %d s curl wall-clock)\n"             "$ELAPSED" "$WALL"
    echo
    echo "GPU telemetry (nvidia-smi, 1s cadence):"
    printf "  VRAM baseline : %6d MiB  (request start; container warm)\n" "$GPU_BASE"
    printf "  VRAM peak     : %6d MiB / 32607 MiB (%.1f%%)\n"             "$GPU_PEAK" "$(awk -v p="$GPU_PEAK" 'BEGIN{print 100*p/32607}')"
    printf "  VRAM delta    : %6d MiB  (peak − baseline)\n"               "$((GPU_PEAK - GPU_BASE))"
    printf "  GPU util peak : %6d %%   (compute utilization)\n"           "$GPU_UTIL_PEAK"
    printf "  GPU util avg  : %6s %%\n"                                   "$GPU_UTIL_AVG"
    printf "  Power peak    : %6s W\n"                                    "$POWER_PEAK"
    printf "  Power avg     : %6s W\n"                                    "$POWER_AVG"
    printf "  SM clock peak : %6d MHz\n"                                  "$CLOCK_PEAK"
    printf "  Temp peak     : %6d °C\n"                                   "$TEMP_PEAK"
    echo
    echo "Per-container CPU + RAM (cgroup v2 memory.current + cpu.stat, 1s cadence):"
    printf "  comfyui  : RAM peak %6.0f MiB,  CPU peak %6s %%\n"          "$COMFY_PEAK" "$COMFY_CPU_PEAK"
    printf "  enhancer : RAM peak %6.0f MiB,  CPU peak %6s %%   (Qwen3.5-9B GGUF + mmproj, CPU-only)\n" "$ENH_PEAK" "$ENH_CPU_PEAK"
    printf "  api      : RAM peak %6.0f MiB,  CPU peak %6s %%\n"          "$API_PEAK"   "$API_CPU_PEAK"
    printf "  stack    : RAM peak %6.0f MiB,  CPU peak %6s %%   (concurrent sum across all three)\n" "$RAM_STACK_PEAK" "$CPU_STACK_PEAK"
    echo
    echo "Host telemetry (/proc, 1s cadence):"
    printf "  host CPU peak     : %5s %%\n"                               "$HOST_CPU_PEAK"
    printf "  host CPU avg      : %5s %%\n"                               "$HOST_CPU_AVG"
    printf "  host RAM peak     : %6d MiB used\n"                         "$HOST_MEM_PEAK"
    printf "  host RAM min free : %6d MiB available\n"                    "$HOST_MEM_AVAIL_MIN"
    printf "  host swap peak    : %6d MiB used\n"                         "$HOST_SWAP_PEAK"
    echo
    echo "Output: $OUT_DIR/run.mp4 ($((MP4_SIZE/1024/1024)) MiB)"
    echo
    echo "Raw data:"
    echo "  gpu.csv     — 1 Hz: timestamp, gpu_used_mib, gpu_free_mib, util_gpu_pct,"
    echo "                util_mem_pct, power_w, clock_sm_mhz, temp_c"
    echo "  ram.csv     — 1 Hz: timestamp, comfy_mem_mib, comfy_cpu%, enh_mem_mib,"
    echo "                enh_cpu%, api_mem_mib, api_cpu%"
    echo "  host.csv    — 1 Hz: timestamp, host_cpu%, host_mem_used_mib,"
    echo "                host_mem_avail_mib, host_swap_used_mib"
    echo "  headers.txt — full response headers (all X-Solphur2-* fields)"
    echo "  api.log     — docker logs solphur2-api (phase enhance / submit / comfy_run lines)"
} | tee "$SUMMARY"

log "measurement complete. Summary: $SUMMARY"
