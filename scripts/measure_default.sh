#!/usr/bin/env bash
# scripts/measure_default.sh — rebuild the API image, then measure peak
# VRAM, peak per-container RAM, and per-phase wall-clock for ONE
# default-config generation.
#
# Why this exists separately from scripts/test.sh and scripts/bench.py:
#   • scripts/test.sh runs functional smoke/headline tests with hard-coded
#     (and non-default) parameters (720p×5s + 1080p×20s), no monitoring.
#   • scripts/bench.py sweeps the (resolution × duration) envelope and only
#     samples VRAM (no RAM, no per-phase split, prompt enhancer disabled).
#   • This script measures EXACTLY the server-side defaults a normal API
#     caller hits: 1280×704 × 10s × 24fps × mode=quality × enhance_prompt=true.
#     Those are the headline numbers the README documents.
#
# Steps (each delegated — no duplicated docker/compose commands here):
#   1. scripts/build.sh                  → rebuild images (layer cache; the
#                                           API image rebuilds quickly when
#                                           api/server.py changes).
#   2. docker compose up -d              → apply the new API image.
#   3. wait for /healthz                 → don't measure during startup.
#   4. sample nvidia-smi + cgroup v2     → 1 s cadence, background.
#   5. POST /generate                    → minimal body, server defaults.
#   6. aggregate peaks                   → from CSVs.
#   7. print summary                     → including the per-phase X-Solphur2
#                                           headers the API returned.
#
# Usage:
#     bash scripts/measure_default.sh "<prompt>"                       # rebuild + measure
#     bash scripts/measure_default.sh "<prompt>" --skip-build          # skip the rebuild
#     bash scripts/measure_default.sh "<prompt>" --no-cache            # full rebuild from scratch
#
# The prompt is a REQUIRED positional argument — there is intentionally no
# default. Defaulting would either ship a SFW prompt (silently bypassing
# Sulphur's uncensored capability and giving misleading timing data) or ship
# an NSFW prompt that nobody asked for. Make the caller think about what
# they're measuring.
#
# Example:
#     bash scripts/measure_default.sh \
#         "a beautiful nude woman lying on satin sheets, soft golden light, slow cinematic tracking camera, 35mm"
#
# Output: ./bench_runs/measure_default_YYYYMMDD-HHMMSS/
#   ├── run.mp4                           → the generated video
#   ├── headers.txt                       → all X-Solphur2-* response headers
#   ├── gpu.csv                           → 1s-cadence nvidia-smi
#   ├── ram.csv                           → 1s-cadence cgroup v2 (memory.current + cpu.stat)
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
usage: bash scripts/measure_default.sh "<prompt>" [--skip-build|--no-cache]

  <prompt>       REQUIRED positional argument. Free-text prompt for /generate.
                 No default — see scripts/measure_default.sh header for why.
  --skip-build   Reuse the current Docker images. Default rebuilds (incremental).
  --no-cache     Full rebuild from scratch (rare; use when upstream wheels
                 changed without a version pin bumping).

EOF
    exit 64
}

PROMPT=""
DO_BUILD=1
BUILD_FLAGS=()
while (( $# > 0 )); do
    case "$1" in
        --skip-build)   DO_BUILD=0 ;;
        --no-cache)     BUILD_FLAGS+=("--no-cache") ;;
        -h|--help)      usage ;;
        --*)            echo "unknown flag: $1" >&2; usage ;;
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
OUT_DIR="$REPO_ROOT/bench_runs/measure_default_${STAMP}"
mkdir -p "$OUT_DIR"

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
# We sample three things at 1 second cadence for the duration of the request:
#   • GPU full telemetry via nvidia-smi: memory used/free, compute
#     utilization %, memory-controller utilization %, power draw W,
#     SM clock MHz, GPU temperature.
#   • Per-container RSS + CPU% via `docker stats --no-stream` (one
#     snapshot per second). All three containers in a fixed column order.
#   • Host-level CPU + RAM via /proc/stat + /proc/meminfo (free hits the
#     same kernel counters but parsing one file is fewer fork/exec).
#
# Each loop writes one CSV row per second. We capture sampler PIDs so we
# can kill them precisely (no broad pkill) once the request returns.

GPU_CSV="$OUT_DIR/gpu.csv"
RAM_CSV="$OUT_DIR/ram.csv"
HOST_CSV="$OUT_DIR/host.csv"

log "starting samplers (1 s cadence) → gpu.csv, ram.csv, host.csv"

# GPU sampler — nvidia-smi --loop emits one row per interval to stdout.
# Column order: timestamp, memory.used (MiB), memory.free (MiB),
# utilization.gpu (%), utilization.memory (%), power.draw (W),
# clocks.sm (MHz), temperature.gpu (deg C).
echo "timestamp_iso,gpu_used_mib,gpu_free_mib,util_gpu_pct,util_mem_pct,power_w,clock_sm_mhz,temp_c" > "$GPU_CSV"
nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 1000 >> "$GPU_CSV" 2>&1 &
GPU_PID=$!

# RAM + CPU sampler — reads cgroup v2 files directly. `docker stats` is
# too slow (a single snapshot call takes ~2-3 seconds on this host because
# the daemon round-trip dominates), which produced sparse RAM/CPU CSVs in
# the first version of this script — RAM was sampled at 0.3 Hz while GPU
# was at 1 Hz, so transient comfy-load spikes were missed. The cgroup v2
# files are kernel-maintained and return in microseconds.
#
# memory.current : bytes resident (includes file-backed cache, which is
#                  important because the FP8 safetensors are mmap'd by
#                  ComfyUI and contribute to the container's RSS).
# cpu.stat       : has a `usage_usec` line (total CPU microseconds since
#                  the cgroup was created). Diff between two reads /
#                  wall-clock-delta gives CPU%.

# Resolve container IDs once; abort cleanly if any container disappeared.
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
    # Bootstrap previous-tick CPU usage so the first computed sample is
    # meaningful (read each cpu.stat twice with a 1 s gap).
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

        # Memory (bytes → MiB, integer truncation).
        comfy_mib=$(( $(<"$COMFY_MEM") / 1048576 ))
        enh_mib=$((   $(<"$ENH_MEM")   / 1048576 ))
        api_mib=$((   $(<"$API_MEM")   / 1048576 ))

        # CPU: usec since cgroup creation. Delta / elapsed-time gives %.
        # On an N-core box, sum of all cores in use ranges 0..N*100%.
        cur_comfy_us=$(read_cpu_usec "$COMFY_CPU")
        cur_enh_us=$(  read_cpu_usec "$ENH_CPU")
        cur_api_us=$(  read_cpu_usec "$API_CPU")
        d_comfy_us=$(( cur_comfy_us - prev_comfy_us ))
        d_enh_us=$((   cur_enh_us   - prev_enh_us   ))
        d_api_us=$((   cur_api_us   - prev_api_us   ))
        prev_comfy_us=$cur_comfy_us
        prev_enh_us=$cur_enh_us
        prev_api_us=$cur_api_us
        # cpu_pct = (d_us * 1000) / d_ns * 100 = d_us * 100000 / d_ns
        # awk for the division (bash arithmetic is integer-only).
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

# Host sampler — overall CPU% (computed from /proc/stat delta) + RAM
# (from /proc/meminfo). Useful so we can confirm the system as a whole
# isn't going into swap or thrashing besides the containers.
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

# --- 4. Fire one POST /generate at server-side defaults ------------------
# The actual HTTP request is delegated to scripts/_measure_client.py — see
# its module docstring for why it's a dedicated Python client rather than
# inline curl + bash JSON-escaping. The body it constructs contains ONLY
# the user-supplied prompt; every other parameter falls through to the
# pydantic defaults in api/server.py:GenerateRequest.
log "POST $API_URL/generate via scripts/_measure_client.py (server-side defaults)"

T0="$(date +%s)"
CLIENT_RC=0
python3 scripts/_measure_client.py \
    --prompt   "$PROMPT" \
    --out-dir  "$OUT_DIR" \
    --api-url  "$API_URL" \
    --timeout-seconds 1800 || CLIENT_RC=$?
T1="$(date +%s)"
WALL=$(( T1 - T0 ))

cleanup
trap - EXIT

if (( CLIENT_RC != 0 )); then
    log "FAIL: _measure_client.py exited $CLIENT_RC after ${WALL}s; first 600 bytes of body:"
    head -c 600 "$OUT_DIR/run.mp4" 2>/dev/null || true; echo
    exit 1
fi

# Extract HTTP status from the first header line for the summary.
HTTP=$(awk 'NR==1 {print $2; exit}' "$OUT_DIR/headers.txt")
[[ "$HTTP" == "200" ]] || fail "expected HTTP 200 in headers; got: $HTTP"

# --- 5. Capture API logs for the same window (phase lines) ---------------
docker logs --since "$((WALL + 10))s" solphur2-api > "$OUT_DIR/api.log" 2>&1 || true

# --- 6. Aggregate peaks --------------------------------------------------
# GPU CSV columns (post-header):
#   1 timestamp_iso, 2 gpu_used_mib, 3 gpu_free_mib, 4 util_gpu_pct,
#   5 util_mem_pct, 6 power_w, 7 clock_sm_mhz, 8 temp_c
GPU_PEAK=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$GPU_CSV")
GPU_BASE=$(awk -F, 'NR==2 {print $2+0; exit}' "$GPU_CSV")
GPU_UTIL_PEAK=$(awk -F, 'NR>1 && $4+0 > max {max=$4+0} END {print max+0}' "$GPU_CSV")
GPU_UTIL_AVG=$( awk -F, 'NR>1 {s+=$4+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$GPU_CSV")
POWER_PEAK=$(  awk -F, 'NR>1 && $6+0 > max {max=$6+0} END {printf "%.1f", max+0}' "$GPU_CSV")
POWER_AVG=$(   awk -F, 'NR>1 {s+=$6+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$GPU_CSV")
CLOCK_PEAK=$(  awk -F, 'NR>1 && $7+0 > max {max=$7+0} END {print max+0}' "$GPU_CSV")
TEMP_PEAK=$(   awk -F, 'NR>1 && $8+0 > max {max=$8+0} END {print max+0}' "$GPU_CSV")

# RAM CSV columns (post-header):
#   1 timestamp_iso, 2 comfy_mem, 3 comfy_cpu%, 4 enh_mem, 5 enh_cpu%,
#   6 api_mem,   7 api_cpu%
COMFY_PEAK=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$RAM_CSV")
ENH_PEAK=$(  awk -F, 'NR>1 && $4+0 > max {max=$4+0} END {print max+0}' "$RAM_CSV")
API_PEAK=$(  awk -F, 'NR>1 && $6+0 > max {max=$6+0} END {print max+0}' "$RAM_CSV")
COMFY_CPU_PEAK=$(awk -F, 'NR>1 && $3+0 > max {max=$3+0} END {printf "%.1f", max+0}' "$RAM_CSV")
ENH_CPU_PEAK=$(  awk -F, 'NR>1 && $5+0 > max {max=$5+0} END {printf "%.1f", max+0}' "$RAM_CSV")
API_CPU_PEAK=$(  awk -F, 'NR>1 && $7+0 > max {max=$7+0} END {printf "%.1f", max+0}' "$RAM_CSV")
# Concurrent maxima — sum of all three container columns per row.
RAM_STACK_PEAK=$( awk -F, 'NR>1 {s=$2+$4+$6; if (s>max) max=s} END {print max+0}' "$RAM_CSV")
CPU_STACK_PEAK=$( awk -F, 'NR>1 {s=$3+$5+$7; if (s>max) max=s} END {printf "%.1f", max+0}' "$RAM_CSV")

# Host CSV columns: 1 timestamp_iso, 2 host_cpu%, 3 host_mem_used_mib,
#   4 host_mem_avail_mib, 5 host_swap_used_mib
HOST_CPU_PEAK=$(   awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {printf "%.1f", max+0}' "$HOST_CSV")
HOST_CPU_AVG=$(    awk -F, 'NR>1 {s+=$2+0; n++} END {if (n>0) printf "%.1f", s/n; else print 0}' "$HOST_CSV")
HOST_MEM_PEAK=$(   awk -F, 'NR>1 && $3+0 > max {max=$3+0} END {print max+0}' "$HOST_CSV")
HOST_MEM_AVAIL_MIN=$( awk -F, 'NR>1 {if (NR==2 || $4+0 < min) min=$4+0} END {print min+0}' "$HOST_CSV")
HOST_SWAP_PEAK=$(  awk -F, 'NR>1 && $5+0 > max {max=$5+0} END {print max+0}' "$HOST_CSV")

# Pull per-phase timings out of the response headers.
PHASE_ENHANCE=$(awk -F': ' 'tolower($1)=="x-solphur2-phaseenhanceseconds"  {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
PHASE_SUBMIT=$( awk -F': ' 'tolower($1)=="x-solphur2-phasesubmitseconds"   {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
PHASE_COMFY=$(  awk -F': ' 'tolower($1)=="x-solphur2-phasecomfyrunseconds" {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
ELAPSED=$(      awk -F': ' 'tolower($1)=="x-solphur2-elapsedseconds"       {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
SEED=$(         awk -F': ' 'tolower($1)=="x-solphur2-seed"                 {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
RES=$(          awk -F': ' 'tolower($1)=="x-solphur2-resolution"           {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
FRAMES=$(       awk -F': ' 'tolower($1)=="x-solphur2-frames"               {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')
MODE=$(         awk -F': ' 'tolower($1)=="x-solphur2-mode"                 {print $2}' "$OUT_DIR/headers.txt" | tr -d '\r\n')

MP4_SIZE=$(stat -c %s "$OUT_DIR/run.mp4")

# --- 7. Print + persist summary ------------------------------------------
SUMMARY="$OUT_DIR/summary.txt"
{
    echo "solphur2 default-config measurement — ${STAMP}"
    echo "==========================================================="
    echo "Config (server-side defaults; only 'prompt' sent in body):"
    echo "  resolution  : $RES"
    echo "  frames      : $FRAMES"
    echo "  mode        : $MODE"
    echo "  seed        : $SEED"
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
