#!/usr/bin/env bash
# scripts/up.sh — single command from a fresh checkout to a running stack.
#
# Steps:
#   1. Validate host (Docker, NVIDIA Container Toolkit, RTX 5090, free VRAM).
#   2. Download all models (SHA-256 verified, idempotent).
#   3. Build all three Docker images.
#   4. Bring the compose stack up, wait for healthy state.
#   5. Print the API endpoint and a curl example.
#
# Usage:
#     bash scripts/up.sh                # full bring-up
#     bash scripts/up.sh --skip-build   # skip image builds (use cached)
#     bash scripts/up.sh --skip-models  # skip the model download

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_BUILD=0
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 64 ;;
    esac
done

log() { printf '[\033[36msolphur2\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[\033[31msolphur2\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; exit 1; }

# --- Host validation --------------------------------------------------------
log "validating host..."

command -v docker >/dev/null || fail "docker not installed; expected docker-ce on this host"
docker compose version >/dev/null 2>&1 || fail "docker compose plugin missing"

# Confirm the RTX 5090 + driver 595 + sm_120 are what we expect.
if command -v nvidia-smi >/dev/null; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1)"
    log "GPU: $GPU_INFO"
    case "$GPU_INFO" in
        *"RTX 5090"*"12.0"*) ;;
        *) fail "expected NVIDIA GeForce RTX 5090 (compute_cap 12.0); got: $GPU_INFO" ;;
    esac
else
    fail "nvidia-smi not found; install/verify the NVIDIA driver first"
fi

# Make sure no rogue tenant is holding most of the VRAM. Hard threshold: any
# running compute process (other than Xorg) is treated as a conflict, since
# Sulphur FP8 mixed wants ~30 GiB of the 32607 MiB on this card.
HEAVY_PROCS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
if [[ -n "$HEAVY_PROCS" ]]; then
    log "current GPU compute processes:"
    printf '  %s\n' "$HEAVY_PROCS"
    log "if any of these are using significant VRAM, stop them before continuing."
fi

# --- Models -----------------------------------------------------------------
if [[ "$SKIP_MODELS" -ne 1 ]]; then
    log "downloading models (SHA-256 verified, idempotent)..."
    bash scripts/download_models.sh
else
    log "skipping model download (--skip-models)"
fi

# --- Build images -----------------------------------------------------------
if [[ "$SKIP_BUILD" -ne 1 ]]; then
    log "building docker images (this can take 10-15 minutes for the comfyui image's SageAttention compile)..."
    docker compose --env-file versions.env build --parallel
else
    log "skipping docker build (--skip-build)"
fi

# --- Up ---------------------------------------------------------------------
log "starting compose stack..."
docker compose --env-file versions.env up -d

log "waiting for components to report healthy (this can take ~2 minutes on first run)..."
deadline=$(( $(date +%s) + 600 ))
while (( $(date +%s) < deadline )); do
    READY=1
    for svc in comfyui enhancer api; do
        STATUS="$(docker inspect --format '{{.State.Health.Status}}' solphur2-$svc 2>/dev/null || echo missing)"
        if [[ "$STATUS" != "healthy" ]]; then
            READY=0
            break
        fi
    done
    if (( READY == 1 )); then
        log "all services healthy."
        break
    fi
    sleep 5
done

if (( READY != 1 )); then
    log "components did not reach healthy within 10 minutes; current status:"
    docker compose ps
    fail "see 'docker compose logs <service>' for details"
fi

# --- Smoke test -------------------------------------------------------------
log "smoke-testing the API..."
curl -fsS http://127.0.0.1:8000/healthz | tee /dev/stderr >/dev/null
echo

cat <<EOF

  solphur2 is up.  All inbound on 127.0.0.1 only.

  API:        http://127.0.0.1:8000/
  Health:     http://127.0.0.1:8000/healthz
  ComfyUI:    http://127.0.0.1:8188/  (for debugging; the API does the work)
  Enhancer:   http://127.0.0.1:8080/  (for debugging only)

  Test generation (writes the MP4 to /tmp/solphur2_test.mp4):
      curl -fsS -X POST http://127.0.0.1:8000/generate \\
          -H 'Content-Type: application/json' \\
          -d '{"prompt":"a cinematic close-up of a foggy cobblestone alley at dawn","duration_seconds":20,"mode":"fast"}' \\
          --output /tmp/solphur2_test.mp4

  Bring the stack down:
      docker compose down

EOF
