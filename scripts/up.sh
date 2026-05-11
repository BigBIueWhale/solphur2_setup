#!/usr/bin/env bash
# scripts/up.sh — one-command bring-up of the solphur2 stack.
#
# Orchestrates the other scripts in the canonical order. Each step is
# idempotent — re-running scripts/up.sh after a successful prior run
# converges to "stack is healthy" without doing redundant work.
#
# Steps (each delegated to its own script — no duplicated commands):
#   1. Validate host hardware (NVIDIA GeForce RTX 5090, sm_120, Docker,
#      NVIDIA Container Toolkit, sufficient VRAM headroom).
#   2. Download all SHA-256-pinned model files (scripts/download_models.sh).
#   3. Build the three Docker images (scripts/build.sh).
#   4. Bring up the compose stack and wait for all three healthchecks.
#   5. Print the bound endpoints and the next-step command.
#
# Verifying generation actually works is delegated to scripts/test.sh —
# this script does not run a generation by itself.
#
# Flags:
#     --skip-validate   skip host hardware validation (use if running on
#                       a non-5090 box for experimentation)
#     --skip-models     skip the model download (assume models/ is populated)
#     --skip-build      skip the Docker build (reuse cached images)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '[\033[36msolphur2-up\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[\033[31msolphur2-up\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; exit 1; }

DO_VALIDATE=1
DO_MODELS=1
DO_BUILD=1
for arg in "$@"; do
    case "$arg" in
        --skip-validate)  DO_VALIDATE=0 ;;
        --skip-models)    DO_MODELS=0 ;;
        --skip-build)     DO_BUILD=0 ;;
        *) echo "scripts/up.sh: unknown flag: $arg" >&2; exit 64 ;;
    esac
done

# --- 1. Host validation ----------------------------------------------------
if [[ "$DO_VALIDATE" -eq 1 ]]; then
    log "validating host..."
    command -v docker >/dev/null || fail "docker not installed"
    docker compose version >/dev/null 2>&1 || fail "docker compose plugin missing"
    command -v nvidia-smi >/dev/null || fail "nvidia-smi not found; install the NVIDIA driver"

    GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1)"
    log "GPU: $GPU_INFO"
    case "$GPU_INFO" in
        *"RTX 5090"*"12.0"*) ;;
        *) fail "expected NVIDIA GeForce RTX 5090 (compute cap 12.0); got: $GPU_INFO" ;;
    esac

    VRAM_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
    log "VRAM free: ${VRAM_FREE_MIB} MiB / 32607 MiB total"
    if [[ "$VRAM_FREE_MIB" -lt 31000 ]]; then
        fail "less than 31 GiB VRAM free; stop other GPU users (try 'docker ps' / 'nvidia-smi') before continuing"
    fi
fi

# --- 2. Download models ----------------------------------------------------
if [[ "$DO_MODELS" -eq 1 ]]; then
    log "downloading models (SHA-256-verified, idempotent)..."
    bash scripts/download_models.sh
else
    log "skipping model download (--skip-models)"
fi

# --- 3. Build images -------------------------------------------------------
if [[ "$DO_BUILD" -eq 1 ]]; then
    log "building Docker images..."
    bash scripts/build.sh
else
    log "skipping Docker build (--skip-build)"
fi

# --- 4. Bring up the stack ------------------------------------------------
log "starting compose stack..."
docker compose --env-file versions.env up -d

log "waiting for all three components to report healthy (≤10 min)..."
deadline=$(( $(date +%s) + 600 ))
while (( $(date +%s) < deadline )); do
    ready=1
    for svc in comfyui enhancer api; do
        status="$(docker inspect --format '{{.State.Health.Status}}' "solphur2-$svc" 2>/dev/null || echo missing)"
        if [[ "$status" != "healthy" ]]; then
            ready=0
            break
        fi
    done
    (( ready == 1 )) && break
    sleep 5
done

(( ready == 1 )) || {
    docker compose --env-file versions.env ps
    fail "components did not reach healthy within 10 minutes; see 'docker compose logs <service>' for details"
}

log "all three components healthy."

# --- 5. Print endpoints ---------------------------------------------------
cat <<EOF

  solphur2 is up. All inbound on 127.0.0.1 only.

  API:        http://127.0.0.1:8000/        (POST /generate)
  ComfyUI:    http://127.0.0.1:8188/        (debug only)
  Enhancer:   http://127.0.0.1:8080/        (debug only)

  Verify everything works end-to-end (~2 min smoke + ~30 min headline):
      bash scripts/test.sh

  Smoke only (~2 min):
      bash scripts/test.sh --smoke-only

  Bring everything down (preserves images + models + outputs):
      bash scripts/down.sh

EOF
