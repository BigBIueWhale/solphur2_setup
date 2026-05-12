#!/usr/bin/env bash
# scripts/clean.sh — full cleanup. Removes everything solphur2 placed on
# this host:
#   • Stop & remove the three containers + bridge network
#   • Delete the three solphur2/* Docker images
#   • Delete every Docker build-cache layer the build produced
#   • Delete the local ./outputs/ directory (generated MP4s)
#   • (with --models) ALSO delete the ./models/ directory (~51 GiB of
#     SHA-256-verified safetensors / GGUF; you'll have to re-download).
#
# Safe by default — does NOT touch models unless you pass --models.
# Use --all to wipe everything (containers + images + cache + outputs + models).
#
# Usage:
#     bash scripts/clean.sh              # containers/images/cache/outputs only
#     bash scripts/clean.sh --models     # ALSO wipe ./models/ (forces re-download)
#     bash scripts/clean.sh --all        # equivalent to --models

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '[\033[36msolphur2-clean\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[\033[33msolphur2-clean\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

WIPE_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --models|--all) WIPE_MODELS=1 ;;
        *) echo "scripts/clean.sh: unknown flag: $arg" >&2; exit 64 ;;
    esac
done

log "stopping and removing containers..."
docker compose --env-file versions.env down --remove-orphans --rmi local 2>&1 | sed 's/^/    /' || true

log "pruning solphur2-tagged images (in case any lingered)..."
for img in solphur2/comfyui solphur2/enhancer solphur2/api; do
    docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${img}:" | xargs -r docker rmi -f 2>&1 | sed 's/^/    /' || true
done

log "pruning Docker build cache..."
docker builder prune -af 2>&1 | tail -5 | sed 's/^/    /' || true

if [[ -d outputs ]]; then
    log "removing ./outputs/ ..."
    rm -rf outputs
fi

if [[ "$WIPE_MODELS" -eq 1 ]]; then
    if [[ -d models ]]; then
        warn "removing ./models/ — this wipes ~51 GiB of pinned model files;"
        warn "scripts/download_models.sh will re-fetch them on next bring-up."
        rm -rf models
    fi
fi

if [[ -d logs ]]; then
    rm -rf logs
fi

log "done."
if [[ "$WIPE_MODELS" -eq 1 ]]; then
    log "Next 'scripts/up.sh' will: download ~51 GiB of models AND rebuild all images."
else
    log "Models preserved. Next 'scripts/up.sh' will only rebuild images."
fi
