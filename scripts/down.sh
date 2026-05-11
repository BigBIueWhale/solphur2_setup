#!/usr/bin/env bash
# scripts/down.sh — gracefully stop the solphur2 stack.
#
# Stops and removes containers + the bridge network. Preserves images,
# models, and outputs (so the next `up.sh` brings the system back in
# seconds without re-downloading or re-building).
#
# For a fuller cleanup (also remove images / model files / build cache),
# use scripts/clean.sh.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '[\033[36msolphur2-down\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "stopping solphur2 stack..."
docker compose --env-file versions.env down --remove-orphans

log "done. Containers removed. Images, models, and outputs preserved."
log "Bring back up with:  bash scripts/up.sh"
