#!/usr/bin/env bash
# scripts/build.sh — build all three Docker images.
#
# Reads pinned versions from versions.env, passes them as build args to
# docker-compose. Use `--no-cache` to force a full rebuild (useful when
# upstream tooling changes silently — pinning catches most of these but
# Triton wheel rebuilds, for example, are not signature-stable).
#
# Usage:
#     bash scripts/build.sh                # incremental build (uses Docker layer cache)
#     bash scripts/build.sh --no-cache     # full rebuild from scratch
#     bash scripts/build.sh --parallel     # build the three services in parallel (CPU-heavy)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '[\033[36msolphur2-build\033[0m %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --no-cache|--parallel|--pull) PASSTHROUGH+=("$arg") ;;
        *) echo "scripts/build.sh: unknown flag: $arg" >&2; exit 64 ;;
    esac
done

log "building solphur2/{comfyui,enhancer,api} images..."
log "(this is normally 15–20 min on first run because SageAttention 2.2.0"
log " is compiled from source for compute capability sm_120; incremental"
log " rebuilds with the Docker layer cache typically finish in ~60 seconds)"
docker compose --env-file versions.env build "${PASSTHROUGH[@]}"

log "done. Built images:"
docker images --format '  {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep solphur2 || true
