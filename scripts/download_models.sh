#!/usr/bin/env bash
# scripts/download_models.sh — idempotent, SHA-256-pinned model fetch.
#
# Downloads every model file the solphur2 stack consumes, placing each into
# the ComfyUI-standard subdirectory under ./models/ and verifying the SHA-256
# digest against the LFS oid pinned in versions.env. Re-running with all files
# already present and intact is a no-op.
#
# Usage:
#     bash scripts/download_models.sh                  # default ./models target
#     MODELS_DIR=/srv/models bash scripts/download_models.sh
#
# Total download: ~58 GiB. With 1 Gbps WAN this takes 10-15 minutes.
# Disk space: 60 GiB on the target volume (some staging headroom).

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"

# shellcheck disable=SC1091
source "$REPO_ROOT/versions.env"

# --- helpers ----------------------------------------------------------------

log()  { printf '[\033[36m%s\033[0m] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '[\033[33m%s\033[0m] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
fail() { printf '[\033[31m%s\033[0m] %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

require() {
    for bin in "$@"; do
        command -v "$bin" >/dev/null || fail "missing required tool: $bin"
    done
}

require curl sha256sum mkdir mv awk

# Verify an existing file's SHA-256 matches the pin.
verify() {
    local path="$1" expected="$2"
    [[ -f "$path" ]] || return 1
    local actual
    actual="$(sha256sum -- "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]]
}

# Download one file:
#   $1  destination subdirectory (relative to $MODELS_DIR)
#   $2  destination filename
#   $3  expected SHA-256
#   $4  expected size in bytes (used to short-circuit a verify if file size matches)
#   $5  remote URL
fetch() {
    local subdir="$1" name="$2" sha="$3" size="$4" url="$5"
    local target_dir="$MODELS_DIR/$subdir"
    local target="$target_dir/$name"
    local staging="$target.partial"

    mkdir -p "$target_dir"

    if verify "$target" "$sha"; then
        log "OK   $subdir/$name  (already present, sha256 matches)"
        return 0
    fi

    if [[ -f "$target" ]]; then
        local actual_size
        actual_size="$(stat -c %s "$target" 2>/dev/null || stat -f %z "$target")"
        warn "BAD  $subdir/$name  (have $actual_size bytes, expected $size; re-downloading)"
        rm -f "$target"
    fi

    log "GET  $subdir/$name  ($((size / 1024 / 1024)) MiB)"
    # -C - resumes a partial download if the server emits Range.
    # -L follows the HF redirect chain to the cas server.
    # --retry 5 + --retry-delay 5 for transient 503s from xet backend.
    # -f fails on HTTP 4xx/5xx so set -e trips.
    # --create-dirs ensures the target dir exists (we already mkdir-p above).
    curl -fL \
         --retry 5 --retry-delay 5 \
         --connect-timeout 30 \
         -o "$staging" \
         -C - \
         "$url" || fail "download failed: $url"

    log "VERIFY $subdir/$name"
    if ! verify "$staging" "$sha"; then
        local actual
        actual="$(sha256sum -- "$staging" | awk '{print $1}')"
        fail "checksum mismatch for $subdir/$name
              expected: $sha
              got:      $actual
              staging file kept at: $staging"
    fi

    mv -- "$staging" "$target"
    log "DONE $subdir/$name"
}

# --- inventory --------------------------------------------------------------
#
# ComfyUI standard model directory layout under models/:
#   checkpoints/      → CheckpointLoaderSimple, LTXVAudioVAELoader.ckpt_name,
#                       LTXAVTextEncoderLoader.ckpt_name (all read here).
#   loras/            → LoraLoaderModelOnly.
#   latent_upscale_models/ → LatentUpscaleModelLoader (LATENT-space upscalers
#                            such as the LTX-2.3 spatial upscaler; this is a
#                            DIFFERENT folder than `upscale_models/` which holds
#                            image-space ESRGAN-style models).
#   text_encoders/    → LTXAVTextEncoderLoader.text_encoder.
#   prompt_enhancer/  → llama-server (mounted directly, not a ComfyUI dir).

fetch checkpoints \
    "${SULPHUR_FP8MIXED_PATH}" \
    "${SULPHUR_FP8MIXED_SHA256}" \
    "${SULPHUR_FP8MIXED_SIZE}" \
    "${SULPHUR_FP8MIXED_URL}"

fetch loras \
    "${LTX_DISTILL_LORA_PATH}" \
    "${LTX_DISTILL_LORA_SHA256}" \
    "${LTX_DISTILL_LORA_SIZE}" \
    "${LTX_DISTILL_LORA_URL}"

fetch latent_upscale_models \
    "${LTX_SPATIAL_UPSCALER_PATH}" \
    "${LTX_SPATIAL_UPSCALER_SHA256}" \
    "${LTX_SPATIAL_UPSCALER_SIZE}" \
    "${LTX_SPATIAL_UPSCALER_URL}"

fetch text_encoders \
    "${GEMMA3_TEXT_ENCODER_PATH}" \
    "${GEMMA3_TEXT_ENCODER_SHA256}" \
    "${GEMMA3_TEXT_ENCODER_SIZE}" \
    "${GEMMA3_TEXT_ENCODER_URL}"

# prompt_enhancer files keep their literal HF subpath stripped to basename
# because the enhancer container mounts the prompt_enhancer dir flat.
ENH_GGUF_BASENAME="${PROMPT_ENHANCER_GGUF_PATH##*/}"
ENH_MMPROJ_BASENAME="${PROMPT_ENHANCER_MMPROJ_PATH##*/}"

fetch prompt_enhancer \
    "${ENH_GGUF_BASENAME}" \
    "${PROMPT_ENHANCER_GGUF_SHA256}" \
    "${PROMPT_ENHANCER_GGUF_SIZE}" \
    "${PROMPT_ENHANCER_GGUF_URL}"

fetch prompt_enhancer \
    "${ENH_MMPROJ_BASENAME}" \
    "${PROMPT_ENHANCER_MMPROJ_SHA256}" \
    "${PROMPT_ENHANCER_MMPROJ_SIZE}" \
    "${PROMPT_ENHANCER_MMPROJ_URL}"

log "all models present and SHA-256-verified under $MODELS_DIR"
log "total size:"
du -sh "$MODELS_DIR"/* 2>/dev/null | sort -h >&2 || true
