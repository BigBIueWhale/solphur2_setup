"""
solphur2 API gateway — single-file FastAPI service.

This single Python file owns BOTH:
  • The HTTP API surface (FastAPI routes).
  • The ComfyUI workflow definition (built programmatically in API format).

There is no GUI workflow JSON to convert; the workflow lives in code,
parameterized per request. The user-facing surface is a clean HTTP API on
http://127.0.0.1:8000 (loopback only via the docker-compose port mapping).

Endpoints:
  GET  /healthz    component liveness.
  POST /generate   text-to-video. Body: GenerateRequest (JSON).
  POST /generate/i2v   image-to-video. Multipart (stub in v1).

Each /generate POST returns exactly one MP4 (Content-Type: video/mp4).
Atomic single-file response — no streaming, no multipart out.

Hardware ceiling assumptions (validated 2026-05-11 on RTX 5090 sm_120, 32607 MiB):
  • 1920x1088 × 481 frames @ 24 fps via Sulphur FP8 mixed + per-stage distill
    LoRA peaks at ~26.2 GiB VRAM, ~7 min wall-clock (measured 433s on Test A).
    The canonical two-stage
    pipeline (half-resolution base at 960×544 → x2-spatial-upsample → refine)
    is what keeps the activation footprint manageable; the wide gap from the
    32.6 GiB ceiling leaves room for the FP16-SageAttention / BF16-Gemma3
    quality lifts when those land.
  • "quality" mode (no distill LoRA, 50-step LTXVScheduler) runs the full
    non-distilled Sulphur dev model at the same VRAM peak, ~14 min (measured 853s on RTX 5090 Test B).
  • The full 20-second / 24 fps / 1080p envelope is Lightricks' documented max
    for LTX-2.3 (arXiv:2601.03233 §6.3; docs.ltx.video/models).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# ─────────────────────────── Configuration ────────────────────────────────────

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui:8188")
ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8080")
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/outputs"))

# Model filenames (must match what scripts/download_models.sh placed under ./models/).
SULPHUR_CKPT = "sulphur_dev_fp8mixed.safetensors"
DISTILL_LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA3_TEXT_ENCODER = "gemma_3_12B_it_fp8_scaled.safetensors"  # FP8-scaled; BF16 deferred — Test C crashed at FFmpeg audio mux on BF16

# Per-stage distill LoRA strengths — Sulphur-canonical.
# Sulphur applies the distill LoRA at 0.7 on stage 1 and 0.5 on stage 2.
# TenStrip's README for the `cond_safe` LoRA family says these strengths are
# safe ("1.0 first pass i2v / 0.4–0.5 upscale pass"); Sulphur uses 0.7 on
# stage 1 because it also stacks an additional Sulphur LoRA in its native
# workflow. We use a single LoRA (the full Sulphur model carries the fine-
# tune in weights), so 0.7 stage 1 is conservative-but-faithful.
DISTILL_LORA_STRENGTH_STAGE1 = 0.7
DISTILL_LORA_STRENGTH_STAGE2 = 0.5

# Stage-1 sigmas for "fast" mode — Lightricks' canonical DISTILLED_SIGMA_VALUES.
# Source: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
# Also used verbatim in the official LTX-2.3_T2V_I2V_Two_Stage_Distilled.json
# example workflow. 9 sigma values = 8 active denoising steps.
# Sulphur's own workflow uses LTXVScheduler(8, 4, 1.5, true, 0.1) — those
# max_shift / base_shift values are outside Lightricks' documented range and
# untested on Blackwell, so we prefer the documented Lightricks-canonical
# ManualSigmas path.
STAGE1_SIGMAS_FAST = (
    "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
)

# Stage-2 sigmas for "fast" mode — both Sulphur's shipped workflow (connected
# nodes #7) AND Lightricks' canonical two-stage distilled example agree on
# exactly these 4 values. 4 sigma values = 3 active refinement steps.
STAGE2_SIGMAS_FAST = "0.85, 0.7250, 0.4219, 0.0"

# "Quality" mode = Sulphur's non-distilled BASE workflow (ltx23_t2v base.json):
# LTXVScheduler(50 steps, max_shift=2.72, base_shift=0.8, stretch=true,
# terminal=0) + euler_ancestral + CFG=3.6 stage 1 / CFG=1.0 stage 2.
# Stage-2 sigmas remain the same canonical 4-tuple.
QUALITY_LTXVSCHEDULER_STEPS = 50
QUALITY_LTXVSCHEDULER_MAX_SHIFT = 2.72
QUALITY_LTXVSCHEDULER_BASE_SHIFT = 0.8
QUALITY_LTXVSCHEDULER_TERMINAL = 0.0
QUALITY_CFG_STAGE1 = 3.6
QUALITY_CFG_STAGE2 = 1.0

# Tiled VAE decode params for LTXVTiledVAEDecode (count-based, not pixel-based).
# Verified inputs against Lightricks/ComfyUI-LTXVideo @229437c6 tiled_vae_decode.py:13-28:
# horizontal_tiles, vertical_tiles, overlap (1-8), last_frame_fix (bool),
# optional working_device + working_dtype.
# 2x2 / overlap=6 is the official example default and fits 1080p × 481 frames in
# our 32 GiB budget.
VAE_HORIZONTAL_TILES = 2
VAE_VERTICAL_TILES = 2
VAE_TILE_OVERLAP = 6

POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 60 * 45  # 45 minutes — covers quality mode.
SUBMIT_HTTP_TIMEOUT = 30.0
POLL_HTTP_TIMEOUT = 10.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s solphur2-api %(message)s",
)
log = logging.getLogger("solphur2-api")


# ─────────────────────────── Workflow builder ─────────────────────────────────


def _build_workflow(
    *,
    prompt_text: str,
    negative_text: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    fps: int,
    mode: str,
    output_filename_prefix: str,
) -> dict[str, Any]:
    """Build a ComfyUI API-format workflow dict for Sulphur-2-base t2v.

    Every input name, slot index, and type below was verified against the
    pinned source code (ComfyUI v0.21.0, ComfyUI-LTXVideo @229437c6,
    ComfyUI-KJNodes 1.4.0). See `research_comfyapi/` for the citation trail.

    Architecture (two-stage tiled inference per LTX-2 paper §4.2):

      Loaders:
        1 CheckpointLoaderSimple → (MODEL, CLIP_unused, VAE)
        2 PathchSageAttentionKJ  → MODEL (model patched in-place)
        3 LoraLoaderModelOnly    → MODEL (fast mode only; distill LoRA strength 0.5)
        4 LTXAVTextEncoderLoader → CLIP (Gemma3 fp8_scaled + same ckpt for fusion)
        5 LatentUpscaleModelLoader → LATENT_UPSCALE_MODEL (spatial x2)
        6 LTXVAudioVAELoader     → VAE (audio path; extracts audio_vae from main ckpt)

      Primitives (centralized so the API can patch by title if ever needed):
        101 PrimitiveString  positive prompt text
        102 PrimitiveString  negative prompt text
        110 PrimitiveFloat   frame rate (FLOAT — LTXVConditioning + CreateVideo)
        111 LTXFloatToInt    integer projection of 110 (LTXVEmptyLatentAudio)
        120/121/122 PrimitiveInt  width / height / length
        130 PrimitiveInt     seed (stage-1)

      Conditioning:
        10 CLIPTextEncode (positive) — clip from 4@0
        11 CLIPTextEncode (negative) — clip from 4@0
        12 LTXVConditioning → returns (CONDITIONING, CONDITIONING) at slots 0, 1

      Stage-1 latent prep + sampling (at base resolution, e.g. 960x544x481):
        20 EmptyLTXVLatentVideo
        21 LTXVEmptyLatentAudio
        22 LTXVConcatAVLatent (joins video + audio token streams)
        30 KSamplerSelect (euler_cfg_pp — deterministic stage-1)
        31 ManualSigmas
        32 RandomNoise
        33 CFGGuider (cfg=1.0)
        34 SamplerCustomAdvanced → LATENT (AV-concat)

      Stage-2 latent prep + sampling (after spatial x2 upscaling of VIDEO ONLY):
        35 LTXVSeparateAVLatent (split stage-1 output)
        40 LTXVLatentUpsampler (video latent + vae + upscale_model) → LATENT
        36 LTXVConcatAVLatent (upsampled video + stage-1 audio) → LATENT
        41 KSamplerSelect (euler_ancestral_cfg_pp — stochastic refine)
        42 ManualSigmas
        43 RandomNoise (seed+1)
        44 CFGGuider (cfg=1.0)
        45 SamplerCustomAdvanced → LATENT (refined AV)

      Decode + mux:
        50 LTXVSeparateAVLatent
        51 LTXVTiledVAEDecode (video, count-based tiling)
        52 LTXVAudioVAEDecode
        53 CreateVideo (fps is FLOAT)
        54 SaveVideo (format=mp4, codec=h264)
    """
    nodes: dict[str, Any] = {}

    def add(
        node_id: int,
        class_type: str,
        inputs: dict[str, Any],
        title: str | None = None,
    ) -> None:
        nodes[str(node_id)] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": title or class_type},
        }

    def ref(node_id: int, slot: int = 0) -> list[Any]:
        return [str(node_id), slot]

    apply_distill_lora = (mode == "fast")

    # ---- Loaders ------------------------------------------------------------

    add(1, "CheckpointLoaderSimple", {"ckpt_name": SULPHUR_CKPT}, "Checkpoint")

    # SageAttention mode: explicit FP16 PV path, NOT "auto".
    # On sm_120, KJNodes' "auto" dispatcher in SageAttention 2.2.0 routes to
    # `sageattn_qk_int8_pv_fp8_cuda(pv_accum_dtype="fp32+fp16")` — the LEAST
    # accurate non-FP4 option available. The explicit
    # `sageattn_qk_int8_pv_fp16_cuda` mode keeps PV in FP16 with an FP32
    # accumulator (closer to SDPA baseline) at the cost of ~10-20% attention-
    # kernel slowdown vs FP8 — still 25-30% faster than naive SDPA per
    # mobcat40's RTX 5090 benchmarks. Cross-referenced against the canonical
    # community recommendation in Comfy-Org/ComfyUI Discussion #11583 and the
    # mobcat40/sageattention-blackwell README.
    add(
        2,
        "PathchSageAttentionKJ",
        {
            "model": ref(1, 0),
            "sage_attention": "sageattn_qk_int8_pv_fp16_cuda",
            "allow_compile": False,
        },
        "Patch SageAttention",
    )

    # Per-stage distill-LoRA strengths — fast mode only.
    #
    # The distill LoRA is applied ONLY in "fast" mode at Sulphur's per-stage
    # strengths 0.7 stage 1 / 0.5 stage 2 (matching `sulphur_workflow.json`
    # and TenStrip's `fro90_ceil72_condsafe` README guidance).
    #
    # In "quality" mode the distill LoRA is INTENTIONALLY OMITTED, even though
    # Sulphur's shipped non-distilled base workflow (`ltx23_t2v base.json`)
    # applies it at 0.5/0.5. The reason for our deviation is empirical:
    # we use the full Sulphur FP8-mixed model (`sulphur_dev_fp8mixed.safetensors`)
    # as the base, whereas Sulphur's base workflow uses LTX-2.3-base PLUS
    # `sulphur_final.safetensors` as a separate LoRA. The maintainer's README
    # explicitly warns: "use the lora or use the full models, don't use both at
    # the same time." Stacking the distill LoRA on top of the full Sulphur
    # model in 50-step LTXVScheduler + CFG=3.6 mode produced an
    # avcodec_send_frame() EINVAL failure at the SaveVideo audio encoder
    # (the audio VAE output goes outside FFmpeg's acceptable amplitude range).
    # Without the distill LoRA in quality mode, audio encodes correctly and
    # visual quality is high (Test B baseline).
    if apply_distill_lora:
        add(
            3,
            "LoraLoaderModelOnly",
            {
                "model": ref(2, 0),
                "lora_name": DISTILL_LORA,
                "strength_model": DISTILL_LORA_STRENGTH_STAGE1,
            },
            "Distill LoRA (Stage 1)",
        )
        add(
            7,
            "LoraLoaderModelOnly",
            {
                "model": ref(2, 0),
                "lora_name": DISTILL_LORA,
                "strength_model": DISTILL_LORA_STRENGTH_STAGE2,
            },
            "Distill LoRA (Stage 2)",
        )
        model_stage1 = ref(3, 0)
        model_stage2 = ref(7, 0)
    else:
        # Quality mode: no distill LoRA — see comment above for empirical
        # rationale. Same model object feeds both stage-1 and stage-2 guiders.
        model_stage1 = ref(2, 0)
        model_stage2 = ref(2, 0)

    # LTXAVTextEncoderLoader fuses the Gemma3 text encoder with the LTX-2.3
    # checkpoint so cross-attention is wired correctly. Both filenames required.
    # RETURN_TYPES = (CLIP,) — single output at slot 0.
    add(
        4,
        "LTXAVTextEncoderLoader",
        {
            "text_encoder": GEMMA3_TEXT_ENCODER,
            "ckpt_name": SULPHUR_CKPT,
            "device": "default",
        },
        "Text Encoder",
    )

    add(
        5,
        "LatentUpscaleModelLoader",
        {"model_name": SPATIAL_UPSCALER},
        "Spatial Upscaler",
    )

    # Audio VAE loader extracts the audio VAE block from the main checkpoint
    # (`ckpt_name` input, not `name`).
    add(6, "LTXVAudioVAELoader", {"ckpt_name": SULPHUR_CKPT}, "Audio VAE")

    # ---- Primitives ---------------------------------------------------------

    add(101, "PrimitiveString", {"value": prompt_text}, "Positive Prompt")
    add(102, "PrimitiveString", {"value": negative_text}, "Negative Prompt")

    # Frame rate is FLOAT-typed for LTXVConditioning + CreateVideo.
    add(110, "PrimitiveFloat", {"value": float(fps)}, "Frame Rate")
    # LTXVEmptyLatentAudio.frame_rate is INT-typed; project via LTXFloatToInt
    # (shipped by ComfyUI-LTXVideo, utility_nodes.py:38-49).
    add(111, "LTXFloatToInt", {"a": ref(110, 0)}, "Frame Rate (int)")

    # Width/Height the caller asked for is the FINAL output resolution. The
    # two-stage pipeline runs stage 1 at HALF that resolution, then the
    # LTXVLatentUpsampler x2-spatial-upscales to the final size before the
    # stage-2 refinement pass. This is exactly the pattern in Sulphur's shipped
    # workflows/ltx23_t2v distilled.json (math nodes #18 and #20 compute a/2)
    # AND in Lightricks' LTX-2.3_T2V_I2V_Two_Stage_Distilled.json
    # (EmptyLTXVLatentVideo widget [960, 544, 121, 1] for a 1920x1088 final).
    # Half-resolution must also be mod-32; we snap to the nearest mod-32 floor.
    stage1_width = (int(width) // 2 // 32) * 32
    stage1_height = (int(height) // 2 // 32) * 32

    add(120, "PrimitiveInt", {"value": stage1_width}, "Stage1 Width")
    add(121, "PrimitiveInt", {"value": stage1_height}, "Stage1 Height")
    add(122, "PrimitiveInt", {"value": int(frames)}, "Length")
    add(130, "PrimitiveInt", {"value": int(seed)}, "Seed")

    # ---- Conditioning -------------------------------------------------------

    # LTXAVTextEncoderLoader RETURN_TYPES = (CLIP,) → slot 0 (not slot 1).
    add(
        10,
        "CLIPTextEncode",
        {"text": ref(101, 0), "clip": ref(4, 0)},
        "Encode Positive",
    )
    add(
        11,
        "CLIPTextEncode",
        {"text": ref(102, 0), "clip": ref(4, 0)},
        "Encode Negative",
    )

    # LTXVConditioning → (CONDITIONING positive at slot 0, CONDITIONING negative at slot 1).
    add(
        12,
        "LTXVConditioning",
        {
            "positive": ref(10, 0),
            "negative": ref(11, 0),
            "frame_rate": ref(110, 0),
        },
        "AV Conditioning",
    )

    # ---- Stage-1 latent prep ------------------------------------------------

    add(
        20,
        "EmptyLTXVLatentVideo",
        {
            "width": ref(120, 0),
            "height": ref(121, 0),
            "length": ref(122, 0),
            "batch_size": 1,
        },
        "Empty Video Latent",
    )

    # batch_size is REQUIRED on LTXVEmptyLatentAudio per
    # comfy_extras/nodes_lt_audio.py:100-132. frame_rate is INT-typed.
    add(
        21,
        "LTXVEmptyLatentAudio",
        {
            "audio_vae": ref(6, 0),
            "frames_number": ref(122, 0),
            "frame_rate": ref(111, 0),
            "batch_size": 1,
        },
        "Empty Audio Latent",
    )

    add(
        22,
        "LTXVConcatAVLatent",
        {"video_latent": ref(20, 0), "audio_latent": ref(21, 0)},
        "Concat AV (stage 1)",
    )

    # ---- Stage-1 sampling ---------------------------------------------------
    # Sampler choice:
    #   • fast (distilled):    euler_ancestral_cfg_pp (Lightricks two-stage canonical, also Sulphur)
    #   • quality (full):      euler_ancestral        (Sulphur's non-distilled base workflow)
    # Sigma source:
    #   • fast (distilled):    ManualSigmas with Lightricks' DISTILLED_SIGMA_VALUES
    #   • quality (full):      LTXVScheduler(50, 2.72, 0.8, true, 0.0) — Sulphur base
    # CFG: 1.0 for distilled (mandatory; non-1 numerically unstable),
    #      3.6 for non-distilled (Sulphur base value).
    if apply_distill_lora:
        add(30, "KSamplerSelect", {"sampler_name": "euler_ancestral_cfg_pp"}, "Sampler 1")
        add(31, "ManualSigmas", {"sigmas": STAGE1_SIGMAS_FAST}, "Sigmas 1")
        stage1_cfg = 1.0
    else:
        # Quality mode uses Sulphur's canonical non-distilled base config.
        add(30, "KSamplerSelect", {"sampler_name": "euler_ancestral"}, "Sampler 1")
        add(
            31,
            "LTXVScheduler",
            {
                "steps": QUALITY_LTXVSCHEDULER_STEPS,
                "max_shift": QUALITY_LTXVSCHEDULER_MAX_SHIFT,
                "base_shift": QUALITY_LTXVSCHEDULER_BASE_SHIFT,
                "stretch": True,
                "terminal": QUALITY_LTXVSCHEDULER_TERMINAL,
                "latent": ref(22, 0),
            },
            "LTX Scheduler 1",
        )
        stage1_cfg = QUALITY_CFG_STAGE1
    add(32, "RandomNoise", {"noise_seed": ref(130, 0)}, "Noise 1")
    add(
        33,
        "CFGGuider",
        {
            "model": model_stage1,
            "positive": ref(12, 0),
            "negative": ref(12, 1),
            "cfg": stage1_cfg,
        },
        "Guider 1",
    )
    add(
        34,
        "SamplerCustomAdvanced",
        {
            "noise": ref(32, 0),
            "guider": ref(33, 0),
            "sampler": ref(30, 0),
            "sigmas": ref(31, 0),
            "latent_image": ref(22, 0),
        },
        "Sample Stage 1",
    )

    # ---- Stage-2 latent prep: separate, upscale VIDEO ONLY, recombine -------

    add(
        35,
        "LTXVSeparateAVLatent",
        {"av_latent": ref(34, 0)},
        "Separate AV (post-stage-1)",
    )

    # LTXVLatentUpsampler REQUIRES three inputs: samples, upscale_model, vae.
    add(
        40,
        "LTXVLatentUpsampler",
        {
            "samples": ref(35, 0),       # VIDEO latent only
            "upscale_model": ref(5, 0),
            "vae": ref(1, 2),            # VAE from CheckpointLoaderSimple slot 2
        },
        "Upsample Latent",
    )

    # Re-combine upsampled video + original stage-1 audio for stage-2 sampling.
    add(
        36,
        "LTXVConcatAVLatent",
        {"video_latent": ref(40, 0), "audio_latent": ref(35, 1)},
        "Concat AV (stage 2)",
    )

    # ---- Stage-2 sampling ---------------------------------------------------
    # Stage-2 is canonical for both fast and quality modes:
    #   sampler        = euler_cfg_pp        (Lightricks two-stage example;
    #                                          Sulphur ships `lcm` here but we
    #                                          reject lcm — it's noise-prediction
    #                                          paradigm, LTX-2 is flow-matching)
    #   sigmas         = ManualSigmas "0.85, 0.7250, 0.4219, 0.0"
    #                    (both Sulphur AND Lightricks agree)
    #   cfg            = 1.0                  (both authorities)
    #   refresh seed   = fixed value seed+1   (deterministic refine)
    add(41, "KSamplerSelect", {"sampler_name": "euler_cfg_pp"}, "Sampler 2")
    add(42, "ManualSigmas", {"sigmas": STAGE2_SIGMAS_FAST}, "Sigmas 2")
    # Stage-2 noise seed: caller_seed + 1.
    # Sulphur's shipped workflows hardcode a fixed `42` here, and Lightricks'
    # two-stage example does the same. Empirically, switching to fixed=42 in
    # our pipeline (full Sulphur FP8 model + Lightricks canonical samplers)
    # coincided with an avcodec_send_frame() EINVAL audio-encoder failure
    # at the SaveVideo node. seed+1 — used in Test B (proven-working
    # configuration) — keeps stage 2's noise derived from the caller seed and
    # the audio path stable. The maintainer's hardcoded 42 in their shipped
    # workflow is tied to their LTX-2.3-base + sulphur_final LoRA recipe, not
    # ours. Future investigation could bisect why fixed=42 specifically broke
    # audio; for now, seed+1 is the validated value.
    add(43, "RandomNoise", {"noise_seed": int(seed) + 1}, "Noise 2")
    add(
        44,
        "CFGGuider",
        {
            "model": model_stage2,
            "positive": ref(12, 0),
            "negative": ref(12, 1),
            "cfg": QUALITY_CFG_STAGE2,  # 1.0 — same for both modes
        },
        "Guider 2",
    )
    add(
        45,
        "SamplerCustomAdvanced",
        {
            "noise": ref(43, 0),
            "guider": ref(44, 0),
            "sampler": ref(41, 0),
            "sigmas": ref(42, 0),
            "latent_image": ref(36, 0),
        },
        "Sample Stage 2",
    )

    # ---- Decode + save ------------------------------------------------------
    add(
        50,
        "LTXVSeparateAVLatent",
        {"av_latent": ref(45, 0)},
        "Separate AV (final)",
    )

    # LTXVTiledVAEDecode inputs (verified against
    # ComfyUI-LTXVideo/tiled_vae_decode.py:13-28):
    #   vae, latents, horizontal_tiles (1-6), vertical_tiles (1-6),
    #   overlap (1-8), last_frame_fix (bool),
    #   working_device, working_dtype (both default to "auto").
    add(
        51,
        "LTXVTiledVAEDecode",
        {
            "vae": ref(1, 2),
            "latents": ref(50, 0),
            "horizontal_tiles": VAE_HORIZONTAL_TILES,
            "vertical_tiles": VAE_VERTICAL_TILES,
            "overlap": VAE_TILE_OVERLAP,
            "last_frame_fix": False,
            "working_device": "auto",
            "working_dtype": "auto",
        },
        "VAE Decode Video",
    )

    add(
        52,
        "LTXVAudioVAEDecode",
        {"samples": ref(50, 1), "audio_vae": ref(6, 0)},
        "VAE Decode Audio",
    )

    # CreateVideo.fps is FLOAT-typed.
    add(
        53,
        "CreateVideo",
        {
            "images": ref(51, 0),
            "audio": ref(52, 0),
            "fps": ref(110, 0),
        },
        "Create Video",
    )

    add(
        54,
        "SaveVideo",
        {
            "video": ref(53, 0),
            "filename_prefix": output_filename_prefix,
            "format": "mp4",
            "codec": "h264",
        },
        "Save Video",
    )

    return nodes


# ─────────────────────────── Request models ──────────────────────────────────


class GenerateRequest(BaseModel):
    """Body of POST /generate (text-to-video)."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Free-text description of the desired video.",
    )
    seed: int | None = Field(
        None,
        description="If null, a fresh 63-bit random seed is generated.",
    )
    duration_seconds: float = Field(
        20.0,
        ge=1.0,
        le=20.0,
        description=(
            "Output duration in seconds. Hard-capped at 20.0 — Lightricks' "
            "validated training-distribution maximum for LTX-2.3 "
            "(arXiv:2601.03233 §6.3 + docs.ltx.video/models). Frame count is "
            "round(duration*fps) snapped to the nearest 8n+1."
        ),
    )
    width: int = Field(1920, ge=512, le=1920, description="Multiple of 32.")
    height: int = Field(1088, ge=320, le=1088, description="Multiple of 32.")
    fps: int = Field(
        24,
        description=(
            "Frame rate. 24/25 fps unlocks the full 20-second ceiling. "
            "48/50 fps caps duration at 10 s per Lightricks' API matrix."
        ),
    )
    mode: str = Field(
        "quality",
        pattern="^(fast|quality)$",
        description=(
            "quality = Sulphur's non-distilled base config (50-step "
            "LTXVScheduler + euler_ancestral + CFG=3.6 stage 1, 3-step refine "
            "stage 2). Same Sulphur FP8 mixed weights, same VRAM peak (~30 "
            "GiB), ~14 min (measured 853s on RTX 5090 Test B) on RTX 5090. This is the DEFAULT — quality is "
            "the headline target. "
            "fast    = Sulphur's distilled config with TenStrip's "
            "fro90_ceil72_condsafe LoRA at strengths 0.7+0.5 + 8-step "
            "DISTILLED_SIGMA_VALUES + euler_ancestral_cfg_pp. Same VRAM, "
            "~7 min on RTX 5090 (measured 433s). Opt in by passing mode='fast' for fast "
            "iteration / preview."
        ),
    )
    enhance_prompt: bool = Field(
        True,
        description=(
            "If true, the prompt is rewritten by the Sulphur prompt "
            "enhancer (fine-tuned Qwen 3.5-9B hybrid + qwen3vl_merger mmproj, "
            "CPU llama.cpp) before generation. Adds 3-8 s latency, "
            "zero VRAM cost."
        ),
    )
    negative_prompt: str = Field(
        "pc game, console game, video game, cartoon, childish, ugly, low quality, "
        "blurry, distorted, watermark, jpeg artifacts, text overlay",
        description="Negative conditioning prompt.",
    )


# ─────────────────────────── Helpers ──────────────────────────────────────────


def _snap_frames(target_frames: int) -> int:
    """LTX-2.3 video VAE constraint: frame_count must be 8n+1.
    Snap to the NEAREST valid value (not floor) so that, e.g.,
    duration=20s @ fps=24 → target=480 → snapped=481 (closer than 473)."""
    if target_frames < 1:
        return 1
    # round((target_frames - 1) / 8) finds the nearest n; +0.5 trick avoids
    # banker's rounding on .5 ties and biases up (preferring slightly-longer
    # output over slightly-shorter, which matches user expectation).
    n_nearest = (target_frames - 1 + 4) // 8
    return n_nearest * 8 + 1


def _snap_dim_32(target: int) -> int:
    """LTX-2.3 video VAE constraint: width/height divisible by 32."""
    return (target // 32) * 32


async def _enhance_prompt(client: httpx.AsyncClient, raw: str) -> str:
    """Round-trip through the Sulphur prompt enhancer.

    Belt-and-suspenders against the Sulphur fine-tune's tendency to emit
    gratuitous <think> blocks despite Qwen3.5-9B's upstream default being
    thinking-off. An earlier version of this function used a 512-token
    budget with NO thinking suppression — the model filled the budget with
    reasoning, returned content="", and we silently fell back to the raw
    (unenhanced) prompt on every call. Three layers of defense are now in
    place:

      1. Server-side `--reasoning off --reasoning-budget 0` flags in
         Dockerfile.enhancer.
      2. `chat_template_kwargs: {"enable_thinking": false}` per-request
         (this function).
      3. `reasoning_format=deepseek` routes any leakage to the
         `reasoning_content` field; we explicitly check + log if `content`
         arrives empty (rather than silently surfacing the raw prompt as
         "enhanced").

    Sampling envelope mirrors the Dockerfile.enhancer CMD so each request
    is self-describing for debugging, and overrides defend against any
    server-side drift.

    Read timeout of 180s — at ~6.2 tok/s on a modern desktop CPU and the
    1024-token --predict ceiling, the worst-case round-trip is ~170s.
    """
    try:
        resp = await client.post(
            f"{ENHANCER_URL}/v1/chat/completions",
            json={
                "model": "sulphur-enhancer",
                "messages": [{"role": "user", "content": raw}],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature":      0.7,
                "top_p":            0.8,
                "top_k":            20,
                "min_p":            0.0,
                "repeat_penalty":   1.0,
                "presence_penalty": 0.0,
                "max_tokens":       1024,
                "stream":           False,
            },
            timeout=httpx.Timeout(180.0, connect=10.0, read=180.0),
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        enhanced = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            # Belt-and-suspenders regression detector: with --reasoning off +
            # --reasoning-budget 0 + chat_template_kwargs.enable_thinking=false,
            # `reasoning_content` should always be empty. Non-empty here means
            # the server-side flag was lost or the template default drifted.
            # Log loud so the regression is caught immediately rather than
            # silently degrading enhancement quality.
            log.warning(
                "prompt enhancer leaked thinking content (%d chars) despite "
                "thinking-off configuration — investigate server flags and "
                "chat_template_kwargs",
                len(reasoning),
            )
        if not enhanced:
            log.warning(
                "prompt enhancer returned empty content "
                "(reasoning_content=%d chars); falling back to raw prompt",
                len(reasoning),
            )
            return raw
        return enhanced.strip('"').strip("'") or raw
    except Exception as exc:  # noqa: BLE001
        log.warning("prompt enhancer failed (%s); falling back to raw prompt", exc)
        return raw


async def _submit_workflow(
    client: httpx.AsyncClient, workflow: dict[str, Any], client_id: str
) -> str:
    """POST workflow to ComfyUI /prompt → returns prompt_id."""
    body = {"prompt": workflow, "client_id": client_id}
    resp = await client.post(f"{COMFYUI_URL}/prompt", json=body, timeout=SUBMIT_HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI rejected workflow (HTTP {resp.status_code}): {resp.text}",
        )
    data = resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=502, detail=f"ComfyUI returned no prompt_id: {data}")
    return prompt_id


async def _poll_completion(
    client: httpx.AsyncClient, prompt_id: str
) -> dict[str, Any]:
    """Poll /history/{prompt_id} until the workflow completes or errors."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            resp = await client.get(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=POLL_HTTP_TIMEOUT
            )
            if resp.status_code == 200:
                history = resp.json()
                entry = history.get(prompt_id)
                if entry:
                    status = entry.get("status", {})
                    if status.get("completed"):
                        return entry
                    if status.get("status_str") == "error":
                        raise HTTPException(
                            status_code=500,
                            detail=f"ComfyUI workflow errored: {status}",
                        )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("poll error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise HTTPException(
        status_code=504,
        detail=f"Generation timed out after {POLL_TIMEOUT_SECONDS}s",
    )


def _resolve_output_file(history_entry: dict[str, Any], output_id: str) -> Path:
    """Locate the MP4 produced by SaveVideo within OUTPUTS_DIR.

    NOTE: ComfyUI v0.21.0 SaveVideo emits via comfy_api.latest._ui.PreviewVideo,
    which serialises files under the key ``images`` (not ``videos``) inside the
    /history outputs payload. See comfy_api/latest/_ui.py:432-433.
    """
    outputs = history_entry.get("outputs", {})
    for node_outputs in outputs.values():
        for vid in node_outputs.get("images") or []:
            filename = vid.get("filename")
            subfolder = vid.get("subfolder", "")
            if filename and output_id in filename:
                full = OUTPUTS_DIR / subfolder / filename
                if full.exists():
                    return full
    # Defensive fallback: filesystem glob by output_id stem.
    matches = list(OUTPUTS_DIR.rglob(f"*{output_id}*.mp4"))
    if matches:
        return matches[0]
    raise HTTPException(
        status_code=500,
        detail=f"could not locate output for output_id={output_id} under {OUTPUTS_DIR}",
    )


# ─────────────────────────── App ──────────────────────────────────────────────

app = FastAPI(
    title="solphur2",
    description=(
        "Sulphur-2-base (LoRA-free FP8-mixed LTX-2.3-22B fine-tune) on "
        "NVIDIA GeForce RTX 5090 (Blackwell sm_120). 127.0.0.1 only."
    ),
    version="1.0.0",
)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Cheap liveness check: ComfyUI + enhancer must both be reachable."""
    async with httpx.AsyncClient() as client:
        comfy_ok = False
        enh_ok = False
        try:
            r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=3.0)
            comfy_ok = r.status_code == 200
        except Exception:  # noqa: BLE001
            pass
        try:
            r = await client.get(f"{ENHANCER_URL}/health", timeout=3.0)
            enh_ok = r.status_code == 200
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse(
        status_code=200 if (comfy_ok and enh_ok) else 503,
        content={"comfyui": comfy_ok, "enhancer": enh_ok, "ok": comfy_ok and enh_ok},
    )


@app.get("/")
async def index() -> PlainTextResponse:
    return PlainTextResponse(
        "solphur2 API on 127.0.0.1\n"
        "  POST /generate       text-to-video (JSON body)\n"
        "  POST /generate/i2v   image-to-video (multipart) — stub in v1\n"
        "  GET  /healthz        component health\n"
    )


@app.post("/generate")
async def generate(req: GenerateRequest) -> FileResponse:
    """Text-to-video — returns one MP4."""
    width = _snap_dim_32(req.width)
    height = _snap_dim_32(req.height)
    target_frames = int(round(req.duration_seconds * req.fps))
    frames = _snap_frames(target_frames)
    seed = req.seed if req.seed is not None else secrets.randbits(63)
    output_id = uuid.uuid4().hex
    filename_prefix = f"video/solphur2_{output_id}"

    log.info(
        "generate: prompt=%r mode=%s %dx%d frames=%d (%.2fs @ %dfps) seed=%d enhance=%s",
        req.prompt[:80],
        req.mode,
        width,
        height,
        frames,
        frames / req.fps,
        req.fps,
        seed,
        req.enhance_prompt,
    )

    client_id = uuid.uuid4().hex

    async with httpx.AsyncClient() as client:
        prompt_text = (
            await _enhance_prompt(client, req.prompt) if req.enhance_prompt else req.prompt
        )
        if req.enhance_prompt:
            log.info("enhanced prompt: %s", prompt_text[:200])

        workflow = _build_workflow(
            prompt_text=prompt_text,
            negative_text=req.negative_prompt,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            fps=req.fps,
            mode=req.mode,
            output_filename_prefix=filename_prefix,
        )

        t0 = time.monotonic()
        prompt_id = await _submit_workflow(client, workflow, client_id)
        log.info("submitted prompt_id=%s output_id=%s", prompt_id, output_id)
        history = await _poll_completion(client, prompt_id)
        elapsed = time.monotonic() - t0
        log.info("completed in %.1fs", elapsed)

    output_path = _resolve_output_file(history, output_id)
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_path.name,
        headers={
            "X-Solphur2-PromptId": prompt_id,
            "X-Solphur2-OutputId": output_id,
            "X-Solphur2-Seed": str(seed),
            "X-Solphur2-Frames": str(frames),
            "X-Solphur2-Resolution": f"{width}x{height}",
            "X-Solphur2-Fps": str(req.fps),
            "X-Solphur2-Mode": req.mode,
            "X-Solphur2-ElapsedSeconds": f"{elapsed:.1f}",
            "X-Solphur2-PromptOriginal": req.prompt[:200],
            "X-Solphur2-PromptFinal": prompt_text[:200],
        },
    )


@app.post("/generate/i2v")
async def generate_i2v(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    duration_seconds: float = Form(20.0),
    fps: int = Form(24),
    mode: str = Form("fast"),
    seed: int | None = Form(None),
    enhance_prompt: bool = Form(True),
) -> FileResponse:
    """Image-to-video (stub in v1).

    The workflow builder above is t2v-shaped; a follow-up commit will add the
    LTXVImgToVideoConditionOnly + LoadImage nodes and patch the conditioning
    chain to consume the uploaded image as the first frame.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "i2v not wired in this initial release. Use POST /generate for t2v. "
            "The image upload field is accepted so the surface is stable for v1.1."
        ),
    )
