# solphur2_setup

Reproducible, Docker-based deployment of `SulphurAI/Sulphur-2-base` (an
uncensored, FP8-mixed fine-tune of `Lightricks/LTX-2.3-22B`) on an NVIDIA
GeForce RTX 5090 (Blackwell architecture, compute capability sm_120), with a
programmatic HTTP API on `127.0.0.1`.

Target hardware (validated 2026-05-11):

- NVIDIA GeForce RTX 5090 graphics card (Blackwell, sm_120)
- 32607 MiB GDDR7 VRAM
- NVIDIA proprietary driver 595.58.03 (open kernel module variant)
- Host CUDA 13.0.3 + NVIDIA Container Toolkit 1.19.0-1
- Docker CE 29.4.1 + Compose v5.1.3
- Ubuntu 24.04.3 LTS (Noble Numbat)
- 64 GiB system RAM, 8 GiB swap

Capability ceiling on this hardware:

- **1920 × 1088 × 481 frames @ 24 fps (1080p × 20 s)** in `fast` mode (Sulphur's
  shipped 6+3-step distilled two-stage workflow with the FP8 mixed checkpoint).
  Peak ~30 GiB VRAM, ~9 minutes per video (matches `bmgjet`'s measurement on
  identical hardware, HF Discussion #16 on `Lightricks/LTX-2.3`).
- The same envelope in `quality` mode (no distill LoRA, 25-step full sampling)
  takes ~30 minutes and produces marginally higher detail at the cost of
  stochastic regression on motion coherence.

This is Lightricks' documented Fast/1080p maximum (`arXiv:2601.03233` §6.3
and `docs.ltx.video/models`). The model's audio-visual training distribution
caps at 20 seconds; longer outputs require I2V chaining.

## Architecture

```
                 ┌────────────────────────────────────────┐
   localhost     │  ./outputs (bind-mounted)              │
   :8000  ◀──────│  ┌────────────┐  ┌──────────────────┐  │
                 │  │ solphur2/  │  │ solphur2/comfyui │  │
                 │  │   api      │──│  ┌─────────────┐ │  │  ◀── GPU
   localhost     │  │ (FastAPI)  │  │  │  Sulphur    │ │  │
   :8188  ◀──────│  └──────┬─────┘  │  │  FP8 mixed  │ │  │
                 │         │        │  │  + distill  │ │  │
                 │         │        │  │  LoRA       │ │  │
                 │         │        │  └─────────────┘ │  │
                 │         │        └──────────────────┘  │
                 │         │                              │
   localhost     │         │        ┌──────────────────┐  │
   :8080  ◀──────│         └──────▶ │ solphur2/        │  │
                 │                  │   enhancer       │  │  ◀── CPU only
                 │                  │ (llama.cpp +     │  │
                 │                  │  Sulphur Q8 GGUF │  │
                 │                  │  Qwen3-VL ~9 B)  │  │
                 │                  └──────────────────┘  │
                 │  Docker network: solphur2net (bridge)  │
                 └────────────────────────────────────────┘

   Inbound surface: 127.0.0.1 only on three ports (8000, 8188, 8080).
   Inter-service traffic stays inside the solphur2net bridge.
```

The video model lives in `solphur2/comfyui` and consumes the GPU.
The Sulphur prompt enhancer lives in `solphur2/enhancer` and runs on the
**CPU** (llama.cpp, `GGML_CUDA=OFF`) so it never competes with the video
transformer for VRAM. Cost: ~10 GiB system RAM, ~5–8 s of CPU inference per
prompt rewrite.

The `solphur2/api` FastAPI service is the only public surface a caller talks
to. It enhances → submits → polls → returns the MP4 in a single HTTP round
trip. Endpoint:

```
POST http://127.0.0.1:8000/generate
Content-Type: application/json
{ "prompt": "<free text>", "duration_seconds": 20, "mode": "fast" }
→ 200 OK
  Content-Type: video/mp4
  <MP4 bytes>
```

See `api/server.py` for the full request model (seed, fps, resolution,
mode toggle, enhance toggle, negative prompt).

## Reproducibility

Every package version, every git SHA, every model file SHA-256 lives in
`versions.env` — the single source of truth. Same input, same output:

| Layer | Pin |
|---|---|
| NVIDIA Container base image | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| Python | 3.12 (apt) |
| PyTorch | `2.11.0+cu130` (sm_120 first-class binary target) |
| Triton | 3.6.0 |
| SageAttention | `v2.2.0` (built from source for sm_120) |
| ComfyUI core | `v0.21.0` (commit `52976f3ea33c`) |
| ComfyUI-LTXVideo | `229437c6b657` (Lightricks master, 2026-05-11) |
| ComfyUI-KJNodes | `1.4.0` (commit `1252598b41be`) |
| ComfyMath | `be9beae9eeeb049db35e3ddd35ed1ed0058d6b59` |
| llama.cpp | tag `b9106` (CPU only) |
| transformers | 5.8.0 |
| diffusers | 0.38.0 |
| FastAPI | 0.136.1 |
| uvicorn | 0.46.0 |
| httpx | 0.28.1 |

All model files SHA-256-verified by `scripts/download_models.sh`:

| File | Size | Subdir |
|---|---:|---|
| `sulphur_dev_fp8mixed.safetensors` | 27.2 GiB | `checkpoints/` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.08 GiB | `loras/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MiB | `upscale_models/` |
| `gemma_3_12B_it_fp8_scaled.safetensors` | 12.3 GiB | `text_encoders/` |
| `sulphur_prompt_enhancer_model-q8_0.gguf` | 8.87 GiB | `prompt_enhancer/` |
| `mmproj-BF16.gguf` | 879 MiB | `prompt_enhancer/` |

Total: ~58 GiB on disk.

## Quickstart

From a fresh checkout, with Docker + NVIDIA Container Toolkit on the host:

```bash
# One command:
bash scripts/up.sh

# After ~15 min (mostly SageAttention compile and model downloads), generate:
curl -fsS -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a cinematic close-up of a foggy cobblestone alley at dawn",
         "duration_seconds":20,"mode":"fast"}' \
    --output ~/Videos/test.mp4
```

`scripts/up.sh` flags:

- `--skip-models` skip model download (assume `./models/` is populated).
- `--skip-build` skip image build (reuse cached layers).

Bring everything down with `docker compose down`. Models, outputs, and built
images persist; bring back up in seconds.

## Why the choices

### FP8-mixed, not BF16 or Q4 GGUF

The Sulphur maintainer ships three checkpoint forms: `sulphur_dev_bf16`
(42.97 GiB), `sulphur_distil_bf16` (42.97 GiB), and `sulphur_dev_fp8mixed`
(27.16 GiB). BF16 does not fit a 32 GiB single GPU. Community Q8/Q6/Q4 GGUFs
exist but quantize the model further from the FP8-native Blackwell tensor
core path. FP8 mixed preserves quality-sensitive layers (norms, gates,
embeddings) in BF16 and casts only the linear-layer weights to FP8 — the
quality/VRAM tradeoff that bmgjet's published `1080p × 20 s @ 24 fps in
547 s, peak 29.8 GiB` measurement validates.

### Sulphur full model, not Sulphur LoRA on LTX-2.3 base

Sulphur ships both `sulphur_dev_fp8mixed.safetensors` (the full fine-tuned
model, ready to use) and `sulphur_lora_rank_768.safetensors` (a LoRA you
can apply on top of base LTX-2.3). The maintainer says "use the lora or
use the full models, don't use both at the same time." We use the full
model — fewer moving parts, no LoRA stacking bugs (e.g. the FP8 + rank-768
artifact regression that `masterkwic2` documented on HF discussion #3).

### Distill LoRA from Lightricks (rank 384) on top, for `fast` mode

`ltx-2.3-22b-distilled-lora-384-1.1.safetensors` at strength 0.5 turns the
non-distilled `sulphur_dev_fp8mixed` into a 6+3-step pipeline — Sulphur's
shipped `workflows/ltx23_t2v distilled.json` is exactly this configuration.
Without the distill LoRA, the full model wants 20–50 sampling steps.

### Multi-stage tiled inference

LTX-2.3 is trained for a max single-pass latent of ~20 temporal tokens
(~5 s at 24 fps). For longer / higher-resolution outputs Lightricks
documents a multi-stage pipeline (`arXiv:2601.03233` §4.2): generate at a
~0.5 MP base latent (we use 960 × 544 × 481 = ~0.5 MP × full duration),
then upsample the **video** latent x2 spatially via the dedicated
`ltx-2.3-spatial-upscaler-x2-1.1` model, then refine in a second sampling
pass. Without this, the model sees out-of-distribution sequence lengths
and emits temporal drift.

### SageAttention 2.2.0 patched per-model, not via the CLI flag

ComfyUI's `--use-sage-attention` global flag calls `import sageattention`
at startup and exits the process with -1 if the import fails (see
`comfy/ldm/modules/attention.py:28-33`). Inside a container, that's a
hard-blocking constraint. We instead use `PathchSageAttentionKJ` (from
`kijai/ComfyUI-KJNodes`) which patches the per-model
`transformer_options["optimized_attention_override"]` at workflow time —
finer granularity, no startup-blocking import side-effect, and the
per-model override takes precedence anyway (`attention.py:127-143`).

### Gemma3 text encoder via the Comfy-Org single-file fp8_scaled

The raw `Lightricks/LTX-2/text_encoder/*.safetensors` (multi-shard
multimodal Gemma3) triggers the regression documented in GitHub issue
`comfy-org/ComfyUI#11920` on Blackwell + FP8 transformer: the path through
`comfy/text_encoders/llama.py` for `Gemma3ForCausalLM` raises
`NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`.
Workaround: use `Comfy-Org/ltx-2`'s pre-packaged single-file
`gemma_3_12B_it_fp8_scaled.safetensors` (12.3 GiB), which uses the
text-only Gemma3 code path. The fix landed in ComfyUI core commit
`e2ddf28d` (2026-03-31), included in the v0.21.0 tag we pin.

### ComfyUI-KJNodes ≥ commit `0e173bbfa9de`

That commit ("LTX2_NAG: Better dtype cast") routes through
`manual_cast_dtype` instead of forcing a global FP8→BF16 cast, fixing the
companion crash on Blackwell. We pin tag `1.4.0` which contains it.

## Project layout

```
solphur2_setup/
├── README.md                    this file
├── LICENSE                      MIT for the orchestration; models keep their upstream licenses
├── versions.env                 single source of truth for every pin
├── docker-compose.yml           three services, 127.0.0.1 only
├── api/
│   └── server.py                single-file FastAPI gateway + workflow builder
├── docker/
│   ├── Dockerfile.comfyui       ComfyUI v0.21.0 + LTX-2.3 + SageAttention for sm_120
│   ├── Dockerfile.enhancer      llama.cpp CPU build for the Sulphur prompt enhancer
│   └── Dockerfile.api           thin FastAPI runtime
├── scripts/
│   ├── download_models.sh       SHA-256-pinned, idempotent fetcher
│   └── up.sh                    one-command bring-up
├── models/                      ← created at first run; gitignored
└── outputs/                     ← created at first run; gitignored
```

## Tuning knobs

The API exposes the typical ones (resolution, duration, fps, seed, mode).
Server-side defaults (in `api/server.py`):

| Knob | `fast` mode | `quality` mode |
|---|---|---|
| Distill LoRA applied? | yes, strength 0.5 | no |
| Stage-1 sampler | `euler_cfg_pp` | `euler_cfg_pp` |
| Stage-1 sigmas | `0.85, 0.7933, 0.68, 0.51, 0.2833, 0.0` (6 steps) | 25-step linear |
| Stage-2 sampler | `euler_ancestral_cfg_pp` | `euler_ancestral_cfg_pp` |
| Stage-2 sigmas | `0.8, 0.55, 0.25, 0.0` (3 steps) | 6-step linear |
| CFG | 1.0 (both stages) | 1.0 |
| VAE tiling | 2×2 spatial tiles, overlap 6, `last_frame_fix=False` | same |
| SageAttention | per-model patch (`PathchSageAttentionKJ`) | same |

## License

This project's orchestration code is MIT licensed. The Sulphur-2-base model
(SulphurAI), LTX-2.3 (Lightricks), and Gemma3 (Google) all carry their own
upstream licenses — see each repo's `LICENSE.txt`. SulphurAI's model is
described by its author as "uncensored" — operate accordingly.
