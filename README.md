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

### Two envelopes — the Sulphur tested envelope vs. the LTX-2.3 base ceiling

The video model is a **fine-tune of LTX-2.3-22B** by SulphurAI. LTX-2.3 base
supports up to 1080p × 20s × 24fps; the Sulphur fine-tune was tested on a
smaller envelope. Two distinct numbers therefore apply:

**Sulphur's tested envelope (the production default)** — what the maintainer
shipped and verified:

- **1280 × 704 × 241 frames @ 24 fps (≈720p × 10 s)** — Sulphur's shipped
  `workflows/ltx23_t2v distilled.json` runs stage-2 at 1344×768 × 241 frames;
  our 1280×704 lands within the same megapixel class with a cleaner 16:9
  mod-32 aspect.
- Peak ~25 GiB VRAM, ~2 minutes per video on the RTX 5090 (measured Test A
  smoke at 1280×704×113f = 126s; 10s clip scales to ~2-3 minutes).
- This is the **default** the API returns when the caller doesn't override.

**LTX-2.3 base ceiling (opt-in via larger request fields)** — what the
underlying architecture allows but the Sulphur fine-tune was NOT verified at:

- **1920 × 1088 × 481 frames @ 24 fps (1080p × 20 s)** in `fast` mode,
  ~26.2 GiB peak VRAM, ~7 min wall-clock (measured Test A headline).
- Same envelope in `quality` mode (no distill LoRA, 50-step LTXVScheduler):
  same VRAM peak, ~14 min wall-clock (measured Test B).
- The API will accept requests up to this ceiling. The server logs
  `OOD-Sulphur` warnings when the request exceeds Sulphur's shipped
  envelope, so quality regressions are observable in the logs.

Why this two-tier design: per a dedicated sub-agent research pass, Sulphur's
training data and tested operating point are around `~1024×576 × 25-125
frames @ 24/25 fps` (inferred from the maintainer's shipped workflow + the
Musubi-tuner LTX-2.3 community standard bucket + TenStrip's 10Eros sister
model). 1920×1088 × 481 frames is 2× the stage-1 area and 2× the frame
count the fine-tune was tested at — within LTX-2.3 native, but out-of-
distribution for Sulphur.

### Modality: t2v is Sulphur's primary path, i2v is inherited

`SulphurAI/Sulphur-2-base` is **primarily a text-to-video fine-tune**.
It inherits image-to-video capability from the LTX-2.3 base architecture
(the model can accept image conditioning), but the maintainer's own
recommendation in the model card is unambiguous about which model to use
for which job:

- **For text-to-video** (`/generate`): `Sulphur-2-base` is the
  maintainer's primary published artifact. The Sulphur shipped workflows
  `ltx23_t2v distilled.json` and `ltx23_t2v base.json` are the maintainer's
  tested t2v paths. **Our `/generate` endpoint maps directly to this path.**
- **For image-to-video**: the Sulphur model card explicitly recommends
  [`TenStrip/LTX2.3-10Eros`](https://huggingface.co/TenStrip/LTX2.3-10Eros)
  — verbatim: *"his i2v merge of sulphur 2, highly recommend for i2v"*.
  TenStrip's "10Eros" is a *merge* of Sulphur-2-base with an additional
  i2v-leaning checkpoint, published specifically because Sulphur-2-base
  alone was perceived to need supplementation for serious i2v use.

Implications for solphur2:

- **`/generate` (t2v)** is the headline use case. Sulphur-2-base is the
  right model for this endpoint. No change recommended.
- **`/generate/i2v`** is currently a stub (returns 501) precisely because
  if/when we wire i2v, the maintainer's own steer is to use TenStrip's
  10Eros merge rather than Sulphur-2-base for that path. Sulphur-2-base
  will *work* for i2v (LTX-2.3 base supports it natively, Sulphur's
  shipped `ltx23_i2v distilled.json` workflow exercises it), but the
  10Eros merge produces materially better results per the maintainer's
  guidance. Adding i2v would mean a separate model download + a separate
  workflow configuration, not a flag-flip on the existing path.

We do not claim i2v is Sulphur-2-base's primary intended use. Our
deployment fits Sulphur's strongest path.

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
                 │                  │  Qwen 3.5-9B +   │  │
                 │                  │  qwen3vl_merger) │  │
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
{ "prompt": "<free text>" }
# Server-side defaults: 1280×704 × 10s @ 24fps in quality mode (Sulphur's
# tested envelope). Override per request to opt up to the LTX-2.3 ceiling
# (1920×1088 × 20s) — the API will log an OOD-Sulphur warning since
# Sulphur was not verified at the larger envelope.
# Override examples:
#   {"prompt": "...", "duration_seconds": 20}                  ← LTX-2.3 max duration
#   {"prompt": "...", "width": 1920, "height": 1088}           ← LTX-2.3 max resolution
#   {"prompt": "...", "mode": "fast"}                          ← speed over quality
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
| ComfyUI-KJNodes | commit `1252598b41be959776f1208428b05b323b3fe17a` (the "version 1.4.0" commit; the repo ships no git tags) |
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
| `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` | 631 MiB | `loras/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MiB | `latent_upscale_models/` |
| `gemma_3_12B_it_fp8_scaled.safetensors` | 12.3 GiB | `text_encoders/` |
| `sulphur_prompt_enhancer_model-q8_0.gguf` | 8.87 GiB | `prompt_enhancer/` |
| `mmproj-BF16.gguf` | 879 MiB | `prompt_enhancer/` |

Total: ~58 GiB on disk.

## Quickstart

From a fresh checkout, with Docker + NVIDIA Container Toolkit on the host,
**all operational commands are scripts** — never raw `docker compose ...`
calls duplicated here:

```bash
bash scripts/up.sh           # validate host → download models → build → bring up → wait healthy
bash scripts/test.sh         # smoke (~2 min) + headline (~14 min) end-to-end validation
bash scripts/down.sh         # graceful teardown (preserves images, models, outputs)
bash scripts/clean.sh        # remove containers + images + cache + outputs (keeps models)
bash scripts/clean.sh --all  # also wipe ./models/ (forces re-download next up)
```

To actually generate a video against a live stack:

```bash
curl -fsS -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a cinematic close-up of a foggy cobblestone alley at dawn"}' \
    --output ~/Videos/headline.mp4
```

The default mode is **quality** (Sulphur's non-distilled 50-step pipeline, ~14 min). To prefer
speed over quality, add `"mode":"fast"` to the request body (~7 min, distill LoRA path).

### Script reference

| Script | What it does | Idempotent? |
|---|---|---|
| `scripts/up.sh` | host check → download models (delegates to `download_models.sh`) → build (delegates to `build.sh`) → `docker compose up -d` → wait healthy | yes |
| `scripts/build.sh [--no-cache]` | pure `docker compose build` — separated so iterations don't re-run model downloads or healthcheck waits | yes (incremental) |
| `scripts/download_models.sh` | SHA-256-pinned model fetcher (resumable, idempotent) | yes |
| `scripts/test.sh [--smoke-only/--headline-only/--enhance]` | end-to-end POST `/generate` round-trips. Writes MP4s to `./test_artifacts/`. | yes |
| `scripts/bench.py` | binary-search sweep over (resolution × duration × mode) for empirical VRAM/time envelope | yes |
| `scripts/down.sh` | stop + remove containers and network. Images and models survive. | yes |
| `scripts/clean.sh [--all]` | full cleanup — containers + images + build cache + outputs (and optionally models) | yes |

Every script is `set -Eeuo pipefail` and fails loudly on the first unexpected condition. No hidden
side effects, no silent retries.

## Why the choices

### FP8-mixed, not BF16 or Q4 GGUF

The Sulphur maintainer ships three checkpoint forms: `sulphur_dev_bf16`
(42.97 GiB), `sulphur_distil_bf16` (42.97 GiB), and `sulphur_dev_fp8mixed`
(27.16 GiB). BF16 does not fit a 32 GiB single GPU. Community Q8/Q6/Q4 GGUFs
exist but quantize the model further from the FP8-native Blackwell tensor
core path. FP8 mixed preserves quality-sensitive layers (norms, gates,
embeddings) in BF16 and casts only the linear-layer weights to FP8 — the
quality/VRAM tradeoff that our same-hardware measurement validates: with the
canonical two-stage pipeline (half-resolution base + x2 spatial latent
upsample + 3-step refine), 1920×1088 × 481 frames @ 24 fps in fast mode
peaks at ~26.2 GiB VRAM and finishes in roughly 10 minutes on this RTX 5090.
The same envelope in quality mode (no distill LoRA, 50-step LTXVScheduler)
uses the same VRAM peak and finishes in ~14 min.

### Sulphur full model, not Sulphur LoRA on LTX-2.3 base

Sulphur ships both `sulphur_dev_fp8mixed.safetensors` (the full fine-tuned
model, ready to use) and `sulphur_lora_rank_768.safetensors` (a LoRA you
can apply on top of base LTX-2.3). The maintainer says "use the lora or
use the full models, don't use both at the same time." We use the full
model — fewer moving parts, no LoRA stacking bugs (e.g. the FP8 + rank-768
artifact regression that `masterkwic2` documented on HF discussion #3).

### Distill LoRA: TenStrip's `fro90_ceil72_condsafe` variant, per-stage strengths

`ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` at
strength 0.7 stage 1, 0.5 stage 2 turns the non-distilled
`sulphur_dev_fp8mixed` into a paper-canonical two-stage distilled pipeline.
This is **NOT** Lightricks' rank-384 v1.1 distill LoRA — it's TenStrip's
re-ranked, cross-attention-safe variant (rank-72, Frobenius-90% truncation,
attention-bridge layers zeroed). Sulphur ships this file under
`distill_loras/` and the maintainer's shipped workflow loads it. The
`condsafe` part is what keeps the distill LoRA from interfering with i2v's
image-conditioning attention layers. Per-stage strengths follow Sulphur's
shipped workflow.

Without any distill LoRA (our `mode=quality` path), the full model wants
50 sampling steps via `LTXVScheduler(50, 2.72, 0.8, true, 0.0)` + sampler
`euler_ancestral` + CFG=3.6 — Sulphur's own non-distilled base workflow.

**Deviation from Sulphur's base workflow in quality mode:** Sulphur's shipped
`ltx23_t2v base.json` *also* applies the distill LoRA at strength 0.5/0.5
even in non-distilled mode (this is the audit recommendation we initially
adopted as "Knob 16a"). We **do not** apply the distill LoRA in our quality
mode, because Sulphur's base workflow stacks the distill LoRA on top of
`LTX-2.3-base + sulphur_final.safetensors` — a fundamentally different
recipe from our `Sulphur full FP8 model alone` recipe. The maintainer's
README explicitly forbids mixing the Sulphur full model with
`sulphur_final.safetensors`. Stacking the distill LoRA on the full Sulphur
FP8 model with 50-step LTXVScheduler + CFG=3.6 produces an audio waveform
that FFmpeg's h264 audio muxer rejects with
`avcodec_send_frame() returned 22` (EINVAL) at the SaveVideo node — we
empirically observed this failure in Tests B' and C. Quality mode therefore
runs the full Sulphur model standalone with the audit-recommended 50-step
schedule. This is "the working maximum-quality config given the recipe
constraint", not "the audit-canonical config."

Future investigation could bisect why exactly `distill LoRA + Sulphur full
model + 50 LTXVScheduler + CFG=3.6` breaks the audio path (most likely the
audio VAE's outputs go out of `[-1, +1]` due to compounding precision
errors when the distill LoRA's distilled-flow assumptions break against the
full 50-step trajectory). For now, the validated quality recipe is what
ships.

### Multi-stage pipeline: half-resolution base, x2 spatial upsample, refine

LTX-2.3 is trained for a max single-pass latent of ~20 temporal tokens
(~5 s at 24 fps). Higher resolutions and longer durations use Lightricks'
multi-stage pipeline (`arXiv:2601.03233` §4.2), and this is also the
pattern Sulphur's shipped `ltx23_t2v distilled.json` follows: the stage-1
`EmptyLTXVLatentVideo` is constructed at **half** the user-facing final
resolution (e.g., 960×544 for a 1080p final), the stage-1 sampler runs at
that base resolution, the resulting video latent is upscaled x2 spatially
via `LTXVLatentUpsampler` driven by the dedicated
`ltx-2.3-spatial-upscaler-x2-1.1` model, and a stage-2 sampler refines the
upsampled latent. The VAE decode then produces the final pixel output at
the user-facing resolution.

The half-resolution base is mandatory: feeding stage 1 at the full
user-facing resolution silently produces a 2x output (4K instead of 1080p)
and exhausts VRAM on 32 GiB cards. solphur2 enforces the half-resolution
base by computing `stage1_width = (width // 2 // 32) * 32` and the same
for height, so the API user request specifies the *final* dimensions and
the workflow takes care of the rest.

Frame count flows through unchanged across both stages — the spatial
upsampler does not touch the temporal axis.

### SageAttention 2.2.0 patched per-model, not via the CLI flag

ComfyUI's `--use-sage-attention` global flag calls `import sageattention`
at startup and exits the process with -1 if the import fails (see
`comfy/ldm/modules/attention.py:28-33`). Inside a container, that's a
hard-blocking constraint. We instead use `PathchSageAttentionKJ` (from
`kijai/ComfyUI-KJNodes`) which patches the per-model
`transformer_options["optimized_attention_override"]` at workflow time —
finer granularity, no startup-blocking import side-effect, and the
per-model override takes precedence anyway (`attention.py:127-143`).

### Prompt enhancer: highest-quality form the maintainer publishes

The Sulphur prompt enhancer is a fine-tuned **Qwen 3.5-9B hybrid** (Gated
DeltaNet + softmax-attention with a 3:1 ratio — full attention every 4th
transformer block, the rest linear-attention / SSM-style). The maintainer
publishes the fine-tune as two companion files:

- `sulphur_prompt_enhancer_model-q8_0.gguf` (8.87 GiB) — the text-side
  weights, quantized **Q8_0**. Q8_0 is the **only** precision the maintainer
  shipped for the fine-tuned text side. Q8_0 is near-lossless (~1 %
  perplexity gap vs. BF16); using it is not a tradeoff we chose — it's the
  maintainer's distribution decision.
- `mmproj-BF16.gguf` (879 MiB) — the Unsloth-packaged `qwen3vl_merger`
  vision projector at **BF16**, i.e., the exact dtype the projector was
  trained in. **No precision loss whatsoever** on the vision side.

So for the prompt enhancer specifically, we run **the highest-quality form
of each side the maintainer offers**. We do NOT downgrade either — neither
to a lower-bit text GGUF (Q4 / Q5 / Q6) nor to an FP8-cast mmproj. The full
Q8_0 + BF16 pair costs ~10 GiB of system RAM but **zero VRAM** (it runs on
CPU via llama.cpp `GGML_CUDA=OFF`), which is the deliberate architectural
choice that lets the video transformer keep the full 32607 MiB to itself.

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
companion crash on Blackwell. The KJNodes repo ships **no git tags**; we pin
the full commit SHA `1252598b41be959776f1208428b05b323b3fe17a` (the commit
whose message says "version 1.4.0", 2026-05-05), which is downstream of
`0e173bbfa9de` and contains the fix.

## Project layout

```
solphur2_setup/
├── README.md                    this file
├── LICENSE                      Unlicense (public domain); upstream models keep their own licenses
├── versions.env                 single source of truth for every pin
├── docker-compose.yml           three services, 127.0.0.1 only
├── api/
│   └── server.py                single-file FastAPI gateway + workflow builder
├── docker/
│   ├── Dockerfile.comfyui       ComfyUI v0.21.0 + LTX-2.3 + SageAttention for sm_120
│   ├── Dockerfile.enhancer      llama.cpp CPU build for the Sulphur prompt enhancer
│   └── Dockerfile.api           thin FastAPI runtime
├── scripts/
│   ├── up.sh                    bring-up orchestrator
│   ├── build.sh                 pure `docker compose build` wrapper
│   ├── download_models.sh       SHA-256-pinned, idempotent model fetcher
│   ├── test.sh                  automated smoke + headline end-to-end tests
│   ├── bench.py                 binary-search VRAM/time envelope sweep
│   ├── down.sh                  graceful teardown
│   └── clean.sh                 full cleanup (containers, images, cache, optionally models)
├── models/                      ← created at first run; gitignored
├── outputs/                     ← created at first run; gitignored
└── test_artifacts/              ← created by scripts/test.sh; gitignored
```

## Tuning knobs

The API exposes the user-facing ones (resolution, duration, fps, seed, mode).
Server-side defaults baked into `api/server.py`, every value cross-verified
against both Sulphur's shipped `workflows/ltx23_t2v distilled.json` and
Lightricks' canonical LTX-2.3 two-stage distilled example:

| Knob | `fast` mode | `quality` mode |
|---|---|---|
| **Default?** | opt-in (`mode="fast"`) — faster iterations | **YES** (the default) — Sulphur's non-distilled headline path |
| Distill LoRA applied? | yes, per-stage strengths 0.7 stage 1 + 0.5 stage 2 | no |
| Stage-1 sampler | `euler_ancestral_cfg_pp` | `euler_ancestral` |
| Stage-1 sigma source | `ManualSigmas` (Lightricks' canonical `DISTILLED_SIGMA_VALUES`) | `LTXVScheduler(50, 2.72, 0.8, true, 0.0)` with `latent` wired |
| Stage-1 sigmas (fast literal) | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` (9 values = 8 active steps) | — (scheduler-generated) |
| Stage-1 CFG | 1.0 (mandatory for distilled flow) | 3.6 (Sulphur base) |
| Stage-2 sampler | `euler_cfg_pp` (Lightricks two-stage canonical; Sulphur ships `lcm` but we reject — `lcm` is noise-prediction paradigm, LTX-2 is flow-matching) | `euler_cfg_pp` |
| Stage-2 sigmas | `0.85, 0.7250, 0.4219, 0.0` (4 values = 3 active steps; both Sulphur AND Lightricks agree) | same |
| Stage-2 CFG | 1.0 | 1.0 |
| Noise seed strategy | stage 1 = caller seed (fixed); stage 2 = caller seed + 1 (fixed). Both deterministic. | same |
| VAE decoder | `LTXVTiledVAEDecode` 2×2 tiles, overlap 6, `last_frame_fix=False` (Lightricks two-stage canonical) | same |
| Spatial upscaler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (newer; Sulphur ships v1.0, both work) | same |
| Text encoder | `gemma_3_12B_it_fp8_scaled.safetensors` (NOT NVFP4 — avoids ComfyUI #11864 Blackwell loader bug; Sulphur ships NVFP4 but it's broken on this GPU) | same |
| SageAttention | per-model patch (`PathchSageAttentionKJ(sage_attention="sageattn_qk_int8_pv_fp16_cuda", allow_compile=False)` — explicit FP16 PV / INT8 QK / FP32 accumulator path, more accurate than `"auto"` which routes to FP8 PV on sm_120) | same |

## License

This project's orchestration code is released under
[The Unlicense](https://unlicense.org/) — public domain, do anything you
want with it. See `LICENSE` for the canonical text.

The bundled and downloaded models retain their own upstream licenses,
which are NOT public domain:

- `SulphurAI/Sulphur-2-base` — see the SulphurAI HF repo. The author
  describes it as *"uncensored"*; operate accordingly.
- `Lightricks/LTX-2.3` (base architecture + spatial upscaler) — see the
  Lightricks HF repo.
- `Comfy-Org/ltx-2` Gemma3 text encoder — derived from Google's Gemma 3
  weights; subject to Google's Gemma license.
- TenStrip's `fro90_ceil72_condsafe` distill LoRA — see TenStrip's HF
  repo for its license.
- Unsloth-quantized Qwen 3.5-9B + qwen3vl_merger mmproj (the prompt
  enhancer) — Apache-2.0 upstream; see Qwen and Unsloth model cards.

Public-domain license on this orchestration code does NOT grant any
rights to the model weights it operates on.
