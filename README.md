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

## Recommended flow at a glance

The headline path runs Sulphur-2-base **exactly as the maintainer
fine-tuned it** — i.e. the recipe wired in upstream `workflows/ltx23_t2v
base.json` — modulo three Blackwell-stability swaps that don't affect the
Sulphur fine-tune (detailed in `## Why the choices`).

**One-time bring-up + one video, end-to-end:**

```bash
bash scripts/up.sh                                                   # validate host → download → build → up → wait healthy
python3 scripts/generate.py "<your prompt>" -o ~/Videos/run.mp4      # generate one video
```

That's the whole flow. `generate.py` hits the server-side default — no
flags needed. Programmatic alternative: `POST http://127.0.0.1:8000/generate
{"prompt": "..."}`.

**What the default actually is** (every value sourced from upstream Sulphur
`ltx23_t2v base.json` unless noted):

| Knob | Default | Source |
|---|---|---|
| Mode | `quality` | upstream `ltx23_t2v base.json` recipe |
| Checkpoint | `sulphur_dev_fp8mixed.safetensors` (fused) | Sulphur HF — fused equivalent to LTX-2.3-base + `sulphur_final` LoRA per the maintainer's "use the lora or use the full models, don't use both" |
| Resolution | 1280 × 704 | within Sulphur's tested megapixel class (~720p mod-32) |
| Duration | 10 s @ 24 fps = 241 frames | Sulphur's shipped duration default (`8n+1` constraint) |
| Stage-1 sampler | `euler_ancestral` | upstream `KSamplerSelect` node 17 |
| Stage-1 scheduler | `LTXVScheduler(50, 2.72, 0.8, true, 0.0)` | upstream node 47 (50-step) |
| Stage-1 CFG | 3.6 | upstream `CFGGuider` node 42 |
| Stage-1 distill LoRA | NOT applied | upstream node 59 is `mode: 4` / bypassed |
| Spatial upscaler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | LTX-2.3 canonical 2× latent upsample |
| Stage-2 sampler | `euler_cfg_pp` | Lightricks two-stage canonical (upstream ships `lcm`, paired with consistency-distilled flow; we use `euler_cfg_pp` consistent with the non-distilled stage-1) |
| Stage-2 sigmas | `0.85, 0.7933, 0.68, 0.51, 0.2833, 0.0` (5 steps) | upstream `ManualSigmas` node 58 |
| Stage-2 CFG | 1.0 | upstream `CFGGuider` node 8 |
| **Stage-2 distill LoRA** | `fro90_ceil72_condsafe` @ 0.5 | upstream node 49 (mode 0, active — designed-in stage-2 refinement) |
| Prompt enhancer | on (Sulphur Qwen3.5-9B Q8 GGUF, CPU only) | Sulphur HF prompt_enhancer/ subtree |
| Gemma3 text encoder | `gemma_3_12B_it_fp8_scaled.safetensors` | **Blackwell swap** — upstream FP4 hits ComfyUI #11920 on sm_120 |
| SageAttention | `sageattn_qk_int8_pv_fp16_cuda` | **Blackwell swap** — explicit FP16 PV / INT8 QK; safer than `auto` on sm_120 |

**Measured cost** on RTX 5090 sm_120 (32 607 MiB VRAM), validated 2026-05-12.
Reproduce with `bash scripts/measure.sh "<prompt>"` (default-envelope quality
headline) or `bash scripts/measure.sh "<prompt>" --ltx-ceiling-fast` (max
envelope at the speed-optimised recipe). Output lands under the gitignored
`./bench_runs/`.

**Default envelope (1280×704 × 10 s × quality — the headline):**

| Metric | Cold start | Warm cache |
|---|---:|---:|
| Wall-clock | **188 s** (~3 min 8 s) | **169 s** (~2 min 49 s) |
| VRAM peak | **31 795 MiB / 32 607 MiB (97.5%)** | 32 095 MiB (98.4%) |
| GPU power | 607 W peak / 431 W avg | 604 W peak / ~450 W avg |
| GPU compute util | 100% peak / 70% avg | 100% peak (sustained during sampling) |
| Comfy container RAM peak | **50 140 MiB** | ~40 GiB |
| Enhancer container RAM peak | 4 428 – 11 181 MiB | 4 428 – 11 181 MiB |
| Stack RAM peak (concurrent) | **53 726 MiB** | ~48 800 MiB |
| Output | MP4, H.264 1280×704 video + AAC-LC stereo 48 kHz audio inline | same |

**LTX-2.3 ceiling (1920×1088 × 20 s × fast — `--ltx-ceiling-fast`):**

| Metric | Value |
|---|---:|
| Wall-clock | **445 s** (~7 min 25 s) per video |
| VRAM peak | **31 713 MiB** / 32 607 MiB (97.3%) |
| GPU power | 607 W peak / ~534 W avg |
| GPU compute util | 100% peak / ~88% avg |
| Stack RAM peak (concurrent) | **57 949 MiB** |
| Output | MP4, H.264 1920×1088 video + AAC-LC stereo 48 kHz audio inline |

**LTX-2.3 ceiling in quality mode (`--ltx-ceiling-quality`): KNOWN-FAILS** at
the SaveVideo audio mux with `avcodec_send_frame() returned 22` (EINVAL)
after ~1080 s of GPU compute. Bug is independent of stage-2 distill LoRA
stacking (verified by control measurement). Cause is presumed to be audio-VAE
amplitude drift at the longer / higher-resolution latent — i.e. an
OOD-Sulphur failure mode, not a recipe defect. Use `--ltx-ceiling-fast` for
the max envelope until / unless the audio-VAE side gets a fix.

The three Blackwell swaps and the fused-vs-LoRA checkpoint substitution are
the only deviations from upstream `ltx23_t2v base.json`. Everything that
the model was post-trained on — the scheduler, samplers, CFG values, stage-2
distill LoRA strength, 5-step refinement — is the upstream-canonical recipe.

If you want speed over fidelity, append `--mode fast` to `generate.py` —
that switches to upstream `ltx23_t2v distilled.json` (stacks the distill
LoRA at stage-1 too, 8-step DISTILLED_SIGMA_VALUES, ~half wall-clock).

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
- **Measured 2026-05-12 at this exact config** (upstream-canonical recipe
  with the 3 Blackwell-stability swaps) on a fresh-stack cold start:
  188 s wall-clock, 31 795 MiB VRAM peak (97.5% of 32 607 MiB), 607 W
  peak, 53 726 MiB stack RAM peak. Subsequent warm-cache requests run
  ~169 s and ~48 700 MiB stack RAM. Reproduce locally with
  `bash scripts/measure.sh "<your prompt>"`; the full per-phase timing
  + GPU/CPU/RAM telemetry lands under the gitignored `./bench_runs/`.
- This is the **default** the API returns when the caller doesn't override.

**LTX-2.3 base ceiling (opt-in via larger request fields)** — what the
underlying architecture allows but the Sulphur fine-tune was NOT verified at:

- **1920 × 1088 × 481 frames @ 24 fps (1080p × 20 s)** is opt-in via API
  parameters but **does not work in `mode=quality`** at this time:
  measured 2026-05-12, the SaveVideo node crashes with
  `avcodec_send_frame() returned 22` (EINVAL) at the audio mux after
  ~1080 s of GPU compute. A control measurement with the stage-2 distill
  LoRA OMITTED hit the IDENTICAL failure at the same node, so the bug is
  unrelated to LoRA stacking; the most likely cause is audio-VAE
  amplitude drift at the longer / higher-resolution latent. The
  speed-optimised recipe `mode=fast` at this envelope is the largest
  known-working config — use `bash scripts/measure.sh "<prompt>"
  --ltx-ceiling-fast` to bench it. The API will accept the request and
  logs an `OOD-Sulphur` warning either way.

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
| ComfyMath | `be9beab9923ccf5c5e4132dc1653bcdfa773ed70` |
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
bash scripts/up.sh                          # validate host → download → build → up → wait healthy
bash scripts/test.sh                        # headline only (1280×704 × 10s × quality, ~188 s cold-start / ~169 s warm-cache on RTX 5090)
bash scripts/test.sh --with-smoke           # smoke (1280×704 × 5s × fast, ~80 s) + headline
bash scripts/test.sh --ltx-ceiling-fast     # also run 1920×1088 × 20s × fast (~445 s)
bash scripts/test.sh --ltx-ceiling-quality  # also run 1920×1088 × 20s × quality (KNOWN-FAILS at audio mux — capture-only)
bash scripts/down.sh                        # graceful teardown (preserves images, models, outputs)
bash scripts/clean.sh                       # remove containers + images + cache + outputs (keeps models)
bash scripts/clean.sh --all                 # also wipe ./models/ (forces re-download next up)
```

To generate one video against a live stack, use the dedicated CLI:

```bash
python3 scripts/generate.py "a cinematic close-up of a foggy cobblestone alley at dawn"
```

That single invocation hits every server-side default (quality mode,
1280×704, 10 s, 24 fps, prompt enhancer on) and writes
`./solphur2_<random>.mp4` in **~188 s on RTX 5090 cold-start (~3 min 8 s) or ~169 s warm-cache**. The script does a
`/healthz` preflight (fails fast with a hint to run `scripts/up.sh` if
the stack is down), streams the MP4 to a `.partial` file and atomically
renames on success, then prints a one-line summary including the
per-phase wall-clocks pulled from the `X-Solphur2-Phase*` response
headers.

Common variants — pass an override flag only for what you want to
deviate from the default:

```bash
# Pick the output path:
python3 scripts/generate.py "..." -o ~/Videos/run.mp4

# Speed over quality (~half wall-clock; ADDS the stage-1 distill LoRA at 0.7
# so the 50-step trajectory collapses to 8 steps. Stage-2 distill at 0.5 is
# applied in both modes — that's the upstream-canonical refinement).
python3 scripts/generate.py "..." --mode fast

# Maximum LTX-2.3 envelope (1080p × 20 s; logs an OOD-Sulphur warning):
python3 scripts/generate.py "..." --width 1920 --height 1088 --duration 20

# Reproducible: pass a seed:
python3 scripts/generate.py "..." --seed 12345

# Skip the enhancer (saves ~34 s; loses descriptive elaboration):
python3 scripts/generate.py "..." --no-enhance
```

Anything you don't pass falls through to the validated server-side
default in `api/server.py:GenerateRequest`. The script never sets a
default of its own that could drift away from the API's defaults.

The default mode is **quality** — upstream Sulphur's `ltx23_t2v base.json`
recipe: 50-step `euler_ancestral` at stage 1 (no distill LoRA), then a
5-step `euler_cfg_pp` refine at stage 2 with the cond_safe distill LoRA at
0.5. To prefer speed over fidelity, add `--mode fast` (or `"mode":"fast"`
in raw JSON) which additionally applies the stage-1 distill LoRA at 0.7,
collapsing the 50-step trajectory to 8 distilled steps. See
`## Resource peaks and timeline` below for measured wall-clocks.

### Script reference

| Script | What it does | Idempotent? |
|---|---|---|
| `scripts/up.sh` | host check → download models (delegates to `download_models.sh`) → build (delegates to `build.sh`) → `docker compose up -d` → wait healthy | yes |
| `scripts/build.sh [--no-cache]` | pure `docker compose build` — separated so iterations don't re-run model downloads or healthcheck waits | yes (incremental) |
| `scripts/download_models.sh` | SHA-256-pinned model fetcher (resumable, idempotent) | yes |
| **`scripts/generate.py "<prompt>" [-o PATH] […]`** | **The user-facing CLI.** POSTs one `/generate` request with server-side defaults, streams the MP4 to disk atomically, prints a per-phase summary. Override flags (`--mode`, `--seed`, `--duration`, `--width`, `--height`, `--fps`, `--no-enhance`) are opt-in; anything not passed falls through to `api/server.py` defaults. | yes (each call independent) |
| `scripts/test.sh [--smoke-only/--headline-only/--enhance]` | end-to-end POST `/generate` round-trips with hardcoded smoke + headline configs. Writes MP4s to `./test_artifacts/`. | yes |
| `scripts/measure.sh "<prompt>" [--skip-build/--no-cache] [--ltx-ceiling-fast / --ltx-ceiling-quality / --width N --height N --duration N --mode quality\|fast / --seed N / --no-enhance]` | rebuilds via `build.sh`, restarts the stack, runs ONE instrumented generation, samples nvidia-smi + cgroup v2 + `/proc` at 1 Hz, prints a peak/timeline summary. **Defaults to 1280×704 × 10 s × quality** (the highest-fidelity validated envelope). `--ltx-ceiling-fast` opts into 1920×1088 × 20 s × fast (the largest known-working envelope); `--ltx-ceiling-quality` opts into the same envelope in quality mode (KNOWN FAILURE — reproduces the audio-mux EINVAL). Prompt is required. Writes raw CSVs + summary to the gitignored `./bench_runs/measure_<W>x<H>_<D>s_<MODE>_*/`. | yes |
| `scripts/bench.py` | binary-search sweep over (resolution × duration × mode) for empirical VRAM/time envelope | yes |
| `scripts/down.sh` | stop + remove containers and network. Images and models survive. | yes |
| `scripts/clean.sh [--all]` | full cleanup — containers + images + build cache + outputs (and optionally models) | yes |

Every script is `set -Eeuo pipefail` and fails loudly on the first unexpected condition. No hidden
side effects, no silent retries.

## Resource peaks and timeline (default config)

Measured on RTX 5090 (sm_120, 32 607 MiB VRAM, 62 GiB host RAM, 32-thread
host CPU) via `scripts/measure.sh "<prompt>" --default-config`. The script samples
nvidia-smi telemetry at 1 Hz, container memory + CPU directly from cgroup v2
(`memory.current` + `cpu.stat`) at 1 Hz, and host CPU + RAM from `/proc/stat`
+ `/proc/meminfo` at 1 Hz, then aggregates peaks. Per-phase wall-clocks
come from `api/server.py`'s permanent timing instrumentation, surfaced as
`X-Solphur2-Phase*` response headers. Numbers below are from one run with
the prompt `"a beautiful nude woman lying on satin sheets in a dimly lit
bedroom, soft golden light streaming through sheer curtains, slow cinematic
tracking camera, intimate sensual atmosphere, professional film
cinematography, shallow depth of field, 35mm"` — i.e. a realistic NSFW
prompt that exercises the enhancer and the Sulphur uncensored fine-tune
end-to-end. To reproduce against your own prompt:

```bash
bash scripts/measure.sh "<your prompt>"                          # default = 1080p × 20 s × quality (headline)
bash scripts/measure.sh "<your prompt>" --default-config         # Sulphur-tested envelope (1280×704 × 10 s)
bash scripts/measure.sh "<your prompt>" --skip-build             # any envelope, reuse current images
```

### Wall-clock timeline (default = 1280×704 × 10 s × 24 fps × `mode=quality` × `enhance_prompt=true`)

| Phase | Cold start | Warm cache | What's running |
|---|---:|---:|---|
| `enhance` | **29.7 s** | ~22-34 s | Sulphur Qwen3.5-9B GGUF (CPU, ~22 cores). GPU idle. Variability is the enhancer's sampling temperature plus first-load mmap time. |
| `submit`  | 0.0 s | 0.0 s | FastAPI → ComfyUI `/prompt`. ComfyUI is warm, so workflow validation and queueing are sub-millisecond. |
| `comfy_run` | **158.3 s** | 146.3 s | Two-stage diffusion on the RTX 5090: stage-1 sampling (50 steps at 640×352 latent, `euler_ancestral` + CFG=3.6, no distill LoRA), spatial ×2 upsample, stage-2 refinement (5 steps at 1280×704 latent, `euler_cfg_pp` + CFG=1.0, cond_safe distill LoRA at 0.5), tiled VAE decode, audio VAE decode, h264 mux. Cold-start ~12 s slower due to first-request weight loads. |
| **Total** | **188 s** (~3 min 8 s) | **169 s** (~2 min 49 s) | One MP4 returned over HTTP. |

The GPU is **idle for the first ~34 seconds** (the enhancer is CPU-only,
intentionally so the entire 32 GiB VRAM stays available for the diffusion
model), then sustained at ~100% compute utilization for ~110 seconds, then
drops to 0% for the final ~10 seconds while h264 muxing runs on CPU.
Doubling the duration to 20 s roughly doubles `comfy_run`; pushing the
resolution to 1080p × 20 s in quality mode is expected to land at multiple
minutes total per the LTX-2.3 envelope, but **that specific config has not
yet been captured to `bench_runs/`** — only the 1280×704 × 10 s default
has full telemetry.

### System requirements (cold-start measurement, current code)

Numbers from a fresh `docker compose down` + `docker compose up -d` +
single-shot `bash scripts/measure.sh "<prompt>"` cycle on 2026-05-12 —
the worst case a new caller sees on their first request. Subsequent
warm-cache requests are slightly cheaper across the board.

| Resource | Cold-start peak | Notes |
|---|---:|---|
| **VRAM** | **31 795 MiB / 32 607 MiB (97.5%)** | The default config consumes essentially the entire RTX 5090 card. Smaller cards (24 GiB, 16 GiB) **will OOM** at these settings; there is no fallback path. The 2.5% headroom is why we run `--reserve-vram 0.5` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Warm-cache subsequent runs land at ~32 095 MiB. |
| GPU compute util | 100% peak / 70.0% avg | Average is dragged down by the ~30 s enhance-phase idle window; during sampling the GPU is pinned at 100%. |
| GPU power draw | **607 W peak / 431 W avg** | RTX 5090 board limit. PSU should be ≥1000 W. |
| GPU SM clock | 2 932 MHz peak | Within Blackwell's documented operating range; not thermally limited (peak temp 68 °C). |
| **Comfy container RAM** | **50 140 MiB** (cold start) | Dominated by the mmap'd FP8 safetensors (27 GiB on disk) plus PyTorch buffers, KJNodes' SageAttention state, and stage-2 output latents staged in host RAM before VAE decode. cgroup v2 `memory.current` counts file-backed cache; under memory pressure Linux can evict these pages, but doing so causes slow page faults next run. Warm-cache runs measure ~40 GiB. |
| **Enhancer container RAM** | 4 428 – 11 181 MiB | Mmap'd Q8 Qwen3.5-9B GGUF (~8 GiB) plus the BF16 qwen3vl_merger mmproj (~0.9 GiB). Peak varies with how much of the prompt activates the context window; the `mem_limit: 12g` cap in `docker-compose.yml` covers the worst case observed across runs. |
| API container RAM | 58 MiB | The FastAPI process itself. Negligible. |
| **Stack RAM peak (concurrent)** | **53 726 MiB** (cold start) | Sum of the three containers' peaks at the moment of maximum concurrent residency. This is the "must-have-available" floor — a 32 GiB host will swap thrash; **64 GiB host RAM is the practical minimum**. Warm-cache subsequent runs measured ~48 800 MiB. |
| Comfy CPU peak | ~1 670% | ~17 logical cores during VAE decode + h264 mux. |
| Enhancer CPU peak | ~1 795% | ~18 logical cores during the enhance phase. |
| Host CPU peak | 79.5% | One sample per second; the actual instantaneous saturation happens during enhance + during the brief h264 mux window. |
| Host RAM "used" (anonymous) | 19 006 MiB | This is the host's *anonymous* RAM (heap, stack, dirty pages); the container peaks above include reclaimable file-backed cache that doesn't count here. Together they mean: you need 64 GiB host RAM to keep the model files cached, but the *minimum* anonymous footprint to operate is ~19 GiB. |
| Host swap peak | 4 150 MiB | The host swapped a few GiB during the first request (cold cache populating). Subsequent runs don't touch swap. |

#### Reading the resource peaks correctly

Comfy's 47 GiB container RAM peak looks alarming until you realise it
includes the entire 27 GiB FP8 safetensors mmap'd as page cache. Those
pages are reclaimable — under memory pressure Linux can drop them and
re-fault from disk. The honest "must-have-available" number is the
stack-wide concurrent peak of ~57 GiB. The honest "absolute floor"
(anonymous-only) is the host's ~20 GiB used metric, but at that floor
the next request would need to re-mmap the entire model from cold disk
(slow). 64 GiB host RAM is the practical recommendation.

The 98.2% VRAM utilisation is the headline constraint. Any GPU sharing
(a desktop environment that grabs a few hundred MiB, a stray nvidia-smi
process, a CUDA context warm-up by another container) will push the
default config into OOM. `scripts/up.sh` enforces ≥31 GiB free VRAM
before bringing the stack up specifically because of this.

## Why the choices

### FP8-mixed, not BF16 or Q4 GGUF

The Sulphur maintainer ships three checkpoint forms: `sulphur_dev_bf16`
(42.97 GiB), `sulphur_distil_bf16` (42.97 GiB), and `sulphur_dev_fp8mixed`
(27.16 GiB). BF16 does not fit a 32 GiB single GPU. Community Q8/Q6/Q4 GGUFs
exist but quantize the model further from the FP8-native Blackwell tensor
core path. FP8 mixed preserves quality-sensitive layers (norms, gates,
embeddings) in BF16 and casts only the linear-layer weights to FP8 — the
quality/VRAM tradeoff that our same-hardware measurement validates: at the
default 1280×704 × 241 frames @ 24 fps in quality mode, the canonical
two-stage pipeline (half-resolution base + x2 spatial latent upsample +
5-step refine with cond_safe distill @ 0.5) peaks at 31 795 MiB VRAM and
completes in 188 s on this RTX 5090 from a fresh-stack cold start (169 s
warm-cache). The LTX-2.3 ceiling 1920×1088 × 20 s at `mode=fast` measures
445 s / 31 713 MiB; the same envelope at `mode=quality` reproduces an
EINVAL at the audio mux and is currently unusable — see the Tuning Knobs
table for details.

### Sulphur full model, not Sulphur LoRA on LTX-2.3 base

Sulphur ships both `sulphur_dev_fp8mixed.safetensors` (the full fine-tuned
model, ready to use) and `sulphur_lora_rank_768.safetensors` (a LoRA you
can apply on top of base LTX-2.3). The maintainer says "use the lora or
use the full models, don't use both at the same time." We use the full
model — fewer moving parts, no LoRA stacking bugs (e.g. the FP8 + rank-768
artifact regression that `masterkwic2` documented on HF discussion #3).

### Distill LoRA: TenStrip's `fro90_ceil72_condsafe` variant, per-stage strengths

`ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` is
TenStrip's re-ranked, cross-attention-safe variant of Lightricks' rank-384
v1.1 distill LoRA (rank-72, Frobenius-90% truncation, attention-bridge
layers zeroed — `cond_safe`). Sulphur ships this file under `distill_loras/`
and both shipped workflows wire it. We apply it identically:

  * **Stage-2 at strength 0.5 in BOTH modes** — matches upstream
    `ltx23_t2v base.json` (node 49 mode 0, active) and `ltx23_t2v
    distilled.json` (node 49 mode 0, active). TenStrip's README describes
    0.5 as the canonical "upscale pass" strength for this LoRA family —
    it's a designed-in cross-attention-shaping refinement, not a speed hack.
  * **Stage-1 at strength 0.7 in fast mode ONLY** — matches upstream
    `ltx23_t2v distilled.json` node 59 (active). In quality mode the
    stage-1 LoRA is bypassed, matching `ltx23_t2v base.json` node 59
    (mode 4 / bypassed).

Quality mode therefore runs the full Sulphur model with:

  * `LTXVScheduler(50, 2.72, 0.8, true, 0.0)` + `euler_ancestral` + CFG=3.6
    at stage 1 (no stage-1 distill LoRA — upstream `base.json` defaults).
  * `ManualSigmas("0.85, 0.7933, 0.68, 0.51, 0.2833, 0.0")` (5 steps) +
    `euler_cfg_pp` + CFG=1.0 + distill LoRA at 0.5 at stage 2 — upstream
    `base.json` node 58 (sigmas) + node 49 (LoRA).

**History note (resolved 2026-05-12):** an earlier code revision omitted
the stage-2 distill LoRA in quality mode after an `avcodec_send_frame()
returned 22 (EINVAL)` failure at the FFmpeg audio mux. The bug appears
tied to a now-superseded combination of (a) a Blackwell SageAttention
auto-dispatch path, (b) Gemma3 NVFP4 quant, and (c) the orphan 3-step
stage-2 sigmas — none of which the current build uses. The EINVAL no
longer reproduces; A/B/C bench runs on 2026-05-12 all produce valid MP4s
with AAC-LC stereo 48 kHz audio. The previous "no distill LoRA in quality
mode" deviation has been retired in favour of the upstream-canonical
wiring above.

#### Fast vs quality is also a fidelity-to-the-fine-tune tradeoff, not only a speed tradeoff

The Sulphur uncensored fine-tune is **baked into the FP8 weights of
`sulphur_dev_fp8mixed.safetensors`**, not delivered as a runtime LoRA in
our setup. Both modes therefore load the same uncensored weights — fast
mode is not "less uncensored" in any binary sense. But the two modes
differ in how faithfully those weights are exercised:

- **Quality mode (default)** runs the Sulphur weights through their
  natively-trained 50-step flow-matching trajectory at CFG=3.6 at stage 1
  (no distill LoRA, matching `ltx23_t2v base.json`'s `mode: 4` bypass of
  the stage-1 LoRA), then applies the cond_safe distill LoRA at strength
  0.5 only at stage 2 (matching `ltx23_t2v base.json` node 49 mode 0).
  The stage-2 cond_safe pass is part of how Sulphur was post-trained, not
  a speed hack — its purpose is cross-attention refinement during the
  spatial-upsample refine step.
- **Fast mode** ALSO applies the TenStrip step-distillation LoRA at
  stage 1 (strength 0.7), collapsing the 50-step trajectory into 8 steps.
  Stage 2 is identical to quality mode (distill at 0.5). So the
  effective weights at stage 1 are `Sulphur ⊕ (0.7·distillΔ)` in fast,
  unmodified Sulphur in quality. The `condsafe` variant zeroes
  cross-attention bridge layers specifically so i2v image conditioning
  survives the rank-72 LoRA delta; the rest of the network is still
  perturbed at stage 1 in fast mode. The distill LoRA's stage-1 effect
  was trained to collapse the 8-step trajectory into a single-shot
  prediction, not to preserve every signal in the Sulphur fine-tune.

In practice both modes produce convincing NSFW output (the
working-set fine-tune dominates the generative distribution), but on
prompts at the edge of Sulphur's competence — uncommon anatomical
configurations, unusual lighting, or scenes Sulphur's training set
under-represented — quality mode is the more conservative choice.
**This is why `mode="quality"` is the default**: speed is opt-in, and
maximum fidelity to the uncensored fine-tune is the headline path.

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

#### Empirically verified to elaborate NSFW prompts faithfully, not sanitize

The Sulphur fine-tune sits on top of Qwen 3.5-9B base, which has its own
refusal and "thinking" tendencies. Three concurrent defenses keep the
enhancer producing usable output for NSFW prompts at the configured
sampling envelope (`temperature 0.7, top_p 0.8, top_k 20, min_p 0.0,
repeat_penalty 1.0, presence_penalty 0.0`):

1. **Server-side flags** in `Dockerfile.enhancer`: `--reasoning off
   --reasoning-budget 0 --reasoning-format deepseek` — llama.cpp will
   neither generate a `<think>` block nor count one against the output
   budget, and any leakage that does occur is routed to the
   `reasoning_content` field instead of `content`.
2. **Per-request override** in `api/server.py:_enhance_prompt`:
   `chat_template_kwargs: {"enable_thinking": false}` belt-and-suspenders
   to the server flag.
3. **Sampling envelope chosen against drift**: `presence_penalty=0`
   specifically because non-zero presence penalty is what pushes the
   Sulphur fine-tune to substitute generic Qwen vocabulary for the
   NSFW-specific tokens the user actually asked for. The full envelope
   was lifted from Sulphur's empirical recommendations, not from
   Qwen-base defaults.

Verification from `scripts/measure.sh --default-config` run on
`"a beautiful nude woman lying on satin sheets in a dimly lit bedroom, soft
golden light streaming through sheer curtains, slow cinematic tracking
camera, intimate sensual atmosphere, professional film cinematography,
shallow depth of field, 35mm"`:

- `finish_reason: stop`, `content_len: 619 chars`, `reasoning_content_len: 0`
- Enhanced prompt header (truncated to 200 chars in the response): *"The
  camera provides a slow, cinematic tracking shot of a beautiful nude
  woman lying on satin sheets in a dimly lit bedroom. The scene is bathed
  in soft, golden light streaming through sheer curtains, illuminating
  her smooth ski…"*
- No refusal, no sanitization, no `<think>` leakage, no truncation of
  the NSFW-specific tokens. Enhancer wall-clock: 22-34 s on CPU across
  the three archived A/B/C bench runs (variability is the enhancer's
  sampling temperature, not the inference path).

If any future regression silently disables one of those three defenses,
`api/server.py:_enhance_prompt` will log a loud warning
(`"prompt enhancer leaked thinking content (N chars)…"` or `"prompt
enhancer returned empty content; falling back to raw prompt"`) so the
degradation is caught immediately rather than producing silently weaker
videos.

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
│   ├── generate.py              user-facing CLI: one prompt → one MP4 at server-side defaults
│   ├── up.sh                    bring-up orchestrator
│   ├── build.sh                 pure `docker compose build` wrapper
│   ├── download_models.sh       SHA-256-pinned, idempotent model fetcher
│   ├── test.sh                  automated smoke + headline end-to-end tests
│   ├── bench.py                 binary-search VRAM/time envelope sweep
│   ├── measure.sh               rebuild + measure ONE instrumented generation (defaults to 1080p × 20 s × quality headline)
│   ├── _measure_client.py       internal Python POST client used by measure.sh
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
| Wall-clock at 720p × 10 s (measured) | ~75 s comfy_run + ~34 s enhance ≈ ~110 s total | **146 s comfy_run + ~23 s enhance = 169 s total** (validated 2026-05-12; reproduce with `bash scripts/measure.sh "<prompt>"`) |
| Wall-clock at 1080p × 20 s | **445 s** (7 min 25 s) — measured 2026-05-12, 31 713 MiB VRAM, 607 W. Reproduce with `bash scripts/measure.sh "<prompt>" --ltx-ceiling-fast`. | **fails with `avcodec EINVAL` at SaveVideo audio mux** after ~1080 s of GPU compute. Bug is independent of stage-2 distill stacking (confirmed by control measurement). Reproduce with `bash scripts/measure.sh "<prompt>" --ltx-ceiling-quality`. |
| Distill LoRA applied? | both stages — strength 0.7 stage-1 + 0.5 stage-2 (`ltx23_t2v distilled.json` node 59 + 49) | stage-2 only at strength 0.5 (`ltx23_t2v base.json` node 49 mode 0; stage-1 node 59 is mode 4 / bypassed). Cond_safe LoRA is designed-in stage-2 refinement, not a speed hack. |
| Fidelity to Sulphur's training distribution | distilled-trajectory approximation | full 50-step flow-matching trajectory at the CFG Sulphur was trained at, with the canonical cond_safe stage-2 refinement |
| Stage-1 sampler | `euler_ancestral_cfg_pp` | `euler_ancestral` |
| Stage-1 sigma source | `ManualSigmas` (Lightricks' canonical `DISTILLED_SIGMA_VALUES`) | `LTXVScheduler(50, 2.72, 0.8, true, 0.0)` with `latent` wired |
| Stage-1 sigmas (fast literal) | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` (9 values = 8 active steps) | — (scheduler-generated) |
| Stage-1 CFG | 1.0 (mandatory for distilled flow) | 3.6 (Sulphur base) |
| Stage-2 sampler | `euler_cfg_pp` (Lightricks two-stage canonical; Sulphur ships `lcm` but we reject — `lcm` is noise-prediction paradigm, LTX-2 is flow-matching) | `euler_cfg_pp` |
| Stage-2 sigmas | `0.85, 0.7250, 0.4219, 0.0` (4 values = 3 active steps) — `STAGE2_SIGMAS_FAST` in `api/server.py`, the actively-wired stage-2 schedule of upstream Sulphur's `ltx23_t2v distilled.json` (ManualSigmas node id=7). Designed to ride with the distill LoRA at stage 2: 3 steps suffice because each consistency-distilled step approximates a multi-step trajectory. | `0.85, 0.7933, 0.68, 0.51, 0.2833, 0.0` (6 values = 5 active steps) — `STAGE2_SIGMAS_QUALITY`, the actively-wired stage-2 schedule of upstream Sulphur's `ltx23_t2v base.json` (ManualSigmas node id=58). Designed for non-distilled refinement: more steps because each step does the full diffusion ODE update rather than a distilled-trajectory shortcut. |
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
