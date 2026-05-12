#!/usr/bin/env python3
"""
scripts/bench.py — Sweep the solphur2 quality/VRAM envelope.

The "max-quality knob" question on a 32 GiB RTX 5090 is: where exactly do
we cross the VRAM line? This script walks an opinionated 9-step linear
list of (resolution, duration, fps, mode) tuples and records (peak VRAM,
wall-clock, output filesize, exit status) per attempt.

Usage:
    python3 scripts/bench.py                          # run the default sweep
    python3 scripts/bench.py --max-runs 5             # cap to N runs
    python3 scripts/bench.py --prompt "custom seed"   # override prompt
    python3 scripts/bench.py --output-dir ./bench_run # results location

Output: JSON Lines at <output-dir>/bench-YYYYMMDD-HHMMSS.jsonl, plus a
summary table at the end.

The sweep is a STATIC ordered list, not an adaptive search. Run #1 is
the validated HEADLINE (1280×704 × 10 s × quality) so `--max-runs 1`
gives a successful highest-fidelity measurement; runs #2-#6 walk the
fast-mode envelope up to the LTX-2.3 ceiling; run #7 is the known-broken
1080p × 20 s × quality (reproduces avcodec EINVAL — kept as a regression
detector); runs #8-#9 push past the documented ceiling. A run that
exceeds 32500 MiB peak VRAM (within 100 MiB of the card's 32607 MiB
ceiling) or returns HTTP 5xx is recorded as failure but does NOT alter
subsequent runs — the list is static.

Dependencies: the standard library only (urllib, json, subprocess). Run from
the project root with the stack already up (scripts/up.sh).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import urllib.error
import urllib.request

API = "http://127.0.0.1:8000"
HARD_VRAM_CEILING_MIB = 32607          # RTX 5090 spec.
SAFETY_HEADROOM_MIB = 100              # treat ≥32500 MiB as too close to OOM.
NVIDIA_SMI = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


@dataclasses.dataclass
class Run:
    width: int
    height: int
    duration_seconds: float
    fps: int
    mode: str
    seed: int
    peak_vram_mib: int | None = None
    wall_clock_seconds: float | None = None
    response_status: int | None = None
    output_bytes: int | None = None
    error: str | None = None

    @property
    def fits(self) -> bool:
        if self.response_status != 200:
            return False
        if self.peak_vram_mib is None:
            return False
        return self.peak_vram_mib + SAFETY_HEADROOM_MIB <= HARD_VRAM_CEILING_MIB


def peak_vram_sampler(state: dict) -> None:
    """Background-thread loop that records the running max VRAM usage."""
    while not state["stop"]:
        try:
            out = subprocess.check_output(
                [NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2,
            )
            used = int(out.strip().splitlines()[0])
            if used > state["peak"]:
                state["peak"] = used
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)


def execute_one(prompt: str, run: Run, output_dir: Path) -> Run:
    """Issue one /generate request and record peak VRAM during it."""
    import threading

    state = {"stop": False, "peak": 0}
    t = threading.Thread(target=peak_vram_sampler, args=(state,), daemon=True)
    t.start()

    body = json.dumps(
        {
            "prompt": prompt,
            "seed": run.seed,
            "duration_seconds": run.duration_seconds,
            "width": run.width,
            "height": run.height,
            "fps": run.fps,
            "mode": run.mode,
            "enhance_prompt": False,  # bench prompts only; skip the enhancer.
            "negative_prompt": "low quality, blurry, artifacts",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    out_file = output_dir / f"bench_{run.width}x{run.height}_{int(run.duration_seconds)}s_{run.mode}_{run.seed}.mp4"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60 * 60) as resp:
            run.response_status = resp.status
            data = resp.read()
            out_file.write_bytes(data)
            run.output_bytes = len(data)
    except urllib.error.HTTPError as e:
        run.response_status = e.code
        run.error = e.read().decode("utf-8", errors="replace")[:500]
    except Exception as e:  # noqa: BLE001
        run.response_status = -1
        run.error = repr(e)[:500]
    finally:
        run.wall_clock_seconds = round(time.monotonic() - t0, 2)
        state["stop"] = True
        t.join(timeout=2)
        run.peak_vram_mib = state["peak"] or None

    return run


def sweep_config(max_runs: int) -> list[Run]:
    """The opinionated sweep order — VALIDATED HEADLINE FIRST.

    Run #1 is the highest-fidelity config validated to work end-to-end
    (1280×704 × 10 s × quality), so `python3 scripts/bench.py --max-runs 1`
    produces a successful headline measurement. Subsequent runs walk the
    fast-mode envelope up to the LTX-2.3 ceiling. The 1920×1088 × 20 s ×
    quality run is deliberately placed late in the sweep because it
    REPRODUCES a known audio-mux EINVAL failure — useful for capturing
    the bug, but not as the lead measurement.
    """
    seed = 4242
    runs = [
        # 1. HEADLINE — Sulphur's tested envelope × quality (upstream-canonical refine).
        Run(1280, 704, 10.0, 24, "quality", seed),
        # 2. Same envelope × fast — speed-optimised distilled recipe.
        Run(1280, 704, 10.0, 24, "fast", seed),
        # 3. 720p × 20 s × fast — full duration at the safe-megapixel resolution.
        Run(1280, 704, 20.0, 24, "fast", seed),
        # 4. 720p × 5 s × fast — cheap regression-guard smoke.
        Run(1280, 704, 5.0, 24, "fast", seed),
        # 5. 1080p × 10 s × fast — full resolution within typical duration.
        Run(1920, 1088, 10.0, 24, "fast", seed),
        # 6. 1080p × 20 s × fast — LTX-2.3 ceiling at the speed recipe (largest known-working).
        Run(1920, 1088, 20.0, 24, "fast", seed),
        # 7. 1080p × 20 s × quality — KNOWN-FAILS at SaveVideo audio mux (avcodec EINVAL).
        #    Kept in the sweep to detect if upstream ever fixes the audio-VAE
        #    amplitude drift at the longer / higher-res latent.
        Run(1920, 1088, 20.0, 24, "quality", seed),
        # 8. push past the documented ceiling: 1440p × 10 s.
        Run(1440, 800, 10.0, 24, "fast", seed),
        # 9. brave attempt: 4K × 10 s — expected to OOM.
        Run(3840, 2176, 10.0, 24, "fast", seed),
    ]
    return runs[: max_runs or len(runs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt",
        default=(
            "a cinematic close-up of a foggy cobblestone alley at dawn, "
            "warm amber lamplight reflecting on wet stones, slow tracking shot, "
            "shallow depth of field, 35mm film grain"
        ),
    )
    ap.add_argument("--output-dir", type=Path, default=Path("./bench_runs"))
    ap.add_argument("--max-runs", type=int, default=0, help="0 = full sweep")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = args.output_dir / f"bench-{stamp}.jsonl"

    runs = sweep_config(args.max_runs)
    print(f"will run {len(runs)} configs, writing to {jsonl_path}")
    print(f"hard VRAM ceiling: {HARD_VRAM_CEILING_MIB} MiB; safety margin: {SAFETY_HEADROOM_MIB} MiB")
    print()

    results: list[Run] = []
    with jsonl_path.open("w") as jf:
        for i, run in enumerate(runs, 1):
            print(
                f"[{i}/{len(runs)}] {run.width}x{run.height} × {run.duration_seconds}s @ "
                f"{run.fps}fps {run.mode}",
                flush=True,
            )
            execute_one(args.prompt, run, args.output_dir)
            results.append(run)
            jf.write(json.dumps(dataclasses.asdict(run)) + "\n")
            jf.flush()

            ok = "OK " if run.fits else "FAIL"
            print(
                f"    {ok} status={run.response_status} "
                f"peak_vram={run.peak_vram_mib} MiB "
                f"wall_clock={run.wall_clock_seconds}s "
                f"out_bytes={run.output_bytes}"
            )
            if run.error:
                print(f"    error: {run.error[:200]}")
            print()

            # Back off the sweep if VRAM exhaustion is imminent: skip larger configs.
            if run.peak_vram_mib and run.peak_vram_mib > HARD_VRAM_CEILING_MIB - 200:
                print(f"    VRAM within 200 MiB of ceiling; stopping sweep early.")
                break

    print()
    print("=" * 78)
    print(
        f"{'config':<32} {'mode':<8} {'status':>7} {'VRAM(MiB)':>10} {'time(s)':>10} {'bytes':>12}"
    )
    print("-" * 78)
    for r in results:
        cfg = f"{r.width}x{r.height} × {r.duration_seconds:>4.0f}s × {r.fps}fps"
        print(
            f"{cfg:<32} {r.mode:<8} "
            f"{r.response_status or '?':>7} "
            f"{r.peak_vram_mib or '?':>10} "
            f"{r.wall_clock_seconds or '?':>10} "
            f"{r.output_bytes or '?':>12}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
