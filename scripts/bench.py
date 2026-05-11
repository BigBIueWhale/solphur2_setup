#!/usr/bin/env python3
"""
scripts/bench.py — Binary-search the solphur2 quality/VRAM envelope.

The "max-quality knob" question on a 32 GiB RTX 5090 is: where exactly do
we cross the VRAM line? This script systematically probes (resolution,
duration, mode, step-count) tuples and records (peak VRAM, wall-clock,
output filesize, exit status) per attempt.

Usage:
    python3 scripts/bench.py                          # run the default sweep
    python3 scripts/bench.py --max-runs 5             # cap to N runs
    python3 scripts/bench.py --prompt "custom seed"   # override prompt
    python3 scripts/bench.py --output-dir ./bench_run # results location

Output: JSON Lines at <output-dir>/bench-YYYYMMDD-HHMMSS.jsonl, plus a
summary table at the end.

The sweep is structured as a binary search over the (resolution × duration)
plane in `fast` mode, then a follow-up confirmation in `quality` mode at the
largest known-good config. A run that returns HTTP 5xx or whose peak VRAM
exceeds 32500 MiB (within 100 MiB of the card's 32607 MiB ceiling) is treated
as failure and the search backs off.

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
    """The opinionated sweep order. Tries the cheapest valid config first
    so we can prove the stack works before stressing it, then walks toward
    the documented ceiling."""
    seed = 4242
    runs = [
        # 1. cheapest smoke: 5 s @ 720p, fast
        Run(1280, 704, 5.0, 24, "fast", seed),
        # 2. 10 s @ 720p
        Run(1280, 704, 10.0, 24, "fast", seed),
        # 3. 20 s @ 720p — bmgjet-equivalent at the lower bound
        Run(1280, 704, 20.0, 24, "fast", seed),
        # 4. 5 s @ 1080p — half-length test before the headline target
        Run(1920, 1088, 5.0, 24, "fast", seed),
        # 5. 10 s @ 1080p — Sulphur's shipped default duration
        Run(1920, 1088, 10.0, 24, "fast", seed),
        # 6. THE HEADLINE: 20 s @ 1080p, fast
        Run(1920, 1088, 20.0, 24, "fast", seed),
        # 7. 20 s @ 1080p, quality mode — same VRAM, ~30 min
        Run(1920, 1088, 20.0, 24, "quality", seed),
        # 8. push past the documented Fast ceiling: 1440p × 10 s
        Run(1440, 800, 10.0, 24, "fast", seed),
        # 9. brave attempt: 4K × 10 s, fast — expected to OOM via 3-stage
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
