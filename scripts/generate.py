#!/usr/bin/env python3
"""
scripts/generate.py — the simplest user-facing CLI for solphur2.

Produces ONE MP4 from your prompt and exits. Every other parameter (mode,
resolution, duration, fps, enhancer, seed) defaults to the **server-side**
default in api/server.py:GenerateRequest — i.e. the validated headline
config (quality mode, 1280×704, 10 s, 24 fps, enhancer on). Pass an
override flag only if you specifically want to deviate from that.

Examples
--------

    # The simplest invocation — server-side defaults for everything:
    python3 scripts/generate.py "a luxury car driving along a coastal road at golden hour"

    # NSFW (Sulphur is uncensored; the model handles it, just describe it):
    python3 scripts/generate.py "a beautiful nude woman lying on satin sheets, soft golden light, slow tracking camera"

    # Choose the output path:
    python3 scripts/generate.py "..." -o ~/Videos/run.mp4

    # Speed over quality (~7 min vs ~14 min at 1080p × 20 s):
    python3 scripts/generate.py "..." --mode fast

    # Maximum envelope (LTX-2.3 ceiling, logs an OOD-Sulphur warning):
    python3 scripts/generate.py "..." --width 1920 --height 1088 --duration 20

    # Reproducible run:
    python3 scripts/generate.py "..." --seed 12345

What this script does
---------------------

  1. Confirms the API is reachable + healthy (GET /healthz). Fails fast
     with a hint to run scripts/up.sh if not.
  2. POSTs /generate with ONLY the fields you explicitly set. Anything
     you don't pass is omitted from the body so the server-side default
     stays authoritative — no risk of accidentally locking in a default
     here that drifts away from api/server.py.
  3. Streams the MP4 response to <output>.partial in 1 MiB chunks, then
     atomically renames to <output>. A killed command leaves at most a
     .partial file alongside, never a half-written final path.
  4. Prints one human-readable summary line plus a per-phase wall-clock
     breakdown pulled from the X-Solphur2-Phase* response headers.

Exit codes
----------

    0   HTTP 200 + MP4 written
    1   API returned non-200 (workflow error, OOM, etc.)
    2   transport / health / preflight failure (stack down, bad path, …)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


API_DEFAULT_URL = "http://127.0.0.1:8000"


def _check_health(api_url: str) -> None:
    """Block early if the stack isn't ready — better than a long hang.

    Three failure modes we explicitly distinguish:

      1. API container itself unreachable (connection refused / DNS / timeout
         on :8000). Most likely: the whole stack is down. We tell the user
         to run scripts/up.sh.

      2. API up but a dependency (comfyui or enhancer) is down. The API's
         /healthz returns HTTP 503 with a JSON body
         `{"comfyui": bool, "enhancer": bool, "ok": false}`. urllib raises
         HTTPError; we recover the body and tell the user WHICH specific
         service is dead so they can `docker compose logs <name>` directly.

      3. API responds 200 but the body says `"ok": false`. Defensive — the
         server-side logic in api/server.py:healthz should always make
         status_code consistent with `ok`, but we don't want to silently
         start a 14-minute generation if the contract slips.
    """
    healthz_url = f"{api_url.rstrip('/')}/healthz"
    payload: dict | None = None
    try:
        with urllib.request.urlopen(healthz_url, timeout=5.0) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # API reachable, /healthz returned non-2xx — body is still JSON.
        try:
            payload = json.loads(exc.read())
        except (json.JSONDecodeError, OSError):
            payload = None
        broken = []
        if isinstance(payload, dict):
            broken = [name for name in ("comfyui", "enhancer") if payload.get(name) is False]
        if broken:
            sys.stderr.write(
                f"solphur2 API at {api_url} responded HTTP {exc.code} on /healthz "
                f"(body: {payload}).\n"
                f"Unhealthy upstream container(s): {', '.join(broken)}.\n"
                f"Inspect:\n"
                + "".join(
                    f"    docker compose logs solphur2-{name}\n" for name in broken
                )
                + f"Or restart the whole stack:\n"
                f"    bash scripts/down.sh && bash scripts/up.sh --skip-models\n"
            )
        else:
            sys.stderr.write(
                f"solphur2 API at {api_url} returned HTTP {exc.code} from /healthz "
                f"with non-parseable body. Inspect:\n"
                f"    docker compose ps\n"
                f"    docker compose logs solphur2-api\n"
            )
        sys.exit(2)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        # Connection refused / DNS / timeout / garbage body — most likely
        # the API container itself is not running, which usually means the
        # whole stack is down.
        sys.stderr.write(
            f"solphur2 API at {api_url} not reachable ({exc!r}).\n"
            f"Most likely the API container is down. Inspect:\n"
            f"    docker compose ps\n"
            f"If no `solphur2-*` containers are listed, bring the stack up:\n"
            f"    bash scripts/up.sh\n"
        )
        sys.exit(2)

    if not isinstance(payload, dict) or not payload.get("ok"):
        broken = []
        if isinstance(payload, dict):
            broken = [name for name in ("comfyui", "enhancer") if payload.get(name) is False]
        sys.stderr.write(
            f"solphur2 API at {api_url} returned 200 from /healthz but body says "
            f"unhealthy: {payload}\n"
            f"Unhealthy upstream container(s): {', '.join(broken) or 'unknown'}.\n"
            f"This usually means status_code/body in api/server.py disagree — "
            f"file a bug. To recover meanwhile:\n"
            f"    bash scripts/down.sh && bash scripts/up.sh --skip-models\n"
        )
        sys.exit(2)


def _header(headers: dict, key: str) -> str:
    """Look up a header case-insensitively. urllib lowercases on HTTP/1.1."""
    needle = key.lower()
    for k, v in headers.items():
        if k.lower() == needle:
            return v
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="generate.py",
        description=(
            "Generate one MP4 from a prompt using the solphur2 API. "
            "Server-side defaults apply unless you override them via flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("prompt", help="Free-text prompt (REQUIRED positional argument).")
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output MP4 path. Default: ./solphur2_<random>.mp4 in the current directory.",
    )
    ap.add_argument(
        "--api-url",
        default=API_DEFAULT_URL,
        help=f"solphur2 API base URL (default: {API_DEFAULT_URL}).",
    )

    # ---- Server-side override flags ----
    # Each is None by default. Only fields the user explicitly set are
    # included in the request body so the api/server.py defaults remain
    # authoritative for anything the caller doesn't care about.
    ap.add_argument(
        "--mode",
        choices=("quality", "fast"),
        default=None,
        help="Override server-side default (quality). 'fast' = ~half wall-clock, distill LoRA stacked on Sulphur weights.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the server's random 63-bit seed. Pass to reproduce a specific output.",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override server-side default duration (10.0 s). Cap 20.0 s (logs OOD-Sulphur warning beyond 10 s).",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override server-side default width (1280). Multiple of 32; max 1920 (LTX-2.3 ceiling).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override server-side default height (704). Multiple of 32; max 1088.",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override server-side default fps (24). 24 / 25 are Sulphur-in-distribution.",
    )
    ap.add_argument(
        "--no-enhance",
        dest="enhance",
        action="store_false",
        default=None,
        help="Skip the Sulphur Qwen3.5-9B prompt enhancer (saves ~34 s; loses descriptive elaboration).",
    )

    args = ap.parse_args()

    # Resolve output path: pick a unique default in cwd if none given.
    output: Path = args.output or Path(f"./solphur2_{uuid.uuid4().hex[:10]}.mp4")
    output = output.expanduser().resolve()
    if output.parent and not output.parent.exists():
        sys.stderr.write(f"output directory does not exist: {output.parent}\n")
        return 2

    _check_health(args.api_url)

    # Construct the request body. Only include keys the user explicitly set.
    body: dict = {"prompt": args.prompt}
    if args.mode     is not None: body["mode"]             = args.mode
    if args.seed     is not None: body["seed"]             = args.seed
    if args.duration is not None: body["duration_seconds"] = args.duration
    if args.width    is not None: body["width"]            = args.width
    if args.height   is not None: body["height"]           = args.height
    if args.fps      is not None: body["fps"]              = args.fps
    if args.enhance  is False:    body["enhance_prompt"]   = False

    print(
        f"solphur2: POST {args.api_url}/generate "
        f"(default quality mode at 1280x704x10s typically takes ~2.5 min on RTX 5090; "
        f"1080p x 20s ~14 min)...",
        flush=True,
    )

    t0 = time.monotonic()
    req = urllib.request.Request(
        f"{args.api_url.rstrip('/')}/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    partial = output.with_suffix(output.suffix + ".partial")
    try:
        with urllib.request.urlopen(req, timeout=60 * 60) as resp:
            headers = dict(resp.headers.items())
            with open(partial, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", "replace")[:500] if exc.fp else ""
        sys.stderr.write(f"API returned HTTP {exc.code}: {msg}\n")
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        sys.stderr.write(f"request transport failure: {exc!r}\n")
        return 2

    elapsed = time.monotonic() - t0
    partial.rename(output)

    size_mib = output.stat().st_size / (1024 * 1024)
    print(
        f"solphur2: {output} written ({size_mib:.1f} MiB) in {elapsed:.1f} s\n"
        f"  phases : enhance {_header(headers, 'X-Solphur2-PhaseEnhanceSeconds')} s + "
        f"submit {_header(headers, 'X-Solphur2-PhaseSubmitSeconds')} s + "
        f"comfy {_header(headers, 'X-Solphur2-PhaseComfyRunSeconds')} s\n"
        f"  config : {_header(headers, 'X-Solphur2-Resolution')} x "
        f"{_header(headers, 'X-Solphur2-Frames')} frames @ "
        f"{_header(headers, 'X-Solphur2-Fps')} fps, "
        f"mode={_header(headers, 'X-Solphur2-Mode')}, "
        f"seed={_header(headers, 'X-Solphur2-Seed')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
