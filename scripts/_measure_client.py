#!/usr/bin/env python3
"""
scripts/_measure_client.py — internal helper for scripts/measure.sh.

Single responsibility: POST one /generate request to the solphur2 API and
persist the resulting MP4 + HTTP response headers under --out-dir. The
request body contains the user-supplied prompt plus any envelope overrides
(width, height, duration, fps, mode, seed, enhance) that the caller
explicitly sets. Anything not set falls through to the
api/server.py:GenerateRequest defaults.

The leading underscore in the filename marks this as a private helper.
scripts/measure.sh is the supported entry point; do not run this script
directly unless you know what you're trading off.

Why a dedicated Python client rather than `curl` from bash:
  • The /generate body is JSON. JSON-escaping a free-text prompt from
    bash requires either a temp file or an unsanitized environment hand-off.
    Both are worse than letting Python's `json.dumps` produce the body
    from a regular CLI argument.
  • The response headers contain the X-Solphur2-Phase* timings that the
    measurement summary depends on. Capturing them programmatically (via
    urllib) is more robust than parsing `curl -D` output across versions.
  • All CLI inputs are explicit CLI args — no environment variables.

Exit codes:
    0   HTTP 200 + non-empty MP4 written
    1   request reached the API but the API returned non-200
    2   transport failure (connection refused, DNS, timeout)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="POST one /generate request; persist MP4 + headers. Any envelope "
                    "override flag you set is included in the request body; anything you "
                    "don't set falls through to the server-side default in api/server.py."
    )
    ap.add_argument("--prompt", required=True, help="Free-text prompt for /generate (REQUIRED)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (REQUIRED)")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000", help="solphur2 API base URL")
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Read timeout. Default 1800 s covers the largest known-working envelope (1920×1088 × 20 s × fast, ~445 s); also long enough to catch the EINVAL crash at 1080p × 20 s × quality (~1080 s).",
    )

    # Envelope overrides — only included in the body if explicitly set.
    ap.add_argument("--width", type=int, default=None, help="Override server-default width.")
    ap.add_argument("--height", type=int, default=None, help="Override server-default height.")
    ap.add_argument("--duration-seconds", type=float, default=None,
                    help="Override server-default duration in seconds.")
    ap.add_argument("--fps", type=int, default=None, help="Override server-default fps.")
    ap.add_argument("--mode", choices=("quality", "fast"), default=None,
                    help="Override server-default mode.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the server's per-request random seed (for reproducibility).")
    ap.add_argument("--no-enhance", dest="enhance", action="store_false", default=None,
                    help="Skip the prompt enhancer (saves ~25-35 s; less descriptive prompt).")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build body: prompt is mandatory; every other field is only included
    # if the caller explicitly set the flag. Anything omitted falls through
    # to GenerateRequest's pydantic default in api/server.py.
    body_obj: dict = {"prompt": args.prompt}
    if args.width            is not None: body_obj["width"]            = args.width
    if args.height           is not None: body_obj["height"]           = args.height
    if args.duration_seconds is not None: body_obj["duration_seconds"] = args.duration_seconds
    if args.fps              is not None: body_obj["fps"]              = args.fps
    if args.mode             is not None: body_obj["mode"]             = args.mode
    if args.seed             is not None: body_obj["seed"]             = args.seed
    if args.enhance          is False:    body_obj["enhance_prompt"]   = False

    body = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        f"{args.api_url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    status: int
    headers_obj: object
    payload: bytes
    try:
        with urllib.request.urlopen(req, timeout=args.timeout_seconds) as resp:
            status = resp.status
            headers_obj = resp.headers
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers_obj = exc.headers
        payload = exc.read() or b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        sys.stderr.write(f"request transport failure: {exc!r}\n")
        return 2
    elapsed = time.monotonic() - t0

    (args.out_dir / "run.mp4").write_bytes(payload)
    with (args.out_dir / "headers.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"HTTP/1.1 {status}\n")
        for key, value in headers_obj.items():
            fh.write(f"{key}: {value}\n")

    print(
        f"_measure_client: status={status} elapsed={elapsed:.1f}s "
        f"bytes={len(payload)} out={args.out_dir / 'run.mp4'}"
    )
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
