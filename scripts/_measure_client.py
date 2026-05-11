#!/usr/bin/env python3
"""
scripts/_measure_client.py — internal helper for scripts/measure_default.sh.

Single responsibility: POST one /generate request to the solphur2 API at the
server-side defaults (i.e. the body contains ONLY the user-supplied prompt
— every other parameter falls through to the GenerateRequest defaults in
api/server.py), then persist the resulting MP4 and HTTP response headers
under --out-dir.

The leading underscore in the filename marks this as a private helper.
scripts/measure_default.sh is the supported entry point; do not run this
script directly unless you know what you're trading off.

Why a dedicated Python client rather than `curl` from bash:
  • The /generate body is JSON. JSON-escaping a free-text prompt from
    bash requires either a temp file or an unsanitized environment hand-off.
    Both are worse than letting Python's `json.dumps` produce the body
    from a regular CLI argument.
  • The response headers contain the X-Solphur2-Phase* timings that the
    measurement summary depends on. Capturing them programmatically (via
    urllib) is more robust than parsing `curl -D` output across versions.
  • All CLI inputs are explicit and required — no environment variables.

Usage (from scripts/measure_default.sh):
    python3 scripts/_measure_client.py \
        --prompt "free-text prompt" \
        --out-dir bench_runs/measure_default_YYYYMMDD-HHMMSS \
        [--api-url http://127.0.0.1:8000] \
        [--timeout-seconds 1800]

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
        description="POST one /generate request at server-side defaults; persist MP4 + headers."
    )
    ap.add_argument("--prompt", required=True, help="Free-text prompt for /generate (REQUIRED)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (REQUIRED)")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000", help="solphur2 API base URL")
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Read timeout. Default 1800 s comfortably covers quality-mode 1080p×20s.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    body = json.dumps({"prompt": args.prompt}).encode("utf-8")
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
