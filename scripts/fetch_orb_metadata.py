#!/usr/bin/env python3
"""Download the ORB (OpenRAG Bench) arXiv metadata files through a HuggingFace mirror.

HuggingFace is unreachable from some networks; `hf-mirror.com` serves the same
repository layout. This script avoids the `datasets` dependency entirely and only
needs the standard library, so it works before any virtualenv is set up.

Usage:
    python fetch_orb_metadata.py --out ../meta
    python fetch_orb_metadata.py --out ../meta --endpoint https://hf-mirror.com

Outputs (in --out):
    pdf_urls.json  queries.json  qrels.json  answers.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "vectara/open_ragbench"
SUBDIR = "pdf/arxiv"
FILES = ("pdf_urls.json", "queries.json", "qrels.json", "answers.json")
DEFAULT_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def build_url(endpoint: str, filename: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{REPO}/resolve/main/{SUBDIR}/{filename}"


def download(url: str, dest: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "orb-subset-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise ValueError("empty response")
    # Fail loudly on an HTML error page instead of writing it as JSON.
    head = data[:1]
    if head not in (b"{", b"["):
        raise ValueError(f"unexpected payload (first byte {head!r}), likely an error page")
    dest.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINTS[0]),
                    help="HuggingFace-compatible endpoint (default: $HF_ENDPOINT or hf-mirror.com)")
    ap.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    endpoints = [args.endpoint] + [e for e in DEFAULT_ENDPOINTS if e != args.endpoint]

    for filename in FILES:
        dest = args.out / filename
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            print(f"skip  {filename} (exists, {dest.stat().st_size} B)")
            continue

        last_error: Exception | None = None
        for endpoint in endpoints:
            url = build_url(endpoint, filename)
            try:
                download(url, dest)
                print(f"ok    {filename}  {dest.stat().st_size} B  <- {endpoint}")
                last_error = None
                break
            except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
                print(f"fail  {filename}  <- {endpoint}: {exc}", file=sys.stderr)
                last_error = exc

        if last_error is not None:
            raise SystemExit(
                f"could not download {filename} from any endpoint. "
                "Set HF_ENDPOINT to a reachable mirror and retry."
            )

    # Sanity check: every file must parse and the four must agree on scale.
    counts = {}
    for filename in FILES:
        with (args.out / filename).open(encoding="utf-8") as f:
            counts[filename] = len(json.load(f))
    print("\nrecord counts:")
    for filename, n in counts.items():
        print(f"  {filename:16s} {n}")


if __name__ == "__main__":
    main()
