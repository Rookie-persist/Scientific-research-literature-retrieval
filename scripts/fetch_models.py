#!/usr/bin/env python3
"""Download a HuggingFace model into a plain directory (no cache symlinks).

Why this exists
---------------
`huggingface_hub` normally stores files in a content-addressed cache and links
them into a snapshot directory. On this machine that linking step silently
produces an **empty snapshot directory**: the download reports success, the
blobs land in `blobs/`, but `AutoTokenizer.from_pretrained(repo_id)` then fails
with "Couldn't instantiate the backend tokenizer".

`local_dir=` copies real files instead of linking, which works. This script
wraps that so the workaround is reproducible rather than a one-off command.

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py --model BAAI/bge-small-en-v1.5 --out models/bge-small-en-v1.5
    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_models.py

Then point the config at the directory:

    embeddings:
      model:
        provider: sentence-transformers
        name: ./models/bge-small-en-v1.5
        vector_dim: 384
    processing:
      chunking_tokenizer: ./models/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Models that haiku.rag's dev/final profiles depend on.
DEFAULT_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
}

REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default=None, help="default: models/<last path segment>")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--force", action="store_true", help="re-download even if the directory looks complete")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    # Xet bypasses the plain HTTP path and fails behind some mirrors.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    out = Path(args.out) if args.out else Path("models") / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    if not args.force and all((out / f).exists() for f in REQUIRED_FILES):
        total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"skip  {out} already present ({total/1e6:.0f} MB)")
        return

    from huggingface_hub import snapshot_download

    print(f"fetch {args.model} -> {out}  (endpoint={os.environ['HF_ENDPOINT']})")
    snapshot_download(args.model, local_dir=str(out))

    # Verify the copy actually contains files: the failure mode this script
    # exists to avoid is a "successful" download that leaves an empty directory.
    files = [p for p in out.rglob("*") if p.is_file()]
    if not files:
        print("FAIL  download reported success but the directory is empty", file=sys.stderr)
        raise SystemExit(1)

    missing = [f for f in REQUIRED_FILES if not (out / f).exists()]
    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"ok    {len(files)} files, {total:.0f} MB")
    if missing:
        print(f"FAIL  missing required files: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    expected_dim = DEFAULT_MODELS.get(args.model)
    if expected_dim:
        print(f"note  set vector_dim: {expected_dim} for {args.model}")


if __name__ == "__main__":
    main()
