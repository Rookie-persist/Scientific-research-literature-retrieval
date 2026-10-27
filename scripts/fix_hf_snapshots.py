#!/usr/bin/env python3
"""Populate the HuggingFace cache snapshots from Docling's flat model directories.

The problem
-----------
On this machine `huggingface_hub` cannot create the symlinks its cache layout
relies on. `snapshot_download()` reports success, the blobs land in
`~/.cache/huggingface/hub/models--*/blobs/`, but `snapshots/<sha>/` stays empty.
Transformers then resolves a repo id to that empty path and fails with
"Image processor config not found" or "Couldn't instantiate the backend tokenizer".

Docling's own downloader writes **real files** into
`~/.cache/docling/models/<org>--<name>/`, so the bytes are already on disk — only
the HF cache view is broken. Re-downloading with `HF_HUB_DISABLE_SYMLINKS=1`
would work but costs another ~1.3 GB. This script copies instead, preserving
subdirectories (the TableFormer weights live under `model_artifacts/`).

Usage:
    python scripts/fix_hf_snapshots.py --dry-run
    python scripts/fix_hf_snapshots.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

DOCLING_CACHE = Path.home() / ".cache" / "docling" / "models"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
ENDPOINT = "https://hf-mirror.com"

SKIP_PARTS = {".cache", "__pycache__"}


def payload_files(src: Path) -> list[Path]:
    """Every real file under src, excluding HuggingFace's own metadata dir."""
    return [
        p
        for p in src.rglob("*")
        if p.is_file() and not (set(p.relative_to(src).parts) & SKIP_PARTS)
    ]


def repo_sha(repo_id: str, src: Path) -> str | None:
    """Resolve the commit sha: local ref, Docling's trees cache, then the mirror."""
    local = HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    if local.exists():
        sha = local.read_text(encoding="utf-8").strip()
        if sha:
            return sha
    # Docling records the commit as .cache/huggingface/trees/<sha>.json
    trees = src / ".cache" / "huggingface" / "trees"
    if trees.exists():
        names = [p.stem for p in trees.glob("*.json") if len(p.stem) == 40]
        if names:
            return names[0]
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/api/models/{repo_id}", timeout=20) as resp:
            data = json.load(resp)
        return data.get("sha")
    except Exception as exc:  # noqa: BLE001
        print(f"  warn  cannot resolve sha for {repo_id}: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DOCLING_CACHE.exists():
        raise SystemExit(f"no docling cache at {DOCLING_CACHE}")

    total_mb = 0.0
    for src in sorted(p for p in DOCLING_CACHE.iterdir() if p.is_dir()):
        if src.name in SKIP_PARTS or "--" not in src.name:
            continue
        repo_id = src.name.replace("--", "/", 1)
        files = payload_files(src)
        if not files:
            print(f"skip  {repo_id}: no files")
            continue

        sha = repo_sha(repo_id, src)
        if not sha:
            print(f"skip  {repo_id}: unknown sha")
            continue

        dst = HF_CACHE / f"models--{src.name}" / "snapshots" / sha
        existing = len(payload_files(dst)) if dst.exists() else 0
        if existing >= len(files):
            print(f"ok    {repo_id}: snapshot already populated ({existing} files)")
            continue

        size = sum(p.stat().st_size for p in files) / 1e6
        total_mb += size
        print(f"copy  {repo_id}: {len(files)} files, {size:.0f} MB -> snapshots/{sha[:8]}...")
        if args.dry_run:
            continue

        for f in files:
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(f, target)

        ref = HF_CACHE / f"models--{src.name}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha, encoding="utf-8")

    print(f"\ndone ({'dry run, ' if args.dry_run else ''}{total_mb:.0f} MB)")


if __name__ == "__main__":
    main()
