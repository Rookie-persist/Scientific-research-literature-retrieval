#!/usr/bin/env python3
"""Download the PDFs listed in a subset's pdf_urls.json, resumably and politely.

Why a dedicated downloader instead of a loop of curl:
  * arXiv throttles bulk access. This keeps concurrency low, spaces requests, and
    backs off on 403/429 instead of hammering the endpoint.
  * Ingestion takes hours; downloads get interrupted. Every file is validated on
    arrival, so a rerun skips what already landed and retries only what failed.
  * A manifest records what failed and why, which the subset stats should report
    rather than silently shrinking the corpus.

Usage:
    python download_pdfs.py --subset ../tmp_150 --out ../pdfs
    python download_pdfs.py --subset ../tmp_150 --out ../pdfs --workers 2 --retries 5

Outputs:
    <out>/<paper_id>.pdf        validated PDFs (first bytes must be %PDF)
    <out>/manifest.jsonl        one line per paper: id, status, bytes, attempts, error
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PDF_MAGIC = b"%PDF"
MIN_BYTES = 10_000
USER_AGENT = "orb-subset-builder/1.0 (research; contact: local)"


def is_valid_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_BYTES:
            return False
        with path.open("rb") as f:
            return f.read(4) == PDF_MAGIC
    except OSError:
        return False


def fetch(paper_id: str, url: str, dest: Path, retries: int, timeout: int) -> dict:
    record = {"paper_id": paper_id, "url": url, "status": "pending", "bytes": 0, "attempts": 0}

    if is_valid_pdf(dest):
        record.update(status="exists", bytes=dest.stat().st_size)
        return record

    tmp = dest.with_suffix(".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        record["attempts"] = attempt
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES or not data.startswith(PDF_MAGIC):
                raise ValueError(f"not a PDF ({len(data)} B, head={data[:8]!r})")
            tmp.write_bytes(data)
            tmp.replace(dest)
            record.update(status="ok", bytes=len(data))
            return record
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            # 403/429 mean "slow down"; anything 4xx other than 408/429 is permanent.
            if exc.code in (403, 429):
                time.sleep(min(60, 5 * attempt) + random.uniform(0, 3))
                continue
            if 400 <= exc.code < 500 and exc.code != 408:
                record.update(status="failed_permanent", error=last_error)
                return record
            time.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))

    record.update(status="failed", error=last_error)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, type=Path, help="directory holding the subset pdf_urls.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=3, help="keep low; arXiv throttles bulk access")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="download only the first N (for a smoke test)")
    args = ap.parse_args()

    urls_file = args.subset / "pdf_urls.json"
    if not urls_file.exists():
        raise SystemExit(f"missing {urls_file}; run build_orb_subset.py first")

    with urls_file.open(encoding="utf-8") as f:
        urls: dict[str, str] = json.load(f)

    items = list(urls.items())
    if args.limit:
        items = items[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    print(f"{len(items)} papers -> {args.out}  (workers={args.workers}, retries={args.retries})")
    started = time.time()
    done = 0

    with manifest_path.open("w", encoding="utf-8") as manifest, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, pid, url, args.out / f"{pid}.pdf", args.retries, args.timeout): pid
            for pid, url in items
        }
        for fut in as_completed(futures):
            record = fut.result()
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            done += 1
            if done % 10 == 0 or record["status"] not in ("ok", "exists"):
                elapsed = time.time() - started
                rate = done / elapsed * 60 if elapsed else 0
                print(f"[{done}/{len(items)}] {record['status']:18s} {record['paper_id']} "
                      f"({rate:.1f}/min)")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    ok = sum(1 for r in records if r["status"] in ("ok", "exists"))
    failed = [r for r in records if r["status"] not in ("ok", "exists")]
    total_mb = sum(r["bytes"] for r in records) / 1e6

    print(f"\ndone: {ok}/{len(records)} ok, {len(failed)} failed, {total_mb:.0f} MB, "
          f"{time.time() - started:.0f}s")
    if failed:
        print("failed papers (excluded from the corpus):")
        for r in failed[:20]:
            print(f"  {r['paper_id']}  {r['status']}  {r.get('error', '')}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more; see {manifest_path}")
        print("\nRerun the same command to retry only the missing files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
