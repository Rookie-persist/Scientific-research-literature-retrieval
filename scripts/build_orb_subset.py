#!/usr/bin/env python3
"""Build a small, self-consistent subset of OpenRAG Bench (arXiv) for local evaluation.

Why this exists
---------------
`evaluations run orb_text --limit N` limits *questions*, not the corpus: without
`--skip-db` the harness still ingests every PDF. On a laptop that means a 15-16 GB
database and hours of ingestion before the first number appears.

This script cuts the corpus instead, and keeps the metadata consistent:

    pdf_urls.json  -> only the selected papers
    qrels.json     -> only queries whose gold document survived
    queries.json   -> same
    answers.json   -> same

The result is a subset you can ingest in tens of minutes, where every retained
question still has its gold document present. Dropping a paper without dropping
its questions would silently turn those questions into guaranteed failures.

Inputs are the four metadata files shipped in the HuggingFace dataset
`vectara/open_ragbench`, under `pdf/arxiv/`:

    pdf_urls.json   {paper_id: pdf_url}
    queries.json    {query_id: {"query": str, "type": str, "source": str}}
    qrels.json      {query_id: {"doc_id": paper_id, ...}}
    answers.json    {query_id: str}

Usage
-----
    python build_orb_subset.py --metadata-dir ./orb_meta --target-papers 150 --out ./orb_subset

Outputs (in --out):
    pdf_urls.json, queries.json, qrels.json, answers.json   subset metadata
    paper_ids.txt                                           one arXiv id per line
    stats.json                                              what was kept and why
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ("pdf_urls.json", "queries.json", "qrels.json", "answers.json")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a JSON object, got {type(data).__name__}")
    return data


def gold_doc_id(qrel: object) -> str | None:
    """qrels entries are {"doc_id": ...}; tolerate a bare string for robustness."""
    if isinstance(qrel, dict):
        doc_id = qrel.get("doc_id")
        return doc_id if isinstance(doc_id, str) and doc_id else None
    if isinstance(qrel, str) and qrel:
        return qrel
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-papers", type=int, default=150)
    ap.add_argument("--drop-image-queries", action="store_true",
                    help="Drop queries whose source mentions an image (text-only pipelines).")
    args = ap.parse_args()

    missing = [name for name in REQUIRED if not (args.metadata_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"missing metadata in {args.metadata_dir}: {', '.join(missing)}\n"
            "Download them from the HuggingFace dataset 'vectara/open_ragbench', "
            "path 'pdf/arxiv/'."
        )

    pdf_urls = load_json(args.metadata_dir / "pdf_urls.json")
    queries = load_json(args.metadata_dir / "queries.json")
    qrels = load_json(args.metadata_dir / "qrels.json")
    answers = load_json(args.metadata_dir / "answers.json")

    # Queries that can be scored: they have a gold document that actually exists in the corpus.
    usable: dict[str, str] = {}
    for qid, qrel in qrels.items():
        doc_id = gold_doc_id(qrel)
        if doc_id and doc_id in pdf_urls and qid in queries:
            usable[qid] = doc_id

    if not usable:
        raise SystemExit("no query has a gold document present in pdf_urls.json; check the inputs")

    if args.drop_image_queries:
        usable = {
            qid: doc_id
            for qid, doc_id in usable.items()
            if "image" not in str(queries[qid].get("source", "")).lower()
        }

    # Greedy: papers covering the most questions first, so a small corpus keeps many
    # scorable questions. Ties break on paper id for determinism.
    per_doc: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in usable.items():
        per_doc[doc_id].append(qid)
    ranked = sorted(per_doc, key=lambda d: (-len(per_doc[d]), d))

    selected = ranked[: args.target_papers]
    selected_set = set(selected)
    kept_qids = [qid for doc_id in selected for qid in per_doc[doc_id]]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, obj: dict) -> None:
        with (out / name).open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    dump("pdf_urls.json", {pid: pdf_urls[pid] for pid in selected})
    dump("queries.json", {qid: queries[qid] for qid in kept_qids})
    dump("qrels.json", {qid: qrels[qid] for qid in kept_qids})
    dump("answers.json", {qid: answers[qid] for qid in kept_qids if qid in answers})

    (out / "paper_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    # For the official harness: `--filter-ids` keeps only these QA cases.
    (out / "query_ids.txt").write_text("\n".join(kept_qids) + "\n", encoding="utf-8")

    qtype = Counter(str(queries[qid].get("type", "unknown")) for qid in kept_qids)
    stats = {
        "corpus_papers_total": len(pdf_urls),
        "scorable_queries_total": len(usable),
        "papers_selected": len(selected),
        "queries_kept": len(kept_qids),
        "queries_without_answer": sum(1 for qid in kept_qids if qid not in answers),
        # ~121 chunks/paper: the official ORB database holds 121,168 chunks over
        # the 1,000-paper corpus. Not 50 — chunk_size 256 plus picture chunks.
        "estimated_chunks": len(selected) * 121,
        "query_type_breakdown": dict(qtype),
        "dropped_image_queries": args.drop_image_queries,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote subset metadata to {out.resolve()}")
    print("Next: download the PDFs listed in paper_ids.txt, then ingest them into a fresh database.")


if __name__ == "__main__":
    main()
