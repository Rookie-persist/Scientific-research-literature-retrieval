#!/usr/bin/env python3
"""D1 baseline: run ask() over a fixed question set and record everything.

Design choices
---------------
* **16 ORB questions, stratified**: 8 extractive + 8 abstractive, sampled at an
  even stride over sorted query ids. Deterministic, so reruns are comparable.
* **4 negatives**: 2 clearly out of domain, 2 topically adjacent but absent from
  the corpus. These test refusal, which retrieval metrics never measure.
* **One process, sequential calls**: each `ask()` is an agent run with tool
  calls; starting a process per question would add ~5 s of model loading each
  time and make latency numbers meaningless.

Output is JSONL (one row per question) plus a printed summary. Answer quality is
graded by a human afterwards — an LLM judge here would be the same model that
produced the answer.

Usage:
    python scripts/run_baseline.py --db ./dbs/smoke.lancedb --subset subset_20 --out eval/baseline.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

NEGATIVES = [
    ("neg1", "What dosage of metformin is recommended for type 2 diabetes?", "out-of-domain: medicine"),
    ("neg2", "Who won the 2022 FIFA World Cup final?", "out-of-domain: world knowledge"),
    ("neg3", "What speedup does FlashAttention-3 report on H100 GPUs?", "adjacent: not in corpus"),
    ("neg4", "What is the melting point of tungsten in degrees Celsius?", "out-of-domain: physical constant"),
]

REFUSAL_MARKERS = (
    "does not contain", "do not contain", "no information", "not contain",
    "cannot find", "can't find", "couldn't find", "could not find",
    "not mentioned", "no relevant", "not provided", "not specified",
    "unable to find", "doesn't contain", "does not mention",
)


def stratified_sample(queries: dict, answers: dict, per_type: int = 8) -> list[dict]:
    """Pick `per_type` extractive and `per_type` abstractive questions at an even stride."""
    picked: list[dict] = []
    for qtype in ("extractive", "abstractive"):
        ids = sorted(qid for qid, d in queries.items() if d.get("type") == qtype)
        if not ids:
            continue
        stride = max(1, len(ids) // per_type)
        chosen = ids[::stride][:per_type]
        for qid in chosen:
            picked.append({
                "qid": qid,
                "type": qtype,
                "question": queries[qid]["query"],
                "gold": answers.get(qid, ""),
                "category": "orb",
            })
    return picked


def refusal(answer: str, n_citations: int) -> bool:
    text = answer.lower()
    return n_citations == 0 or any(m in text for m in REFUSAL_MARKERS)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="./dbs/smoke.lancedb")
    ap.add_argument("--subset", default="subset_20")
    ap.add_argument("--out", default="eval/baseline.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    ap.add_argument("--policy", action="store_true", help="enable CitationPolicyCapability (enhancement A)")
    ap.add_argument("--qids", default=None, help="comma-separated ORB qid prefixes; negatives always included")
    ap.add_argument("--no-negatives", action="store_true", help="skip the 4 negatives")
    ap.add_argument("--negatives-file", default=None, help="JSON list of {qid, question}; overrides the built-in 4")
    ap.add_argument("--negatives-only", action="store_true", help="run only the negatives")
    args = ap.parse_args()

    from haiku.rag.client import HaikuRAG

    if args.policy:
        from ask_policy import ask_with_policy
        ask_fn = ask_with_policy
    else:
        ask_fn = lambda client, q: client.ask(q)  # noqa: E731

    subset = Path(args.subset)
    queries = json.loads((subset / "queries.json").read_text(encoding="utf-8"))
    answers = json.loads((subset / "answers.json").read_text(encoding="utf-8"))

    items = stratified_sample(queries, answers)
    if args.qids:
        wanted = [x.strip() for x in args.qids.split(",") if x.strip()]
        items = [i for i in items if any(i["qid"].startswith(w) for w in wanted)]
    if args.negatives_only:
        items = []
    if not args.no_negatives:
        if args.negatives_file:
            extra = json.loads(Path(args.negatives_file).read_text(encoding="utf-8"))
            for e in extra:
                items.append({"qid": e["qid"], "type": "negative", "question": e["question"],
                              "gold": "", "category": "out-of-domain"})
        else:
            for qid, q, note in NEGATIVES:
                items.append({"qid": qid, "type": "negative", "question": q, "gold": "", "category": note})
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(items)} questions -> {out}")
    rows = []
    async with HaikuRAG(args.db) as client:
        with out.open("w", encoding="utf-8") as f:
            for i, item in enumerate(items, 1):
                t0 = time.perf_counter()
                try:
                    answer, citations = await ask_fn(client, item["question"])
                    error = None
                except Exception as exc:  # noqa: BLE001 - a failed case is data too
                    answer, citations, error = "", [], f"{type(exc).__name__}: {exc}"
                latency = time.perf_counter() - t0

                cites = [
                    {
                        "chunk_id": c.chunk_id,
                        "pages": c.page_numbers,
                        "headings": c.headings,
                        "uri": c.document_uri.split("/")[-1] if c.document_uri else None,
                    }
                    for c in citations
                ]
                row = {
                    **item,
                    "answer": answer,
                    "citations": cites,
                    "n_citations": len(cites),
                    "has_pages": any(c["pages"] for c in cites),
                    "latency_s": round(latency, 2),
                    "refused": refusal(answer, len(cites)) if item["type"] == "negative" else None,
                    "error": error,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}/{len(items)}] {item['qid'][:8]} {latency:5.1f}s "
                      f"cites={len(cites)} pages={'y' if row['has_pages'] else 'n'}"
                      f"{' ERROR' if error else ''}")

    ok = [r for r in rows if not r["error"]]
    lat = [r["latency_s"] for r in ok]
    orb = [r for r in ok if r["category"] == "orb"]
    neg = [r for r in ok if r["type"] == "negative"]

    summary = {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "orb_questions": len(orb),
        "citation_rate": round(sum(1 for r in orb if r["n_citations"] > 0) / len(orb), 3) if orb else None,
        "page_rate": round(sum(1 for r in orb if r["has_pages"]) / len(orb), 3) if orb else None,
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_max_s": round(max(lat), 2) if lat else None,
        "negatives_refused": sum(1 for r in neg if r["refused"]),
        "negatives_total": len(neg),
    }
    (out.parent / "baseline-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
