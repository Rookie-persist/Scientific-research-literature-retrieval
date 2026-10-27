"""检索层诊断：把「答错」拆成「没检到」与「检到了没用上」。

只调用 search（嵌入检索，不涉及 LLM），几十秒可跑完。
输出 recall@k、首个金标命中位次、参考文献块占比、公式占位块占比。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import statistics
import sys

REF_HEADING = re.compile(r"\b(references?|bibliography|works cited)\b", re.I)
CITE_DENSE = re.compile(r"\[\d{1,3}\]")
FORMULA_GAP = "formula-not-decoded"


def looks_like_reference(content: str, headings: list[str]) -> bool:
    """参考文献块的两种特征：章节标题命中，或正文里密集出现 [n]。"""
    if any(REF_HEADING.search(h or "") for h in headings or []):
        return True
    cites = CITE_DENSE.findall(content or "")
    # 正常正文一段里偶尔引用几条；参考文献条目是「一条一段」
    return len(cites) >= 5 and len(content) < 1200


async def run(db: str, subset: pathlib.Path, limit: int) -> dict:
    from haiku.rag.client import HaikuRAG

    queries = json.loads((subset / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((subset / "qrels.json").read_text(encoding="utf-8"))
    ids = [ln.strip() for ln in (subset / "query_ids.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    rows = []
    async with HaikuRAG(db) as client:
        for qid in ids:
            q = queries.get(qid)
            if not q:
                continue
            gold = qrels.get(qid, {}).get("doc_id", "")
            results = await client.search(q["query"], limit=limit)
            uris = [r.document_uri for r in results]
            rank = next((i + 1 for i, u in enumerate(uris) if gold and gold in u), None)
            ref_hits = sum(1 for r in results if looks_like_reference(r.content, r.headings))
            gap_hits = sum(1 for r in results if FORMULA_GAP in (r.content or ""))
            rows.append(
                {
                    "qid": qid,
                    "type": q.get("type"),
                    "gold": gold,
                    "rank": rank,
                    "n": len(results),
                    "ref_hits": ref_hits,
                    "gap_hits": gap_hits,
                    "top_score": results[0].score if results else 0.0,
                    "has_pages": bool(results and results[0].page_numbers),
                }
            )

    n = len(rows)
    hit = sum(1 for r in rows if r["rank"])
    ranks = [r["rank"] for r in rows if r["rank"]]
    summary = {
        "questions": n,
        "k": limit,
        "recall_at_k": round(hit / n, 4) if n else 0.0,
        "mrr": round(sum(1 / r for r in ranks) / n, 4) if n else 0.0,
        "median_rank": statistics.median(ranks) if ranks else None,
        "avg_ref_chunks_in_topk": round(sum(r["ref_hits"] for r in rows) / n, 2) if n else 0.0,
        "avg_formula_gap_in_topk": round(sum(r["gap_hits"] for r in rows) / n, 2) if n else 0.0,
        "top1_has_pages": round(sum(1 for r in rows if r["has_pages"]) / n, 4) if n else 0.0,
        "top1_score_median": round(statistics.median([r["top_score"] for r in rows]), 4) if n else 0.0,
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = asyncio.run(run(args.db, pathlib.Path(args.subset), args.limit))

    out.write_text(json.dumps(data["rows"], ensure_ascii=False, indent=1), encoding="utf-8")
    s = data["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print("\n未命中金标的题目：")
    for r in data["rows"]:
        if not r["rank"]:
            print(f"  {r['qid'][:8]}  top_score={r['top_score']:.4f}  ref_chunks={r['ref_hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
