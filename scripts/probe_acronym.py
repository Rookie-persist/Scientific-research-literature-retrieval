"""缩写探针：同一概念用「缩写」与「全称」两种问法，检索结果差多少。

ORB 的查询只有 9% 含词典缩写，用它测不出 M1 的效果。这个探针从词典反查
金标文档，为每个概念造一对等价查询（`What is WER?` / `What is Word Error
Rate?`），金标相同，于是差距只能来自查询措辞。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics


async def build_pairs(db: str, dictionary: pathlib.Path, min_occurrences: int, limit: int):
    from haiku.rag.client import HaikuRAG

    entries = json.loads(dictionary.read_text(encoding="utf-8"))["acronyms"]
    entries = {k: v for k, v in entries.items() if v["occurrences"] >= min_occurrences}

    pairs = []
    async with HaikuRAG(db) as client:
        docs = await client.list_documents()
        for acro, meta in entries.items():
            expansion = meta["expansion"]
            # 金标：同时出现全称与缩写的文档（定义所在处）
            gold = []
            for doc in docs:
                text = "\n".join(
                    c.content or ""
                    for c in await client.chunk_repository.get_by_document_id(doc.id)
                )
                if expansion.lower() in text.lower() and acro in text:
                    gold.append(doc.uri)
            if len(gold) != 1:  # 多义或未定义，无法判定
                continue
            pairs.append({"acro": acro, "expansion": expansion, "gold": gold[0]})

        rows = []
        for p in pairs:
            for form, q in (
                ("acro", f"What is {p['acro']}?"),
                ("full", f"What is {p['expansion']}?"),
            ):
                res = await client.search(q, limit=limit)
                rank = next(
                    (i + 1 for i, r in enumerate(res) if p["gold"] in r.document_uri), None
                )
                rows.append({**p, "form": form, "query": q, "rank": rank})
    return rows


def summarize(rows: list[dict], limit: int) -> dict:
    out = {}
    for form in ("acro", "full"):
        sub = [r for r in rows if r["form"] == form]
        hit = [r for r in sub if r["rank"]]
        out[form] = {
            "n": len(sub),
            "recall_at_k": round(len(hit) / len(sub), 4) if sub else 0.0,
            "mrr": round(sum(1 / r["rank"] for r in hit) / len(sub), 4) if sub else 0.0,
            "median_rank": statistics.median([r["rank"] for r in hit]) if hit else None,
        }
    out["k"] = limit
    out["gap_recall"] = round(out["full"]["recall_at_k"] - out["acro"]["recall_at_k"], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dictionary", default="m1/acronyms.json")
    ap.add_argument("--min-occurrences", type=int, default=20)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = asyncio.run(build_pairs(args.db, pathlib.Path(args.dictionary), args.min_occurrences, args.limit))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    s = summarize(rows, args.limit)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print("\n缩写形式未命中的：")
    for r in rows:
        if r["form"] == "acro" and not r["rank"]:
            print(f"  {r['acro']:8s} {r['expansion'][:45]:45s} -> {r['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
