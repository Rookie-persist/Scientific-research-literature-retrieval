"""从语料中抽取「全称 (缩写)」对，产出 M1 用的领域词典。

只依赖语料本身，不引外部词表：论文里第一次出现缩写时几乎都会给出全称，
这条规律使得「从自己的库抽词典」比维护一份通用缩写表更贴合领域。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
from collections import Counter

# 全称 (缩写)：全称 1–7 个词，缩写 2–7 个大写字母/数字
PAIR = re.compile(r"\b([A-Z][A-Za-z\-]+(?:\s+(?:of|for|and|in|on|the|a|an|to|[A-Z][A-Za-z\-]+)){0,6})\s*\(([A-Z][A-Z0-9\-]{1,6})s?\)")
# 缩写本身
ACRO = re.compile(r"\b([A-Z][A-Z0-9\-]{1,6})s?\b")


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def is_valid_pair(acro: str, name: str) -> bool:
    """缩写必须是全称首字母的子序列。

    没有这条校验，跨句匹配会产出 `US -> Food and Drug Administration`
    这类错对（正则把前面句子的括号吞进来了）。数字后缀（PSID81）先剥掉。
    """
    letters = [w[0].lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]*", name)]
    wanted = [c.lower() for c in acro if c.isalpha()]
    if not wanted:
        return False
    i = 0
    for ch in letters:
        if i < len(wanted) and ch == wanted[i]:
            i += 1
    return i == len(wanted)


async def collect(db: str) -> tuple[Counter, Counter]:
    from haiku.rag.client import HaikuRAG

    names: Counter = Counter()
    seen: Counter = Counter()
    async with HaikuRAG(db) as client:
        for doc in await client.list_documents():
            # 正文不在 Document.content 上，块才是真实文本
            for chunk in await client.chunk_repository.get_by_document_id(doc.id):
                text = chunk.content or ""
                for m in PAIR.finditer(text):
                    name, acro = clean_name(m.group(1)), m.group(2)
                    if len(name.split()) < 2 or not is_valid_pair(acro, name):
                        continue
                    names[(acro, name)] += 1
                for m in ACRO.finditer(text):
                    seen[m.group(1)] += 1
    return names, seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-occurrences", type=int, default=3)
    ap.add_argument("--max-entries", type=int, default=120)
    args = ap.parse_args()

    names, seen = asyncio.run(collect(args.db))

    # 同一缩写可能对应多个全称，取定义次数最多的那个
    best: dict[str, tuple[str, int]] = {}
    for (acro, name), n in names.items():
        total = seen[acro]
        if total < args.min_occurrences:
            continue
        if acro not in best or n > best[acro][1]:
            best[acro] = (name, n)

    entries = sorted(best.items(), key=lambda kv: -seen[kv[0]])[: args.max_entries]
    payload = {
        "source": "corpus",
        "db": args.db,
        "min_occurrences": args.min_occurrences,
        "acronyms": {acro: {"expansion": name, "definitions": n, "occurrences": seen[acro]}
                     for acro, (name, n) in entries},
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"pairs seen={len(names)}  kept={len(entries)}  -> {out}")
    for acro, (name, n) in entries[:15]:
        print(f"  {acro:8s} x{seen[acro]:3d}  {name[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
