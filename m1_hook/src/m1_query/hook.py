"""M1 查询理解 Hook。

两个能力，都可单独开关：

1. **双向缩写扩展**：查询里出现词典中的缩写时补上全称，出现全称时补上缩写。
   采用「追加」而非「替换」，因为混合检索的两路（向量 + FTS）都吃这个改写：
   保留原词让 FTS 仍能命中缩写，追加全称让向量侧拿到语义更完整的查询。

2. **语料覆盖门控**：查询里的实词如果在本库词表中找不到（允许 4 字符前缀
   的词干匹配），说明这个问题大概率不在语料范围内。策略可配置为
   `off / annotate / cap / starve`。

默认两个能力都关闭。这不是保守，而是 D2 实测的结论：在本语料上缩写扩展
的检索收益为零（缩写与全称两种问法 recall@5 均为 100%），门控的收益与风险
也未见得划算。详见 `D2-RECORD.md`。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from haiku.rag.hooks import Hook, SearchRequest

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    """a an the is are was were be been being do does did of for in on at to from by
    with about into over under what which who whom how why when where that this these
    those it its as and or not no than then there their they we you i he she them his
    her our your can could should would may might will shall have has had using use used
    based between within during""".split()
)
TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
NO_MATCH = "uri = '__m1_no_match__'"


def _words(text: str) -> list[str]:
    return [w.lower() for w in TOKEN.findall(text) if w.lower() not in STOPWORDS]


class QueryUnderstandingHook(Hook):
    """按配置改写查询、并在需要时收窄检索。"""

    def __init__(
        self,
        dictionary: str = "m1/acronyms.json",
        expand: bool = True,
        max_expansions: int = 2,
        overfetch: int = 1,
        coverage_policy: str = "off",
        coverage_minlen: int = 4,
        trace: str | None = None,
    ) -> None:
        self.dictionary_path = Path(dictionary)
        self.expand = expand
        self.max_expansions = max_expansions
        self.overfetch = max(1, overfetch)
        # YAML 1.1 把裸写的 off 解析成布尔 False，这里统一归一化
        if isinstance(coverage_policy, bool):
            coverage_policy = "off" if not coverage_policy else "on"
        self.coverage_policy = str(coverage_policy).lower()
        self.coverage_minlen = coverage_minlen
        self.trace_path = Path(trace) if trace else None

        self._acronyms: dict[str, str] = {}
        self._expansions: dict[str, str] = {}
        self._vocab: Counter[str] = Counter()
        self._stems: Counter[str] = Counter()
        self._acro_re: re.Pattern[str] | None = None
        self._ready = False
        self.stats: Counter[str] = Counter()

    # ---------- 生命周期 ----------

    @asynccontextmanager
    async def lifespan(self, client):
        """在客户端打开期间建好词典与词表，避免首次检索时阻塞。"""
        self._load_dictionary()
        await self._build_vocab(client)
        self._ready = True
        logger.info(
            "m1-query ready: %d acronyms, %d vocab terms, policy=%s",
            len(self._acronyms),
            len(self._vocab),
            self.coverage_policy,
        )
        yield

    def _load_dictionary(self) -> None:
        if not self.dictionary_path.exists():
            logger.warning("m1-query: dictionary %s missing; expansion disabled", self.dictionary_path)
            return
        payload = json.loads(self.dictionary_path.read_text(encoding="utf-8"))
        entries = payload.get("acronyms", {})
        self._acronyms = {k: v["expansion"] for k, v in entries.items()}
        self._expansions = {v["expansion"].lower(): k for k, v in entries.items()}
        if self._acronyms:
            # 长缩写优先，避免 WISE 被 WI 抢先匹配
            keys = sorted(self._acronyms, key=len, reverse=True)
            self._acro_re = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")s?\b")

    async def _build_vocab(self, client) -> None:
        for doc in await client.list_documents():
            for chunk in await client.chunk_repository.get_by_document_id(doc.id):
                for w in _words(chunk.content or ""):
                    self._vocab[w] += 1
        for w, n in self._vocab.items():
            for length in range(self.coverage_minlen, len(w) + 1):
                self._stems[w[:length]] += n

    # ---------- 改写与门控 ----------

    def _expand_query(self, query: str) -> tuple[str, list[str]]:
        if not self.expand or self._acro_re is None:
            return query, []
        added: list[str] = []
        lowered = query.lower()

        for acro in self._acro_re.findall(query):
            if len(added) >= self.max_expansions:
                break
            expansion = self._acronyms.get(acro)
            # 查询里已经写出全称就不再重复
            if expansion and expansion.lower() not in lowered:
                added.append(f"{acro} ({expansion})")

        for expansion, acro in self._expansions.items():
            if len(added) >= self.max_expansions:
                break
            if expansion in lowered and acro.lower() not in lowered.split():
                added.append(acro)

        if not added:
            return query, []
        return f"{query} [{' ; '.join(added)}]", added

    def _unknown_terms(self, query: str) -> list[str]:
        missing = []
        for w in _words(query):
            if self._vocab.get(w):
                continue
            if len(w) >= self.coverage_minlen and self._stems.get(w[: self.coverage_minlen]):
                continue
            missing.append(w)
        return missing

    @staticmethod
    def _capitalized(query: str) -> set[str]:
        """查询里原样大写的词：多半是专名（Tesla / FIFA），不是语料缺失的证据。"""
        return {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z\-]{2,}", query)}

    def _gate_fires(self, query: str, missing: list[str]) -> bool:
        """是否触发门控。`conservative` 只认两类高置信信号。

        实测依据：Tesla 是专名、语料里只有 TSLA，把它算作"语料缺失"会误杀一个
        本来能答的问题（引用 2 -> 0）。所以单个大写专名不触发，其余才触发。
        """
        if not missing:
            return False
        if self.coverage_policy != "conservative":
            return True
        if len(missing) >= 2:
            return True
        return missing[0] not in self._capitalized(query)

    async def before_search(self, client, request: SearchRequest) -> SearchRequest:
        if not self._ready or not isinstance(request.query, str):
            return request

        original = request.query
        query, added = self._expand_query(original)
        missing = self._unknown_terms(query) if self.coverage_policy != "off" else []
        gate = self._gate_fires(query, missing)

        if self.overfetch > 1:
            request.limit = request.limit * self.overfetch

        policy = self.coverage_policy
        if gate and policy == "cap":
            request.limit = max(1, request.limit // 2)
        elif gate and policy in ("starve", "conservative"):
            # 没有命中任何文档的过滤条件：让 agent 立刻看到 "No results found."
            request.filter = NO_MATCH if not request.filter else f"({request.filter}) AND ({NO_MATCH})"

        request.query = query
        self._record(original, query, added, missing if gate else [], policy, request.limit)
        return request

    def _record(self, original, query, added, missing, policy, limit) -> None:
        if added:
            self.stats["expanded"] += 1
        if missing:
            self.stats["coverage_miss"] += 1
        if policy != "off" and missing:
            self.stats[f"gated_{policy}"] += 1
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "original": original,
                        "rewritten": query,
                        "expansions": added,
                        "unknown_terms": missing,
                        "policy": policy,
                        "limit": limit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def create(**options) -> QueryUnderstandingHook:
    """Entry point factory; ``config.hooks[].options`` are passed as kwargs."""
    return QueryUnderstandingHook(**options)
