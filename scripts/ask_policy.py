#!/usr/bin/env python3
"""`ask()` with the citation policy enabled — enhancement A from PROJECT-PLAN.

Why this exists
---------------
`client.ask()` registers only `RAGCapability`:

    agent = Agent(model, capabilities=[capability])

so nothing forces the model to call `cite` before answering, and a model that
skips it returns `(answer, [])`. On the D1 smoke test that produced a confident
but uncited answer about the wrong paper.

`CitationPolicyCapability` is the library's own enforcement: if an answer arrives
without citations, it redirects the model once ("record what grounded the answer
you already gave"), and the model has to call `cite` and restate.

This module reproduces `client.ask()` with both capabilities registered. It uses
one private symbol (`_AgentDeps`) because the public API has no way to add a
capability; if upstream changes that, this is the line to fix.

Usage:
    python scripts/ask_policy.py --db ./dbs/smoke.lancedb --question "..."
"""

from __future__ import annotations

import argparse
import asyncio
import time

from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client.agents import _AgentDeps
from haiku.rag.utils.models import get_model


async def ask_with_policy(client, question: str, filter: str | None = None):
    """Same contract as `client.ask()` but with citation enforcement on."""
    from pydantic_ai import Agent

    capability = create_capability(config=client._config, rag=client)
    deps = _AgentDeps(
        state={"rag": RAGState(document_filter=filter, sources=None).model_dump(mode="json")}
    )
    model = get_model(client._config.qa.model, client._config)
    agent = Agent(
        model,
        deps_type=_AgentDeps,
        capabilities=[capability, citation_policy()],
    )
    result = await agent.run(question, deps=deps)

    state = RAGState.model_validate(deps.state["rag"])
    citations = [
        state.citation_index[cid] for cid in state.citations if cid in state.citation_index
    ]
    return result.output, citations


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./dbs/smoke.lancedb")
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(args.db) as client:
        t0 = time.perf_counter()
        answer, citations = await ask_with_policy(client, args.question)
        dt = time.perf_counter() - t0

    print(f"\n{dt:.1f}s  citations={len(citations)}")
    print("answer:", answer[:800])
    for c in citations[:5]:
        print(f"  - pages={c.page_numbers} headings={c.headings} uri={c.document_uri.split('/')[-1]}")


if __name__ == "__main__":
    asyncio.run(_main())
