#!/usr/bin/env python3
"""Preflight check for a local Ollama used as haiku.rag's QA model.

Checks, in order:
  1. the server is reachable and reports a version;
  2. the configured model is present locally;
  3. it answers with thinking disabled (a thinking model that is left on can spend
     the whole token budget on reasoning and return an empty `response`);
  4. whether the server exposes /api/embed, and whether an embedding model exists.

Point 3 matters more than it looks: with `think` left at the model default, a
short answer comes back empty and `done_reason` is `length`. haiku.rag's QA would
then record an empty answer that looks like a retrieval failure.

Usage:
    python check_ollama.py --model qwen3.5:9b-q4_K_M
    python check_ollama.py --model qwen3.5:9b-q4_K_M --embed-model bge-m3

Exit code 0 when the model can answer; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(base: str, path: str, payload: dict, timeout: int = 180) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": body[:300]}


def get(base: str, path: str, timeout: int = 15) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}{path}", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        return 0, {"error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True, help="QA model name as shown by /api/tags")
    ap.add_argument("--embed-model", default=None, help="optional embedding model to probe")
    args = ap.parse_args()

    ok = True

    status, version = get(args.base, "/api/version")
    if status != 200:
        print(f"FAIL  server unreachable at {args.base}: {version.get('error')}")
        raise SystemExit(1)
    print(f"ok    server {version.get('version')}")

    status, tags = get(args.base, "/api/tags")
    names = [m.get("name", "") for m in tags.get("models", [])]
    print(f"ok    models: {', '.join(names) if names else '(none)'}")

    if args.model not in names:
        print(f"FAIL  model {args.model!r} not found; run: ollama pull {args.model}")
        raise SystemExit(1)
    print(f"ok    model {args.model} present")

    # Thinking disabled: the only mode in which a short answer is usable.
    status, out = post(args.base, "/api/generate", {
        "model": args.model,
        "prompt": "Reply with exactly: OK",
        "stream": False,
        "think": False,
        "options": {"num_predict": 32, "temperature": 0},
    })
    answer = (out.get("response") or "").strip()
    if status == 200 and answer:
        tps = out.get("eval_count", 0) / (out.get("eval_duration", 1) / 1e9)
        print(f"ok    think=false -> {answer[:40]!r}  ({tps:.0f} tok/s)")
    else:
        ok = False
        print(f"FAIL  think=false produced no answer (status={status}, "
              f"done_reason={out.get('done_reason')}, error={out.get('error')})")

    # Contrast: what happens with the model's default thinking behaviour.
    status, out = post(args.base, "/api/generate", {
        "model": args.model,
        "prompt": "Reply with exactly: OK",
        "stream": False,
        "options": {"num_predict": 32, "temperature": 0},
    })
    if status == 200 and not (out.get("response") or "").strip() and out.get("thinking"):
        print("note  default thinking mode returns an empty `response` "
              "(reasoning consumed the budget) — keep thinking: false in haiku.rag.yaml")

    if args.embed_model:
        status, out = post(args.base, "/api/embed",
                           {"model": args.embed_model, "input": "probe"}, timeout=60)
        if status == 200 and out.get("embeddings"):
            dim = len(out["embeddings"][0])
            print(f"ok    embed {args.embed_model} -> {dim} dims "
                  f"(set vector_dim: {dim})")
        else:
            print(f"note  embed probe failed: {out.get('error') or status}")
            print("      the server may need to be started with --embeddings, "
                  "or the model may not be an embedding model")

    print("\nready" if ok else "\nnot ready")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
