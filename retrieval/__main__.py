"""Query the local RAG index from a terminal."""

from __future__ import annotations

import argparse
import json

from .store import RetrievalStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--category")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    store = RetrievalStore()
    try:
        print(json.dumps(store.search(args.query, args.category, args.top_k), ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0


raise SystemExit(main())
