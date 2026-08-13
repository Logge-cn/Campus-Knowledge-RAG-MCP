"""Command-line entry point for the local hybrid retrieval index."""

import argparse
import json

from retrieval import build_index, retrieve, search, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "search", "retrieve", "status"))
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_index(force=args.force)
    elif args.command in {"search", "retrieve"}:
        if not args.query:
            parser.error(f"{args.command} requires a query")
        result = retrieve(args.query, args.limit) if args.command == "retrieve" else search(args.query, args.limit)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
