#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ LSP Client using httpx to communicate with cpp_lsp_server.

Subcommands:
    check   – Check syntax of a C++ file.
    symbol  – Query symbol information (definition, declaration, hover, …).
"""

import argparse
import json
import sys
from pathlib import Path

import httpx


def check_syntax(client: httpx.Client, args) -> int:
    file_path = Path(args.file).resolve()
    payload = {
        "file_path": str(file_path),
        "content": args.content,
        "verbose": args.verbose,
        "timeout": args.timeout,
    }
    try:
        r = client.post("/check_syntax", json=payload)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return 2

    data = r.json()
    if not data.get("success"):
        print("Request failed.")
        return 2

    diagnostics = data.get("diagnostics", [])
    if not diagnostics:
        print("[OK] No issues found!")
        return 0

    if args.verbose:
        for line in data.get("formatted", []) or []:
            print(line)

    print(
        f"Total: {data.get('errors', 0)} error(s), "
        f"{data.get('warnings', 0)} warning(s)"
    )
    return 1 if data.get("errors", 0) > 0 else 0


def symbol(client: httpx.Client, args) -> int:
    file_path = Path(args.file).resolve()
    payload = {
        "file_path": str(file_path),
        "line": args.line,
        "character": args.character,
        "action": args.action,
        "timeout": args.timeout,
    }
    try:
        r = client.post("/symbol", json=payload)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return 2

    data = r.json()
    result = data.get("result")
    if result is None:
        print("No result found.")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="C++ LSP Client")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="HTTP request timeout in seconds",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # check
    # ------------------------------------------------------------------
    check_parser = subparsers.add_parser("check", help="Check syntax of a C++ file")
    check_parser.add_argument("file", help="Path to the C++ source file")
    check_parser.add_argument(
        "--content",
        default=None,
        help="Override file content (useful for ad-hoc snippets)",
    )
    check_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose diagnostic information",
    )
    check_parser.add_argument(
        "--lsp-timeout",
        type=float,
        default=10.0,
        dest="timeout",
        help="LSP internal timeout",
    )

    # ------------------------------------------------------------------
    # symbol
    # ------------------------------------------------------------------
    symbol_parser = subparsers.add_parser(
        "symbol", help="Query symbol metadata (definition, hover, …)"
    )
    symbol_parser.add_argument("file", help="Path to the C++ source file")
    symbol_parser.add_argument(
        "line",
        type=int,
        nargs="?",
        default=0,
        help="0-based line number (ignored for documentSymbol)",
    )
    symbol_parser.add_argument(
        "character",
        type=int,
        nargs="?",
        default=0,
        help="0-based character number (ignored for documentSymbol)",
    )
    symbol_parser.add_argument(
        "--action",
        default="definition",
        choices=[
            "definition",
            "declaration",
            "typeDefinition",
            "implementation",
            "references",
            "hover",
            "documentSymbol",
        ],
        help="LSP action to perform",
    )
    symbol_parser.add_argument(
        "--lsp-timeout",
        type=float,
        default=10.0,
        dest="timeout",
        help="LSP internal timeout",
    )

    args = parser.parse_args()

    with httpx.Client(base_url=args.server, timeout=args.timeout) as client:
        if args.command == "check":
            return check_syntax(client, args)
        elif args.command == "symbol":
            return symbol(client, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
