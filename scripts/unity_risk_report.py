#!/usr/bin/env python3
"""Classify internal-linkage entities by unity-build risk.

An entity is *at risk* when the (named-)scope it lives in is shared with another
translation unit of the same target: a future same-named entity in a sibling
file, or a re-batching of the unity build, then makes the two merge into one
scope.  Entities that already sit in a file-unique sub-namespace (the
`namespace { namespace foo_detail { ... } }` idiom used across this code base)
are *safe* by construction.

Usage: python scripts/unity_risk_report.py [--all] [--changed-only]
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unity_symbol_scan import scan, target_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEREST = {"static_function", "static_decl", "anon_function", "anon_decl", "anon_class"}


def unity_batches(gens="build/.gens", plat="windows"):
    """Source file -> (target, generated unity unit) as xmake actually groups them."""
    out = {}
    for tgt in sorted(os.listdir(gens)):
        d = os.path.join(gens, tgt, plat, "unity_build")
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".cpp"):
                continue
            for line in open(os.path.join(d, name), encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line.startswith("#include"):
                    continue
                p = line[line.index('"') + 1 : line.rindex('"')]
                p = os.path.normpath(os.path.join(d, p)).replace("\\", "/")
                out[p.replace(os.getcwd().replace("\\", "/") + "/", "")] = (tgt, name)
    return out


def main() -> int:
    changed_only = "--all" not in sys.argv
    changed = set()
    if changed_only:
        txt = subprocess.run(
            ["git", "log", "-50", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, cwd=ROOT).stdout
        changed = {l.strip() for l in txt.splitlines() if l.strip()}

    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=ROOT).stdout.split()
    by_target = collections.defaultdict(list)
    for f in tracked:
        if not f.endswith(".cpp"):
            continue
        t, b = target_of(f)
        if (b or 0) > 1:
            by_target[t].append(f)

    gens = unity_batches()
    total = 0
    for tgt, files in sorted(by_target.items()):
        scopes = collections.defaultdict(set)
        info = {}
        for f in files:
            if not os.path.isfile(f):
                continue
            try:
                items = scan(f)
            except Exception as e:  # noqa: BLE001
                print(f"ERR {f}: {e}", file=sys.stderr)
                continue
            info[f] = [it for it in items if it["kind"] in INTEREST]
            for it in info[f]:
                scopes[it["scope"]].add(f)
        for f in sorted(info):
            risky = [it for it in info[f] if len(scopes[it["scope"]]) > 1]
            if not risky:
                continue
            if changed_only and f not in changed:
                continue
            total += len(risky)
            batch = gens.get(f, ("?", "?"))[1]
            print(f"\n== {f}  [{tgt} {batch}]  {len(risky)} at-risk / {len(info[f])} internal")
            for s, c in collections.Counter(it["scope"] for it in risky).most_common():
                print(f"    {s}  : {c} entities (scope shared by {len(scopes[s])} files)")
            for it in risky:
                print(f"      L{it['line']:>5} {it['kind']:<14} {it['name']}")
    print(f"\nTOTAL at-risk entities: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
