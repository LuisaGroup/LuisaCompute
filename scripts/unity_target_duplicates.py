#!/usr/bin/env python3
"""Report internal-linkage names that two translation units of the same build
target declare at the same effective scope (latent unity-build collisions).

These names only collide when the unity build puts the two files into one
batch, so the check is *target* wide on purpose: re-batching (adding one file)
is enough to turn a latent collision into a compile error.

Usage: python scripts/unity_target_duplicates.py
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unity_symbol_scan import scan, target_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEREST = {"static_function", "static_decl", "anon_function", "anon_decl", "anon_class"}


def main() -> int:
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=ROOT).stdout.split()
    by_target = collections.defaultdict(list)
    for f in tracked:
        if not f.endswith(".cpp"):
            continue
        t, b = target_of(f)
        if (b or 0) > 1:
            by_target[t].append(f)
    cache = {}
    total = 0
    for tgt, files in sorted(by_target.items()):
        syms = collections.defaultdict(list)
        for f in files:
            if not os.path.isfile(os.path.join(ROOT, f)):
                continue
            if f not in cache:
                cache[f] = scan(os.path.join(ROOT, f))
            for it in cache[f]:
                if it["kind"] in INTEREST:
                    syms[(it["scope"], it["name"])].append((f, it))
        dups = {k: v for k, v in syms.items() if len({x[0] for x in v}) > 1}
        if not dups:
            continue
        print(f"\n==== {tgt}  ({len(dups)} duplicated internal names, {len(files)} files)")
        for (scope, name), v in sorted(dups.items()):
            total += 1
            print(f"  {scope}::{name}")
            for f, it in sorted(v, key=lambda x: (x[0], x[1]["line"])):
                print(f"      {f}:{it['line']}  {it['kind']}")
    print(f"\nTOTAL latent duplicate internal names: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
