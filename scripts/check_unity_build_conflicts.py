#!/usr/bin/env python3
"""Fail when a unity (jumbo) build batch would merge two identically named
file-local symbols.

A unity build concatenates several translation units of one target into a single
translation unit, so two files that both declare e.g. `namespace { bool is_zero(..) }`
or `static void clone_metadata(..)` at the same namespace scope either fail to
compile (redefinition) or silently change overload resolution.  xmake merges the
files of a target in fixed-size batches (`_config_project{batch_size = ...}`), so
a collision only shows up once two files land in the same batch, i.e. it is easy
to introduce and hard to notice.

Checks performed
----------------
1. exact batches (only when `build/.gens/<target>/<plat>/unity_build` exists, i.e.
   after a build): every generated unity unit is checked for internal-linkage
   names declared by more than one of its included files;
2. conservative target-wide check: every unity-built target is checked for
   internal-linkage names declared by more than one of its files, whatever the
   current batching is.  This is the check that also catches collisions that only
   appear after adding/removing a file (re-batching).

`src/ext/**` (vendored third-party code) is skipped.

Usage:
  python scripts/check_unity_build_conflicts.py [--gens build/.gens] [--plat windows]
                                                [--json out.json] [--verbose]
Exit codes: 0 clean, 1 conflicts found, 2 checker could not run.
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unity_symbol_scan import (INTERNAL_KINDS, ROOT, parse_unity_batches,  # noqa: E402
                               scan, scan_macros, target_of, tracked_cpp)

SKIP_PREFIXES = ("src/ext/",)


def collect(files):
    syms = collections.defaultdict(list)
    for f in files:
        if any(f.startswith(p) for p in SKIP_PREFIXES):
            continue
        path = os.path.join(ROOT, f)
        if not os.path.isfile(path):
            continue
        for it in scan(path):
            if it["kind"] in INTERNAL_KINDS:
                syms[(it["scope"], it["name"])].append((f, it))
    return syms


def collect_macros(files):
    """Macro names defined by more than one file, and the files defining them.

    A macro defined in one translation unit of a batch stays defined for every
    file merged after it, so two files defining the same macro is the same class
    of hazard as two files declaring the same helper.
    """
    defined = collections.defaultdict(list)
    leaked = collections.defaultdict(list)
    for f in files:
        if any(f.startswith(p) for p in SKIP_PREFIXES):
            continue
        path = os.path.join(ROOT, f)
        if not os.path.isfile(path):
            continue
        defs, undefs = scan_macros(path)
        for name, line in defs.items():
            defined[name].append(f"{f}:{line}")
            if name not in undefs:
                leaked[name].append(f"{f}:{line}")
    dupes = {k: v for k, v in defined.items() if len({x.split(":")[0] for x in v}) > 1}
    return dupes, leaked


def main() -> int:
    args = sys.argv[1:]
    gens, plat, json_out, verbose = "build/.gens", "windows", None, False
    if "--gens" in args:
        gens = args[args.index("--gens") + 1]
    if "--plat" in args:
        plat = args[args.index("--plat") + 1]
    if "--json" in args:
        json_out = args[args.index("--json") + 1]
    if "--verbose" in args:
        verbose = True

    files = tracked_cpp()
    targets = collections.defaultdict(list)
    for f in files:
        tgt, batch = target_of(f)
        if (batch or 0) > 1:
            targets[tgt].append(f)
    if not targets:
        print("check_unity_build_conflicts: no unity-built target found", file=sys.stderr)
        return 2

    problems = []
    macro_problems = []
    # 1. exact generated batches
    batches = {} if "--no-gens" in args else parse_unity_batches(gens, plat)
    units = collections.defaultdict(list)
    for f, (tgt, unit) in batches.items():
        units[(tgt, unit)].append(f)
    for (tgt, unit), unit_files in sorted(units.items()):
        syms = collect(unit_files)
        for (scope, name), v in sorted(syms.items()):
            if len({x[0] for x in v}) > 1:
                problems.append({"check": "batch", "target": tgt, "unit": unit,
                                 "scope": scope, "name": name,
                                 "files": sorted(f"{f}:{it['line']}" for f, it in v)})
        macro_dupes, macro_leaks = collect_macros(unit_files)
        for name, v in sorted(macro_dupes.items()):
            problems.append({"check": "macro-batch", "target": tgt, "unit": unit,
                             "scope": "#define", "name": name, "files": sorted(v)})
        for name, v in sorted(macro_leaks.items()):
            macro_problems.append({"check": "macro-leak", "target": tgt, "unit": unit,
                                   "scope": "#define", "name": name, "files": sorted(v)})
    if batches:
        print(f"checked {len(units)} generated unity batches "
              f"({len(batches)} translation units)")
    else:
        print("no build/.gens found - only the conservative target-wide check runs")

    # 2. conservative target-wide check
    for tgt, tgt_files in sorted(targets.items()):
        syms = collect(tgt_files)
        for (scope, name), v in sorted(syms.items()):
            if len({x[0] for x in v}) > 1:
                problems.append({"check": "target", "target": tgt, "unit": None,
                                 "scope": scope, "name": name,
                                 "files": sorted(f"{f}:{it['line']}" for f, it in v)})

    if json_out:
        json.dump({"conflicts": problems, "macro_leaks": macro_problems},
                  open(json_out, "w", encoding="utf-8"), indent=2)

    if problems:
        print(f"\n{len(problems)} unity-build name collision(s):\n")
        for p in problems:
            where = f"{p['target']}/{p['unit']}" if p["unit"] else p["target"]
            print(f"  [{p['check']}] {where}: {p['scope']}::{p['name']}")
            for f in p["files"]:
                print(f"        {f}")
        print("\nMove each file-local helper into a file-unique namespace, e.g.\n"
              "    namespace {  ->  namespace { namespace <file>_detail {\n"
              "and qualify its uses as <file>_detail::name (see\n"
              "scripts/unity_isolate_locals.py).")
        return 1
    print("unity-build check: no conflicting file-local symbols")
    if macro_problems:
        print(f"({len(macro_problems)} macro(s) are defined without a matching #undef and "
              f"therefore leak into the rest of their unity batch - informational)")
    if verbose:
        for tgt, tgt_files in sorted(targets.items()):
            print(f"  {tgt}: {len(tgt_files)} translation units, "
                  f"{len(collect(tgt_files))} internal-linkage names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
