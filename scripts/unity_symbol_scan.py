#!/usr/bin/env python3
"""Unity-build (jumbo) symbol scanner for LuisaCompute.

A "unity build" concatenates several translation units of one target into a
single translation unit.  Any name a file declares at namespace scope with
internal linkage (`static` or inside an anonymous namespace) then shares its
scope with the identically named declarations of every other file in the batch,
which either fails to compile (redefinition) or - worse - silently changes
overload resolution.

This module provides the scanning primitives used by

  * scripts/check_unity_build_conflicts.py  (build guard, fails on a collision)
  * scripts/unity_risk_report.py            (lists file-local entities whose
                                             scope is shared with sibling files)
  * scripts/unity_isolate_locals.py         (moves such entities into a
                                             file-unique `T_detail` namespace)

Reported kinds
--------------
  static_function / static_decl   `static` at namespace scope
  anon_function / anon_decl / anon_class
                                  declared inside an anonymous namespace
  member_definition               `Class::method` definition (never a collision)
  extern_function                 plain free function (linkage, not unity)

Usage:
  python scripts/unity_symbol_scan.py [--json out.json] <file.cpp>...
"""
from __future__ import annotations

import json
import os
import re
import sys

import tree_sitter_cpp
from tree_sitter import Language, Parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CPP = Language(tree_sitter_cpp.language())
PARSER = Parser(CPP)

SKIP_DIRS = "declaration_list"

# Which xmake target compiles a source file, and its unity batch size.
# batch == 0/None means the target does not use a unity build at all.
TARGET_RULES = [
    ("src/core/", "lc-core", 4),
    ("src/vstl/", "lc-vstl", 4),
    ("src/tile/", "lc-tile", 4),
    ("src/runtime/", "lc-runtime", 8),
    ("src/ast/", "lc-runtime", 8),
    ("src/xir/", "lc-runtime", 8),
    ("src/osl/", "lc-osl", 16),
    ("src/backends/vk/", "lc-backend-vk", 8),
    ("src/backends/dx/", "lc-backend-dx", 8),
    ("src/backends/cuda/", "lc-backend-cuda", 4),
    ("src/backends/common/hlsl/", "lc-hlsl-codegen", 2),
    ("src/backends/common/spirv/", "lc-spirv", 2),
    ("src/backends/common/rtx/", "luisa-fallback-rtx", 2),
    ("src/backends/common/", "lc-backends-common", None),
    ("src/backends/fallback/", "lc-fallback", 8),
    ("src/backends/metal/", "lc-backend-metal", 0),
    ("src/backends/metal4/", "lc-backend-metal4", 0),
    ("src/backends/simd/", "lc-backend-simd", None),
    ("src/backends/validation/", "lc-validation-layer", None),
    ("src/backends/tools/", "lc-backends-tools", None),
    ("src/gui/", "lc-gui", None),
    ("src/dsl/", "lc-dsl", 0),
    ("src/coro/", "lc-coro", 0),
    ("src/clangcxx/", "lc-clangcxx", None),
    ("src/py/", "lcapi", None),
    ("src/tests/", "tests (no unity build)", 0),
    ("examples/", "examples (no unity build)", 0),
    ("tutorials/", "tutorials (no unity build)", 0),
]

# Entity kinds that share the enclosing scope with sibling translation units.
INTERNAL_KINDS = ("static_function", "static_decl", "anon_function", "anon_decl",
                  "anon_class")

IDENT_TYPES = ("identifier", "type_identifier", "field_identifier", "namespace_identifier",
               "destructor_name", "operator_name")


def target_of(path: str):
    p = path.replace("\\", "/")
    for prefix, tgt, batch in TARGET_RULES:
        if p.startswith(prefix):
            return tgt, batch
    return "?", None


def text_of(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def is_qualified_name(node) -> bool:
    """True when a declaration defines a member/namespace-scoped entity (`A::b`).

    Only the declarator chain that introduces the *name* is inspected: a
    qualified *type* (return type or parameter type, e.g. `void f(tvm::PrimExpr)`)
    must not mark the declaration as a member definition, otherwise such helpers
    would be invisible to the unity-build checks.
    """
    d = node.child_by_field_name("declarator")
    while d is not None:
        if d.type in ("qualified_identifier", "scoped_type_identifier", "template_method"):
            return True
        inner = d.child_by_field_name("declarator")
        if inner is None or inner.id == d.id:
            break
        d = inner
    return False


def declarator_name(node, src: bytes):
    """Walk a declarator chain and return the innermost identifier."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in ("identifier", "field_identifier", "type_identifier",
                      "destructor_name", "operator_name"):
            return text_of(n, src)
        for ch in reversed(n.children):
            if ch.type in ("identifier", "qualified_identifier", "pointer_declarator",
                           "reference_declarator", "function_declarator", "array_declarator",
                           "parenthesized_declarator", "init_declarator", "destructor_name",
                           "operator_name"):
                stack.append(ch)
    return None


def declared_name(node, src: bytes):
    for field in ("declarator", "type"):
        d = node.child_by_field_name(field)
        if d is not None:
            nm = declarator_name(d, src)
            if nm:
                return nm
    return None


def _walk_scope(node, src: bytes, scope_path, out, in_anon):
    for ch in node.children:
        t = ch.type
        if t == "namespace_definition":
            nm_node = ch.child_by_field_name("name")
            body = ch.child_by_field_name("body")
            if nm_node is None:
                out.append({"kind": "anonymous_namespace", "name": f"anon@{ch.start_point[0] + 1}",
                            "line": ch.start_point[0] + 1,
                            "scope": "::".join(scope_path) or "::",
                            "snippet": text_of(ch, src).splitlines()[0][:100]})
                if body is not None:
                    _walk_scope(body, src, scope_path, out, True)
            else:
                if body is not None:
                    _walk_scope(body, src, scope_path + [text_of(nm_node, src)], out, in_anon)
            continue
        if t in ("linkage_specification", "preproc_ifdef", "preproc_if", "preproc_else",
                 "preproc_elif", "preproc_ifndef", "template_declaration"):
            body = ch.child_by_field_name("body")
            _walk_scope(body if body is not None else ch, src, scope_path, out, in_anon)
            continue
        if t in ("class_specifier", "struct_specifier", "union_specifier", "enum_specifier"):
            if in_anon:
                nm = ch.child_by_field_name("name")
                out.append({"kind": "anon_class",
                            "name": text_of(nm, src) if nm else "<anonymous>",
                            "line": ch.start_point[0] + 1,
                            "scope": "::".join(scope_path) or "::",
                            "snippet": text_of(ch, src).splitlines()[0][:100]})
            continue
        if t in ("function_definition", "declaration"):
            static = any(c.type == "storage_class_specifier" and
                         text_of(c, src).strip() == "static" for c in ch.children)
            if is_qualified_name(ch):
                out.append({"kind": "member_definition", "name": declared_name(ch, src) or "?",
                            "line": ch.start_point[0] + 1,
                            "scope": "::".join(scope_path) or "::",
                            "snippet": " ".join(text_of(ch, src).split())[:140]})
                continue
            nm = declared_name(ch, src) or "?"
            if static:
                kind = "static_function" if t == "function_definition" else "static_decl"
            elif in_anon:
                kind = "anon_function" if t == "function_definition" else "anon_decl"
            else:
                kind = "extern_function" if t == "function_definition" else "other"
            out.append({"kind": kind, "name": nm, "line": ch.start_point[0] + 1,
                        "scope": "::".join(scope_path) or "::",
                        "snippet": " ".join(text_of(ch, src).split())[:140]})


def scan(path: str):
    """All namespace-scope declarations of one translation unit."""
    src = open(path, "rb").read()
    tree = PARSER.parse(src)
    out = []
    _walk_scope(tree.root_node, src, [], out, False)
    return out


def parse_unity_batches(gens="build/.gens", plat="windows"):
    """Source file -> (target, generated unity unit), from xmake's own batches.

    xmake only rewrites these files when the target's source set changes, so they
    are the ground truth for the batching that is actually compiled.
    """
    out = {}
    # accept either the build directory (xmake writes the generated units to
    # <builddir>/.gens/<target>/<plat>/unity_build/unity_N.cpp) or that .gens
    # directory itself
    candidates = [os.path.join(ROOT, gens), os.path.join(ROOT, gens, ".gens")]
    base = None
    for c in candidates:
        if not os.path.isdir(c):
            continue
        if any(os.path.isdir(os.path.join(c, t, plat, "unity_build"))
               for t in os.listdir(c)):
            base = c
            break
    if base is None:
        return out
    for tgt in sorted(os.listdir(base)):
        d = os.path.join(base, tgt, plat, "unity_build")
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
                out[p.replace(ROOT.replace("\\", "/") + "/", "")] = (tgt, name)
    return out


def tracked_cpp():
    import subprocess

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=ROOT).stdout.split()
    return [f for f in files if f.endswith(".cpp")]


def main() -> int:
    args = sys.argv[1:]
    out = None
    if args and args[0] == "--json":
        out, args = args[1], args[2:]
    result = {}
    for p in args:
        p = p.strip()
        if not p:
            continue
        try:
            result[p] = scan(os.path.join(ROOT, p))
        except OSError as e:
            print(f"!! cannot read {p}: {e}", file=sys.stderr)
            continue
    if out:
        json.dump(result, open(out, "w", encoding="utf-8"), indent=2)
    for p, items in sorted(result.items()):
        internal = [it for it in items if it["kind"] in INTERNAL_KINDS]
        if not internal:
            continue
        print(f"== {p}")
        for it in internal:
            print(f"  L{it['line']:>5} {it['kind']:<14} {it['scope']}::{it['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def scan_macros(path: str):
    """(defined, undefined) macro names, parsed from the preprocessor nodes.

    Raw string literals and comments cannot produce false positives because the
    scan walks tree-sitter's `preproc_def` / `preproc_function_def` nodes.
    """
    src = open(path, "rb").read()
    tree = PARSER.parse(src)
    defined, undefd = {}, set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in ("preproc_def", "preproc_function_def"):
            name = n.child_by_field_name("name")
            if name is None:
                for ch in n.children:
                    if ch.type == "identifier":
                        name = ch
                        break
            if name is not None:
                defined[text_of(name, src)] = n.start_point[0] + 1
        elif n.type == "preproc_call":
            directive = next((c for c in n.children if c.type == "preproc_directive"), None)
            if directive is not None and text_of(directive, src).strip() == "#undef":
                arg = next((c for c in n.children if c.type == "preproc_arg"), None)
                if arg is not None:
                    undefd.add(text_of(arg, src).strip())
        stack.extend(n.children)
    return defined, undefd
