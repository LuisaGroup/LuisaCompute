#!/usr/bin/env python3
"""Isolate a translation unit's file-local helpers into a file-unique namespace so
that unity (jumbo) builds can never merge them with a sibling translation unit's
helpers.

A unity build concatenates several translation units of one target into a single
translation unit.  Every name a file declares at namespace scope with internal
linkage (`static`, or anything inside an anonymous namespace) then shares its
scope with an identically named declaration of every other file of the batch,
which either fails to compile (redefinition) or silently changes overload
resolution.

For a file with tag `T`, every entity whose named-namespace scope is shared with
sibling files of the same target is moved into an extra scope named `T_detail`
(`T_detail_2`, ... for further isolation blocks in other scopes of the same file):

    namespace A {                       namespace A {
    namespace {                         namespace {
    helper declarations          ==>    namespace T_detail {
    }                                     ...            (unchanged)
    body                                }  // namespace T_detail
    using `helper`                      }// namespace
    }                                   body using `T_detail::helper`
                                        }

* an entity inside an anonymous namespace keeps that namespace (so it keeps
  internal linkage) and gains the unique `T_detail` scope - by default the whole
  anonymous block is moved, which also covers declarations the scanner cannot
  classify;
* `--minimal` moves only the requested entities, wrapping each in place;
* a `static` at namespace scope (or an entity in a shared *named* namespace like
  `detail`) gets a small `namespace T_detail { ... }` wrapper in place at its own
  scope, which preserves linkage as well as declaration order.

Because the extra scope carries the file stem it is unique inside the project, so
no two translation units can contribute the same name to one scope, whatever the
unity batching is.  A file that already contains `T_detail` namespaces is handled
by reusing the one of the same scope and otherwise picking the next free suffix.
Names that are `#define`d in the file or used inside a preprocessor directive are
never touched (a macro would expand inside a qualified name, and `#if NAME` cannot
see a namespace member) - such cases are reported instead.

Usage:
  python scripts/unity_isolate_locals.py --tag NAME --names a,b,c [--minimal] [--dry-run] <file.cpp>...
"""
from __future__ import annotations

import os
import re
import sys

import tree_sitter_cpp
from tree_sitter import Language, Parser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unity_symbol_scan import declarator_name, is_qualified_name  # noqa: E402

CPP = Language(tree_sitter_cpp.language())
PARSER = Parser(CPP)

IDENT_TYPES = ("identifier", "type_identifier", "field_identifier", "namespace_identifier",
               "destructor_name", "operator_name")
DECL_TYPES = ("declaration", "function_definition", "class_specifier", "struct_specifier",
              "union_specifier", "enum_specifier", "type_definition", "alias_declaration")
PREPROC = ("preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif", "preproc_ifndef",
           "linkage_specification")


class Doc:
    def __init__(self, path: str):
        self.path = path
        self.raw = open(path, "rb").read()
        n_crlf, n_lf = self.raw.count(b"\r\n"), self.raw.count(b"\n")
        self.eol = b"\r\n" if n_crlf * 2 > n_lf else b"\n"
        self.tree = PARSER.parse(self.raw)

    def text(self, node) -> str:
        return self.raw[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def nodes(self, types=None):
        stack = [self.tree.root_node]
        while stack:
            n = stack.pop()
            if types is None or n.type in types:
                yield n
            stack.extend(n.children)


def declared_names(node, doc: Doc):
    """The identifier a declaration introduces (never its type)."""
    out = []
    for field in ("declarator", "name"):
        d = node.child_by_field_name(field)
        if d is None:
            continue
        nm = declarator_name(d, doc.raw)
        if nm:
            out.append(nm)
            break
    if not out and node.type in ("type_definition", "alias_declaration", "enum_specifier"):
        for ch in node.children:
            if ch.type in IDENT_TYPES:
                out.append(doc.text(ch))
                break
    return out


def is_class_member(node) -> bool:
    """True when a declaration introduces a class member, not a file-local entity.

    A constructor/member declared inside a class body must never be moved into a
    namespace: the wrapper would cut the class definition in half.
    """
    p = node.parent
    while p is not None:
        if p.type == "field_declaration_list":
            return True
        if p.type in ("translation_unit", "namespace_definition", "declaration_list",
                      "linkage_specification", "template_declaration", "preproc_ifdef",
                      "preproc_if", "preproc_else", "preproc_elif", "preproc_ifndef"):
            return False
        p = p.parent
    return False


def collect_block_names(doc: Doc, node) -> set:
    """Every name declared inside an anonymous-namespace body (it moves as a whole)."""
    out = set()
    for ch in node.children:
        if ch.type == "namespace_definition":
            nm = ch.child_by_field_name("name")
            body = ch.child_by_field_name("body")
            if nm is not None:
                out.add(doc.text(nm))
            if body is not None:
                out |= collect_block_names(doc, body)
            continue
        if ch.type in PREPROC or ch.type == "template_declaration":
            out |= collect_block_names(doc, ch)
            continue
        if ch.type in DECL_TYPES:
            out |= set(declared_names(ch, doc))
    return out


def preproc_collisions(doc: Doc, names: set) -> set:
    """Names that must not be qualified because of the preprocessor.

    Unsafe when the file `#define`s the name (a macro expands even inside a
    qualified name) or when the name appears in a preprocessor directive/argument
    (`#if NAME` cannot see a namespace member).
    """
    unsafe = set()
    src = doc.raw.decode("utf-8", "replace")
    for n in names:
        if re.search(rf"^[ \t]*#[ \t]*(define|undef)[ \t]+{re.escape(n)}\b", src, re.M):
            unsafe.add(n)
    for node in doc.nodes({"preproc_arg", "preproc_directive"}):
        text = doc.text(node)
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", text):
                unsafe.add(n)
    return unsafe


class Analyzer:
    def __init__(self, doc: Doc):
        self.doc = doc
        self.path = {}
        self.anon_of = {}
        self.detail_ns = []   # (scope path, name) of existing `<tag>_detail*`
        self._walk(doc.tree.root_node, (), None)

    def _walk(self, node, path, anon):
        for ch in node.children:
            if ch.type == "namespace_definition":
                # the enclosing path of the namespace node itself (used to match an
                # already isolated block with the block that is being created now)
                self.path[ch.id] = path
                nm = ch.child_by_field_name("name")
                body = ch.child_by_field_name("body")
                if nm is None:
                    if body is not None:
                        self.path[body.id] = path
                        self._walk(body, path, ch)
                elif body is not None:
                    sub = path + (self.doc.text(nm),)
                    self.path[body.id] = sub
                    self._walk(body, sub, anon)
                continue
            self.path[ch.id] = path
            self.anon_of[ch.id] = anon
            self._walk(ch, path, anon)

    def entities(self, names: set):
        found = []
        for node in self.doc.nodes(DECL_TYPES):
            if is_class_member(node):
                continue
            parent = node.parent
            # `struct X {...} y;` -> the enclosing declaration is the entity
            if parent is not None and parent.type == "declaration" and \
                    node.type in ("class_specifier", "struct_specifier", "enum_specifier",
                                  "union_specifier"):
                continue
            if is_qualified_name(node):
                continue
            for nm in declared_names(node, self.doc):
                if nm in names:
                    found.append((nm, node))
                    break
        return found

    def wrap_anchor(self, node):
        while node.parent is not None and node.parent.type == "template_declaration":
            node = node.parent
        return node


def qualify(use_path, wrap_path, ns_name) -> str:
    use_path, wrap_path = tuple(use_path), tuple(wrap_path)
    common = 0
    while common < len(use_path) and common < len(wrap_path) and \
            use_path[common] == wrap_path[common]:
        common += 1
    if common == 0:
        return "::" + "::".join(wrap_path + (ns_name,)) + "::"
    return "::".join(wrap_path[common:] + (ns_name,)) + "::"


def resolve_prefix(doc: Doc, az: Analyzer, scope, wrap_path) -> bool:
    """Does the namespace prefix of a qualified id name the isolated scope?"""
    text = doc.text(scope)
    segs = tuple(x for x in text.split("::") if x)
    if not segs:
        return False
    if text.startswith("::"):
        return segs == tuple(wrap_path)
    path = tuple(az.path.get(scope.id, ()))
    for k in range(len(path), -1, -1):
        if path[:k] + segs == tuple(wrap_path):
            return True
    return False


def inside_detail(az: Analyzer, node, tag) -> bool:
    return any(re.fullmatch(rf"{re.escape(tag)}_detail(_\d+)?", seg)
               for seg in az.path.get(node.id, ()))


def main() -> int:
    args = sys.argv[1:]
    tag = names_arg = None
    dry = minimal = force = False
    files = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tag":
            tag, i = args[i + 1], i + 2
        elif a == "--names":
            names_arg, i = set(args[i + 1].split(",")), i + 2
        elif a == "--dry-run":
            dry, i = True, i + 1
        elif a == "--minimal":
            minimal, i = True, i + 1
        elif a == "--force":
            force, i = True, i + 1
        else:
            files.append(a)
            i += 1
    if not tag or names_arg is None:
        print("--tag and --names are required", file=sys.stderr)
        return 2

    for f in files:
        doc = Doc(f)
        az = Analyzer(doc)
        ents = az.entities(names_arg)
        found_names = {n for n, _ in ents}
        missing = names_arg - found_names
        print(f"== {f}: tag={tag} entities={len(ents)}"
              + (f"  MISSING={sorted(missing)}" if missing else ""))
        if not ents:
            continue

        # --- names that must not be qualified because of the preprocessor ----
        unsafe = preproc_collisions(doc, found_names)
        if unsafe:
            print(f"!! {f}: {sorted(unsafe)} are #define'd here or used in a preprocessor "
                  f"directive; left alone", file=sys.stderr)
            ents = [(n, node) for n, node in ents if n not in unsafe]
            names_arg -= unsafe
            found_names -= unsafe
        if not ents:
            continue
        if not force and all(inside_detail(az, node, tag) for _, node in ents):
            print("    already isolated, skipping")
            continue

        # --- group the entities into isolation blocks -----------------------
        blocks = []
        inplace = []
        seen_anon = {}
        for nm, node in ents:
            anon = None if minimal else az.anon_of.get(node.id)
            if anon is not None:
                seen_anon.setdefault(anon.id, (anon, []))[1].append((nm, node))
            else:
                inplace.append((nm, az.wrap_anchor(node)))
        for anon, group in seen_anon.values():
            body = anon.child_by_field_name("body")
            if body is None:
                continue
            expanded = collect_block_names(doc, body) - unsafe
            found_names |= expanded
            blocks.append({"start": body.start_byte + 1, "end": body.end_byte - 1,
                           "indent": b" " * anon.start_point[1], "kind": "anon",
                           "path": az.path.get(body.id, ()), "ents": group,
                           "names": expanded})
        by_parent = {}
        for nm, node in inplace:
            by_parent.setdefault(node.parent.id, (node.parent, []))[1].append((nm, node))
        for parent, items in by_parent.values():
            index = {ch.id: k for k, ch in enumerate(parent.children)}
            items.sort(key=lambda e: index.get(e[1].id, 1 << 30))
            run, prev = [], None
            for nm, node in items:
                idx = index.get(node.id)
                if idx is None:
                    continue
                if prev is not None and idx != prev + 1:
                    flush(doc, az, blocks, run)
                    run = []
                run.append((nm, node))
                prev = idx
            if run:
                flush(doc, az, blocks, run)

        # --- assign one namespace name per (scope, block kind) --------------
        existing = {}
        for ns in doc.nodes({"namespace_definition"}):
            nm = ns.child_by_field_name("name")
            if nm is None:
                continue
            name = doc.text(nm)
            if not re.fullmatch(rf"{re.escape(tag)}_detail(_\d+)?", name):
                continue
            par = ns.parent
            in_anon = (par is not None and par.type == "declaration_list" and
                       par.parent is not None and
                       par.parent.type == "namespace_definition" and
                       par.parent.child_by_field_name("name") is None)
            existing[(az.path.get(ns.id, ()), "anon" if in_anon else "run")] = name
        names_by_key, used = {}, set(existing.values())
        for b in blocks:
            key = (b["path"], b["kind"])
            if key in names_by_key:
                continue
            # reuse an existing namespace only when it was created for the same
            # scope *and* block kind: two different scopes must never share the
            # name, or every reference to it becomes ambiguous
            if key in existing:
                names_by_key[key] = existing[key]
            if key not in names_by_key:
                k = 1
                while True:
                    cand = f"{tag}_detail" if k == 1 else f"{tag}_detail_{k}"
                    if cand not in used:
                        break
                    k += 1
                used.add(cand)
                names_by_key[key] = cand

        # --- build the wrapper edits ---------------------------------------
        edits, regions, name_of_name = [], [], {}
        for b in blocks:
            ns_name = names_by_key[(b["path"], b["kind"])]
            indent = b["indent"]
            if b["kind"] == "anon":
                edits.append((b["start"], b["start"],
                              doc.eol + indent + f"namespace {ns_name} {{".encode(),
                              f"open {ns_name}"))
                edits.append((b["end"], b["end"],
                              indent + f"}}  // namespace {ns_name}".encode() + doc.eol + indent,
                              f"close {ns_name}"))
            else:
                edits.append((b["start"], b["start"],
                              f"namespace {ns_name} {{".encode() + doc.eol, f"open {ns_name}"))
                edits.append((b["end"], b["end"],
                              f"}}  // namespace {ns_name}".encode() + doc.eol, f"close {ns_name}"))
            regions.append((b["start"], b["end"], b["path"], ns_name))
            for nm in b.get("names", set()) | {n for n, _ in b["ents"]}:
                name_of_name[nm] = (b["path"], ns_name)
            for nm, node in b["ents"]:
                if not (b["start"] <= node.start_byte <= b["end"]):
                    print(f"!! {f}: {nm} L{node.start_point[0] + 1} is outside its block",
                          file=sys.stderr)
            print(f"    block {ns_name:<30} scope={'::'.join(b['path']) or '::':<30} "
                  f"entities={len(b['ents'])}")

        unwrapped = [nm for nm, node in inplace
                     if not any(s <= node.start_byte <= e for s, e, _, _ in regions)]
        if unwrapped:
            print(f"!! {f}: not wrapped: {unwrapped}", file=sys.stderr)

        # --- function-local declarations shadowing an entity ----------------
        shadowed = set()
        for sub in doc.nodes({"init_declarator", "parameter_declaration", "declaration"}):
            d = sub.child_by_field_name("declarator")
            if d is None:
                continue
            dn = declarator_name(d, doc.raw)
            if dn not in found_names:
                continue
            anc = sub.parent
            while anc is not None and anc.type not in (
                    "function_definition", "translation_unit", "namespace_definition",
                    "declaration_list"):
                anc = anc.parent
            if anc is not None and anc.type == "function_definition":
                shadowed.add((dn, anc.id))
        if shadowed:
            print(f"!! {f}: {sorted({n for n, _ in shadowed})} are also declared inside a "
                  f"function; those uses are left untouched", file=sys.stderr)

        # --- rewrite every use ---------------------------------------------
        uses = 0
        for nm in sorted(found_names):
            if nm not in name_of_name:
                print(f"!! {f}: {nm} has no isolation block", file=sys.stderr)
                continue
            wrap_path, ns_name = name_of_name[nm]
            for node in doc.nodes(IDENT_TYPES):
                if doc.text(node) != nm:
                    continue
                parent = node.parent
                # `Prefix::name`: the prefix must resolve to the isolated scope for
                # the reference to denote the moved entity.  This also occurs *inside*
                # a wrapped block, where a sibling is reached through the outer scope.
                if (parent is not None and
                        parent.type in ("qualified_identifier", "scoped_type_identifier") and
                        parent.child_by_field_name("name") is not None and
                        parent.child_by_field_name("name").id == node.id):
                    scope = parent.child_by_field_name("scope")
                    if scope is None:
                        continue
                    if resolve_prefix(doc, az, scope, wrap_path):
                        edits.append((node.start_byte, node.start_byte,
                                      f"{ns_name}::".encode(), f"qualify {nm} (scoped)"))
                        uses += 1
                    else:
                        print(f"!! {f}: qualified use L{node.start_point[0] + 1} "
                              f"'{doc.text(parent)}' does not resolve to the isolated scope; "
                              f"left untouched", file=sys.stderr)
                    continue
                if any(s <= node.start_byte <= e for s, e, _, _ in regions):
                    continue  # unqualified use of a sibling inside the isolated block
                if parent is not None:
                    if parent.type == "field_expression":
                        fld = parent.child_by_field_name("field")
                        if fld is not None and fld.id == node.id:
                            continue
                    if parent.type == "namespace_definition":
                        continue
                    if parent.type == "nested_namespace_specifier":
                        if (parent.child_by_field_name("name") is not None and
                                parent.child_by_field_name("name").id == node.id):
                            continue
                anc = node.parent
                while anc is not None and anc.type != "function_definition":
                    anc = anc.parent
                if anc is not None and (nm, anc.id) in shadowed:
                    continue
                use_path = tuple(az.path.get(node.id, ()))
                edits.append((node.start_byte, node.end_byte,
                              f"{qualify(use_path, wrap_path, ns_name)}{nm}".encode(),
                              f"qualify {nm}"))
                uses += 1
        print(f"    edits={len(edits)} uses={uses} blocks={len(blocks)}")
        if dry:
            continue

        edits.sort(key=lambda e: (e[0], e[1]))
        out, pos = bytearray(), 0
        for start, end, rep, note in edits:
            if start < pos:
                print(f"!! {f}: overlapping edit at byte {start} ({note})", file=sys.stderr)
                return 1
            out += doc.raw[pos:start] + rep
            pos = end
        out += doc.raw[pos:]
        open(f, "wb").write(bytes(out))
    return 0


def flush(doc: Doc, az: Analyzer, blocks, run):
    """Append one contiguous run of declarations that is wrapped in place."""
    run.sort(key=lambda e: e[1].start_byte)
    first, last = run[0][1], run[-1][1]
    end = last.end_byte
    # a class/struct/enum specifier node stops at `}` - take the `;` with it
    j = end
    while j < len(doc.raw) and doc.raw[j : j + 1] in (b" ", b"\t", b"\r", b"\n"):
        j += 1
    if j < len(doc.raw) and doc.raw[j : j + 1] == b";":
        end = j + 1
    while end < len(doc.raw) and doc.raw[end : end + 1] in (b"\r", b"\n"):
        end += 1
    blocks.append({"start": first.start_byte, "end": end,
                   "indent": b" " * first.start_point[1], "kind": "run",
                   "path": az.path.get(first.id, ()), "ents": run, "names": set()})


if __name__ == "__main__":
    raise SystemExit(main())
