#!/usr/bin/env python3
"""Inventory of file-local (internal-linkage) C++ entities and macros.

This is the scanning library behind ``check_unity_build_conflicts.py`` (and the
other unity-build helpers).  A unity (jumbo) build concatenates several
translation units of one target into a single translation unit, so two files
that both declare e.g. ``namespace { bool is_zero(..) }`` or
``static void clone_metadata(..)`` at the same namespace scope either fail to
compile (redefinition) or silently change overload resolution.  Finding those
names needs a real parse of every file, which is what this module provides.

Public API
----------
``ROOT``                     repository root (absolute path string).
``INTERNAL_KINDS``           the item kinds that carry internal linkage.
``tracked_cpp()``            every ``src/**/*.cpp`` file, repo-relative, POSIX
                            separators, vendored ``src/ext`` excluded.
``target_of(f)``             ``(xmake target, unity batch size)`` of one file,
                            read from the nearest ``xmake.lua``.
``scan(path)``               internal-linkage entities declared at namespace
                            scope, as ``{"kind", "scope", "name", "line"}``.
``scan_macros(path)``        ``({name: line}, {name: undef_line})`` of the
                            ``#define``/``#undef`` directives of one file.
``parse_unity_batches``      ``{file: (target, unit)}`` read back from the
                            generated ``unity_build/unity_N.cpp`` files.

Item kinds
----------
``anonymous-namespace``  declaration inside ``namespace { ... }``
``static``               namespace-scope ``static`` function or variable
``internal-const``       namespace-scope ``const``/``constexpr`` variable

Only *definitions* are reported for functions: a prototype in one file and the
definition in another are legal in one translation unit, so reporting
declarations would produce false positives.  Class/enum names are reported when
the type is defined (a forward declaration is not a definition and cannot
collide).  Class members, function-local entities and ``using`` aliases are not
namespace-scope names and are never reported.

Usage (debugging aid):
    python scripts/unity_symbol_scan.py src/backends/vk/stream.cpp
    python scripts/unity_symbol_scan.py --batches build/.gens --plat windows
"""
from __future__ import annotations

import fnmatch
import glob
import os
import re
import sys

try:
    import tree_sitter_cpp
    from tree_sitter import Language, Parser
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "unity_symbol_scan: tree-sitter is required "
        "(pip install tree-sitter tree-sitter-cpp): " + str(exc)
    ) from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kinds produced by `scan()` that have internal linkage and therefore collide
# when two files of one unity batch are merged.
INTERNAL_KINDS = frozenset({
    "anonymous-namespace",
    "static",
    "internal-const",
})

_LANGUAGE = Language(tree_sitter_cpp.language())
_PARSER = Parser(_LANGUAGE)

# Namespace-scope declarations that are *not* entities (they introduce no name
# that another file could collide with at the same scope).
_IGNORED_DECLARATIONS = frozenset({
    "static_assert_declaration",
    "preproc_include",
    "preproc_def",
    "preproc_function_def",
    "preproc_call",
    "preproc_if",
    "preproc_ifdef",
    "preproc_else",
    "preproc_elif",
    "using_declaration",
    "using_directive",
    "namespace_alias_definition",
})

_TYPE_SPECIFIERS = ("struct_specifier", "class_specifier",
                    "union_specifier", "enum_specifier")

# Names that only ever denote a declaration's own declarator chain.
_DECLARATOR_WRAPPERS = ("init_declarator", "pointer_declarator",
                        "reference_declarator", "array_declarator",
                        "function_declarator", "parenthesized_declarator",
                        "attributed_declarator")


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def _text(node) -> str:
    return node.text.decode("utf-8", "replace")


def _has_body(node) -> bool:
    """True when a class/enum specifier is a definition, not a forward decl."""
    for field in ("body",):
        if node.child_by_field_name(field) is not None:
            return True
    return any(child.type in ("field_declaration_list",
                             "enumerator_list") for child in node.children)


def _declarator_name(node):
    """Innermost identifier of a declarator chain, or None."""
    if node is None:
        return None
    stack = [node]
    while stack:
        current = stack.pop(0)
        if current.type in ("identifier", "field_identifier"):
            return _text(current)
        if current.type in ("operator_name", "destructor_name",
                            "qualified_identifier"):
            # Operator overloads and out-of-line definitions are not
            # file-local helpers in practice; the qualified form belongs to a
            # class, not to this scope.
            if current.type == "qualified_identifier":
                return None
            return _text(current)
        stack = [c for c in current.children if c.type in _DECLARATOR_WRAPPERS
                 or c.type in ("identifier", "field_identifier",
                               "operator_name", "destructor_name",
                               "qualified_identifier")] + stack
    return None


def _declared_names(node):
    """Names introduced by one namespace-scope declaration node."""
    kind = node.type
    if kind == "function_definition":
        name = _declarator_name(node.child_by_field_name("declarator"))
        return [(name, "definition")] if name else []
    if kind == "template_declaration":
        names = []
        for child in node.named_children:
            if child.type in ("declaration", "function_definition",
                              "class_specifier", "struct_specifier",
                              "alias_declaration"):
                names.extend(_declared_names(child))
        return names
    if kind == "class_specifier" or kind == "struct_specifier" or \
            kind == "union_specifier":
        if not _has_body(node):
            return []  # forward declaration
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return [(_text(name_node), "definition")]
        # Anonymous class in an anonymous namespace: nothing nameable.
        return []
    if kind == "enum_specifier":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            if not _has_body(node):
                return []
            return [(_text(name_node), "definition")]
        if not _has_body(node):
            return []
        # Anonymous enum: its enumerators are the names that can collide.
        names = []
        for child in node.named_children:
            if child.type == "enumerator_list":
                for enumerator in child.named_children:
                    if enumerator.type == "enumerator":
                        enumerator_name = enumerator.child_by_field_name("name")
                        if enumerator_name is not None:
                            names.append((_text(enumerator_name), "definition"))
        return names
    if kind == "alias_declaration":
        # `using X = ...;` is a type alias: two identical aliases in one
        # translation unit are legal, so it never collides.
        return []
    if kind != "declaration":
        return []

    names = []
    storage = None
    qualifiers = set()
    for child in node.children:
        if child.type == "storage_class_specifier":
            storage = _text(child)
        elif child.type == "type_qualifier":
            qualifiers.add(_text(child))
        elif child.type in _TYPE_SPECIFIERS:
            names.extend(_declared_names(child))
        elif child.type == "function_declarator":
            name = _declarator_name(child)
            if name:
                names.append((name, "declaration"))
        elif child.type == "init_declarator":
            name = _declarator_name(child)
            if name:
                names.append((name, "definition"))
        elif child.type == "identifier":
            # `static int f;` has no init_declarator.
            names.append((_text(child), "declaration"))
    # A declaration with a body-less class specifier contributes nothing above.
    return [n for n in names if n[0]]


def _walk_namespace(node, scope, inside_anonymous, out):
    for child in node.named_children:
        kind = child.type
        if kind == "namespace_definition":
            name_node = child.child_by_field_name("name")
            body = child.child_by_field_name("body")
            if body is None:
                continue
            if name_node is None:
                # anonymous namespace: everything inside has internal linkage
                _walk_namespace(body, scope, True, out)
            else:
                _walk_namespace(body, scope + [_text(name_node)],
                                inside_anonymous, out)
            continue
        if kind == "linkage_specification":
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_namespace(body, scope, inside_anonymous, out)
            continue
        if kind in _IGNORED_DECLARATIONS:
            continue
        storage = None
        qualifiers = set()
        for sub in child.children:
            if sub.type == "storage_class_specifier":
                storage = _text(sub).strip()
            elif sub.type == "type_qualifier":
                qualifiers.add(_text(sub).strip())
        item_kind = None
        if inside_anonymous:
            item_kind = "anonymous-namespace"
        elif storage == "static":
            item_kind = "static"
        elif qualifiers & {"const", "constexpr"}:
            item_kind = "internal-const"
        if item_kind is None:
            continue
        # `static` members and `static` local variables are never reached here:
        # the traversal only descends into namespaces and linkage blocks.
        for name, definition in _declared_names(child):
            if definition != "definition":
                continue
            out.append({
                "kind": item_kind,
                "scope": "::".join(scope),
                "name": name,
                "line": child.start_point[0] + 1,
            })


def scan(path: str):
    """Internal-linkage entities declared at namespace scope of one file."""
    with open(path, "rb") as handle:
        source = handle.read()
    tree = _PARSER.parse(source)
    out: list[dict] = []
    _walk_namespace(tree.root_node, [], False, out)
    return out


_PP_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
_PP_UNDEF = re.compile(r"^\s*#\s*undef\s+([A-Za-z_]\w*)")


def scan_macros(path: str):
    """``#define`` line per macro name and the ``#undef`` lines of one file."""
    defines: dict[str, int] = {}
    undefs: dict[str, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            match = _PP_DEFINE.match(line)
            if match:
                defines.setdefault(match.group(1), lineno)
                continue
            match = _PP_UNDEF.match(line)
            if match:
                undefs.setdefault(match.group(1), lineno)
    return defines, undefs


# ---------------------------------------------------------------------------
# repository / build-system inventory
# ---------------------------------------------------------------------------

def _relative(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def tracked_cpp():
    """Every ``src/**/*.cpp`` file of this checkout (vendored code excluded).

    The build tree is walked instead of ``git ls-files`` so a brand-new file is
    checked before it is committed: a unity batch that only collides with an
    untracked file still fails to compile.
    """
    result = []
    src = os.path.join(ROOT, "src")
    skip = os.path.join(src, "ext") + os.sep
    for dirpath, dirnames, filenames in os.walk(src):
        if (dirpath + os.sep).startswith(skip):
            dirnames[:] = []
            continue
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".cpp"):
                result.append(_relative(os.path.join(dirpath, name)))
    return result


_TARGET_START = re.compile(r'^\s*target\s*\(\s*"([^"]+)"', re.M)
_BATCH_SIZE = re.compile(r"batch_size\s*=\s*(\d+)")
_ADD_FILES = re.compile(r"add_files\s*\(")
_ADD_FILES_METHOD = re.compile(r'target:add\s*\(\s*"files"\s*,')
_LOCAL_BINDING = re.compile(r"^\s*local\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)


def _split_call_arguments(text: str, open_index: int):
    """Split a Lua call argument list starting at ``text[open_index] == '('``.

    Returns ``(arguments, end_index)``.  Quoted strings, nested parentheses and
    option tables are honored, so ``f(a, g(b, c), {x = 1, y = ")"})`` yields
    ``["a", "g(b, c)", "{x = 1, y = \")\"}"]``.
    """
    depth = 0
    quote = None
    arguments: list[str] = []
    buffer: list[str] = []
    index = open_index
    while index < len(text):
        char = text[index]
        if quote is not None:
            buffer.append(char)
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
        elif char in "\"'":
            quote = char
            buffer.append(char)
        elif char in "([{":
            depth += 1
            if depth > 1:
                buffer.append(char)
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                tail = "".join(buffer).strip()
                if tail:
                    arguments.append(tail)
                return arguments, index
            buffer.append(char)
        elif char == "," and depth == 1:
            arguments.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        index += 1
    return None, index


def _eval_lua_path(expr: str, scriptdir: str, bindings: dict, depth: int = 0):
    """Evaluate the subset of Lua path expressions this repository uses.

    Understands string literals, ``os.scriptdir()``, ``path.join``,
    ``path.absolute``, ``path.normalize`` and identifiers bound by a preceding
    ``local name = <same forms>``.  Anything else (function calls such as
    ``rela(...)``, string concatenation, configuration lookups) yields ``None``
    so the caller can skip that pattern instead of guessing.
    """
    if expr is None or depth > 8:
        return None
    expr = expr.strip()
    while expr.endswith(",") or expr.endswith(")"):
        if expr.endswith(","):
            expr = expr[:-1].strip()
        else:
            break
    if not expr:
        return None
    if len(expr) >= 2 and expr[0] in "\"'" and expr[-1] == expr[0]:
        return expr[1:-1]
    if expr == "os.scriptdir()":
        return scriptdir
    for function in ("path.join", "path.absolute", "path.normalize"):
        if expr.startswith(function) and "(" in expr:
            arguments, _ = _split_call_arguments(expr, expr.index("("))
            if not arguments:
                return None
            values = [_eval_lua_path(a, scriptdir, bindings, depth + 1)
                      for a in arguments]
            values = [v for v in values if v is not None]
            if not values:
                return None
            if function == "path.normalize":
                return os.path.normpath(values[0])
            if function == "path.absolute" and len(values) > 1:
                # path.absolute(p, base) is path.join(base, p), not path.join(p, base)
                values = [values[-1]] + values[:-1]
            return os.path.normpath(os.path.join(*values))
    binding = re.fullmatch(r"([A-Za-z_]\w*)", expr)
    if binding:
        return bindings.get(binding.group(1))
    return None


def _expand_xmake_glob(pattern: str):
    """Expand one xmake file pattern to absolute paths.

    ``**`` means "any depth" in xmake, which maps onto ``glob``'s recursive mode
    once it is followed by a separator (glob's ``**`` matches zero or more
    directories).  A ``|`` inside one pattern separates alternatives.
    """
    result = []
    for alternative in pattern.split("|"):
        alternative = alternative.replace("\\", "/").strip()
        if not alternative:
            continue
        if "**" in alternative:
            head, tail = alternative.split("**", 1)
            if not tail:
                recursive = head + "**"
            elif tail.startswith("/"):
                # `dir/**/name.cpp`: `**` already separates directories.
                recursive = head + "**/" + tail.lstrip("/")
            else:
                # `dir/**.cpp`: `**` is followed by the suffix, so it stands for
                # "any directories, then any file name".
                recursive = head + "**/*" + tail
            result.extend(glob.glob(recursive, recursive=True))
        else:
            result.extend(glob.glob(alternative))
    return result


def _specificity(pattern: str) -> int:
    """How specific a file pattern is: literal characters, wildcards ignored."""
    return sum(1 for c in pattern if c not in "*?[]")


_xmake_cache: dict[str, list[tuple[str, int, list[str]]]] = {}


def _read_xmake(path: str):
    """Parse one ``xmake.lua`` into ``[(target, batch_size, [pattern]), ...]``.

    Patterns are the raw file patterns of ``add_files(...)`` and
    ``target:add("files", ...)`` of that target, still relative to the script
    directory (or absolute when the script built them that way).
    """
    cached = _xmake_cache.get(path)
    if cached is not None:
        return cached
    scriptdir = os.path.dirname(path)
    targets: list[tuple[str, int, list[str]]] = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        starts = list(_TARGET_START.finditer(text))
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            block = text[match.end():end]
            batch_match = _BATCH_SIZE.search(block)
            batch = int(batch_match.group(1)) if batch_match else 0
            bindings: dict[str, str | None] = {}
            for binding in _LOCAL_BINDING.finditer(block):
                bindings[binding.group(1)] = _eval_lua_path(
                    binding.group(2), scriptdir, bindings)
            patterns: list[str] = []
            argument_lists: list[list[str]] = []
            for call in _ADD_FILES.finditer(block):
                arguments, _ = _split_call_arguments(text, match.end() + call.end() - 1)
                if arguments:
                    argument_lists.append(arguments)
            for call in _ADD_FILES_METHOD.finditer(block):
                arguments, _ = _split_call_arguments(
                    text, match.end() + block.index("(", call.start() - 0))
                if arguments:
                    argument_lists.append(arguments[1:])
            for arguments in argument_lists:
                for argument in arguments:
                    if argument.startswith("{"):
                        continue
                    resolved = _eval_lua_path(argument, scriptdir, bindings)
                    if not resolved:
                        continue
                    if not os.path.isabs(resolved):
                        resolved = os.path.join(scriptdir, resolved)
                    patterns.append(os.path.normpath(resolved))
            targets.append((match.group(1), batch, patterns))
    _xmake_cache[path] = targets
    return targets


def _xmake_files():
    files = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
        dirnames.sort()
        if "xmake.lua" in filenames:
            files.append(os.path.join(dirpath, "xmake.lua"))
    return files


_target_index: dict[str, tuple[int, str, int]] | None = None


def _build_target_index():
    """Map every owned source file to ``(specificity, target, batch size)``."""
    index: dict[str, tuple[int, str, int]] = {}
    for xmake in _xmake_files():
        for target, batch, patterns in _read_xmake(xmake):
            for pattern in patterns:
                for path in _expand_xmake_glob(pattern):
                    if not path.lower().endswith((".cpp", ".cc", ".cxx", ".mm")):
                        continue
                    key = os.path.normcase(os.path.normpath(path))
                    specificity = _specificity(pattern)
                    current = index.get(key)
                    if current is None or specificity > current[0]:
                        index[key] = (specificity, target, batch)
    return index


def target_of(f: str):
    """``(xmake target, unity batch size)`` that owns the file ``f``.

    ``f`` is repo-relative (POSIX or native separators).  The owning target is
    taken from the ``add_files`` / ``target:add("files", ...)`` patterns of the
    repository's ``xmake.lua`` files, where the most specific pattern wins so a
    ``tile/*.cpp`` target beats a ``*.cpp`` one.  Files that no target claims
    report ``("", 0)`` and are therefore ignored by the checker.
    """
    global _target_index
    if _target_index is None:
        _target_index = _build_target_index()
    key = os.path.normcase(os.path.normpath(
        os.path.join(ROOT, f.replace("\\", "/").replace("/", os.sep))))
    entry = _target_index.get(key)
    if entry is not None:
        return entry[1], entry[2]
    # Files a target adds from on_load with a computed path are not reachable
    # through a static pattern: fall back to the nearest xmake.lua that declares
    # exactly one unity-batched target, which is the module that owns them.
    directory = os.path.dirname(os.path.join(ROOT, f.replace("/", os.sep)))
    while len(directory) >= len(ROOT):
        local = os.path.join(directory, "xmake.lua")
        if os.path.isfile(local):
            batched = [(name, batch) for name, batch, _ in _read_xmake(local)
                       if batch > 1]
            if len(batched) == 1:
                return batched[0]
            break
        if directory == ROOT:
            break
        directory = os.path.dirname(directory)
    return "", 0


_UNITY_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def parse_unity_batches(gens: str = "build/.gens", plat: str = "windows"):
    """Read the generated unity batches back into ``{file: (target, unit)}``.

    ``gens`` is the xmake autogen root (``build/.gens`` by default) and ``plat``
    the platform component that selects this host's batches.  The layout is
    ``<gens>/<target>/<plat>[/<arch>[/<mode>]]/unity_build/unity_N.cpp``; each
    unit lists the translation units it merged as quoted ``#include`` lines.
    """
    root = gens if os.path.isabs(gens) else os.path.join(os.getcwd(), gens)
    if not os.path.isdir(root):
        root = os.path.join(ROOT, gens)
    batches: dict[str, tuple[str, str]] = {}
    if not os.path.isdir(root):
        return batches
    # `--gens` accepts either the autogen root (``build/.gens``) or the build
    # directory that contains it (``build``, what the xmake target passes).
    autogen = os.path.join(root, ".gens")
    if os.path.isdir(autogen):
        root = autogen
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if os.path.basename(dirpath) != "unity_build":
            continue
        parts = os.path.relpath(dirpath, root).replace(os.sep, "/").split("/")
        if plat not in parts:
            continue
        target = parts[0]
        for name in sorted(filenames):
            if not name.endswith(".cpp"):
                continue
            unit = os.path.splitext(name)[0]
            with open(os.path.join(dirpath, name), "r", encoding="utf-8",
                      errors="replace") as handle:
                for line in handle:
                    match = _UNITY_INCLUDE.match(line)
                    if not match:
                        continue
                    included = match.group(1).replace("\\", "/")
                    resolved = os.path.normpath(
                        os.path.join(dirpath, included))
                    batches[_relative(resolved)] = (target, unit)
    return batches


# ---------------------------------------------------------------------------
# debugging entry point
# ---------------------------------------------------------------------------

def _main(argv) -> int:
    if not argv:
        print(f"checked paths: {len(tracked_cpp())} translation units",
              file=sys.stderr)
        return 0
    if argv[0] == "--batches":
        gens = argv[argv.index("--gens") + 1] if "--gens" in argv else "build/.gens"
        plat = argv[argv.index("--plat") + 1] if "--plat" in argv else "windows"
        batches = parse_unity_batches(gens, plat)
        for f, (target, unit) in sorted(batches.items()):
            print(f"{target}/{unit}: {f}")
        print(f"{len(batches)} translation units in "
              f"{len({(t, u) for t, u in batches.values()})} batches")
        return 0
    for path in argv:
        target, batch = target_of(_relative(path))
        print(f"== {path} (target {target or '?'}, batch size {batch})")
        for item in scan(path):
            print(f"   {item['kind']:20s} {item['scope']}::{item['name']} "
                  f"@ {item['line']}")
        defines, undefs = scan_macros(path)
        leaked = sorted(n for n in defines if n not in undefs)
        print(f"   macros: {len(defines)} defined, {len(leaked)} without #undef"
              + (f" ({', '.join(leaked[:5])})" if leaked else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
