#!/usr/bin/env python3
"""Build an isolated legacy Tile -> ordinary AST exporter, not a legacy backend.

Requires the pinned Git/submodule objects locally; never switches branches or
updates dependency checkouts. The only historical library edit adds AST JSON
schema/usage/warp metadata required by the current strict decoder. It changes
neither the example bodies nor tile_to_kernel or its optimization decisions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

COMMIT = "ccdfcbebef7fa95431c988e1fcbdd87ffdce9fdc"
LOWERING_BLOB = "7dd4c2b65bc277d8682f334f841f59d230281536"
EXAMPLES_BLOB = "552a6c495402ee81455bf7a790dbf13164d1ba7e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def export(repository, destination, commit, relative, records):
    if Path(git(repository, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repository.resolve():
        raise RuntimeError(f"Dependency is not initialized: {repository}")
    git(repository, "cat-file", "-e", commit + "^{commit}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"Refusing to populate a nonempty source directory: {destination}")
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", commit], stdout=subprocess.PIPE)
    unpack = subprocess.run(["tar", "-xf", "-", "-C", str(destination)], stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() != 0 or unpack.returncode != 0:
        raise RuntimeError(f"Cannot export pinned source: {relative}")
    records.append(dict(path=relative, pinned_commit=commit, checkout_commit=git(repository, "rev-parse", "HEAD").decode().strip()))
    for entry in git(repository, "ls-tree", "-rz", commit).split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, _, revision = metadata.split()
            if mode == b"160000":
                name = path.decode()
                # git archive may emit an empty submodule directory; export
                # checks that it is empty without deleting an existing path.
                export(repository / name, destination / name, revision.decode(), relative + "/" + name, records)


def instrument_serializer(path):
    text = path.read_text()
    edits = [
        ('{"type", _type_index(v.type())},', '{"type", _type_index(v.type())},\n            {"usage", luisa::to_string(_func_ctx->f.variable_usage(v.uid()))},'),
        ('ctx.j["tag"] = luisa::to_string(f.tag());', 'ctx.j["tag"] = luisa::to_string(f.tag());\n        if (auto warp_size = f.allowed_warp_size()) { ctx.j["allowed_warp_size"] = static_cast<uint32_t>(*warp_size); }'),
        ('j["entry"] = entry;', 'j["schema"] = "luisa.compute.ast";\n        j["version"] = 1u;\n        j["entry"] = entry;'),
    ]
    for before, after in edits:
        if text.count(before) != 1:
            raise RuntimeError("Historical AST serializer changed; refusing an ambiguous instrumentation patch")
        text = text.replace(before, after)
    path.write_text(text)


def parameterized_examples(source):
    """Derive a separate translation unit; never edit the original examples.

    Only the container dimensions become ordinary host staging values. Block
    sizes, threads, pipeline settings and every operation body stay unchanged.
    The original exporter remains available for bit-identical default checks.
    """
    text = (source / "examples/compute/tile_bench.cpp").read_text()
    text = text[:text.index("int main(int argc, char *argv[]) {")]
    edits = []
    pattern = r"constexpr tile_i32 (M = \d+, N = \d+(?:, K = \d+)?|N = 2048);"

    def dimensions(match):
        names = [item.split(" = ")[0] for item in match.group(1).split(", ")]
        replacement = "const tile_i32 " + ", ".join(f"{name} = legacy_shape.{name.lower()}" for name in names) + ";"
        edits.append(dict(before=match.group(0), after=replacement))
        return replacement

    text, count = re.subn(pattern, dimensions, text)
    if count != 9:
        raise RuntimeError(f"Unexpected dimension declaration count: {count}")
    original = "constexpr tile_i32 OP_M = 8192, OP_N = 512;"
    replacement = "const tile_i32 &OP_M = legacy_shape.m, &OP_N = legacy_shape.n;"
    if text.count(original) != 1:
        raise RuntimeError("Unexpected shared dimension declaration")
    text = text.replace(original, replacement)
    edits.append(dict(before=original, after=replacement))
    # Referenced globals are read after main initializes the selected shape.
    marker = "using tile_i32 = luisa::compute::tile::int32;"
    text = text.replace(marker, marker + "\nstruct LegacyShape { tile_i32 m{}, n{}, k{}; };\nLegacyShape legacy_shape;", 1)
    (source / "legacy_tile_bench_parameterized.cpp").write_text(text)
    return edits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are never overwritten")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error("jobs must be between 1 and 64")
    repository, output = args.repository.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for path, expected in [("src/ast/tile_to_kernel.cpp", LOWERING_BLOB), ("examples/compute/tile_bench.cpp", EXAMPLES_BLOB)]:
        if git(repository, "rev-parse", COMMIT + ":" + path).decode().strip() != expected:
            raise RuntimeError("Legacy baseline identity mismatch")
    records = []
    source, build = output / "source", output / "build"
    export(repository, source, COMMIT, ".", records)
    original = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    (output / "original-source.json").write_text(json.dumps(original, indent=2) + "\n")
    instrument_serializer(source / "src/ast/ast2json.cpp")
    dimension_edits = parameterized_examples(source)
    shutil.copy2(Path(__file__).with_name("legacy_tile_export.cpp"), source / "emit_legacy_tile.cpp")
    with (source / "CMakeLists.txt").open("a") as file:
        file.write("\nluisa_compute_add_executable(emit_legacy_tile emit_legacy_tile.cpp)\n")
        file.write("luisa_compute_add_executable(emit_legacy_tile_sized emit_legacy_tile.cpp)\n")
        file.write("target_compile_definitions(emit_legacy_tile_sized PRIVATE LUISA_LEGACY_SIZED=1)\n")
    changed = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file() and original.get(str(p.relative_to(source))) != sha(p)}
    if set(changed) != {"src/ast/ast2json.cpp", "emit_legacy_tile.cpp", "CMakeLists.txt", "legacy_tile_bench_parameterized.cpp"}:
        raise RuntimeError("Unexpected change to historical source")
    report = dict(source_commit=COMMIT, lowering_blob=LOWERING_BLOB, examples_blob=EXAMPLES_BLOB,
                  pinned_repositories=records, instrumentation_sha256=changed,
                  sized_exporter_dimension_edits=dimension_edits, runs=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LUISA_", "DYLD_"))}
    configure = ["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                 "-DLUISA_COMPUTE_BUILD_TESTS=OFF", "-DLUISA_COMPUTE_ENABLE_DSL=ON", "-DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF"]
    configure += [f"-DLUISA_COMPUTE_ENABLE_{backend}=OFF" for backend in ["SIMD", "METAL", "METAL4", "GUI", "CUDA", "HIP", "DX", "VULKAN", "FALLBACK", "REMOTE"]]
    for name, command in [("configure", configure), ("build", ["cmake", "--build", build, "--target", "emit_legacy_tile", "emit_legacy_tile_sized", "--parallel", str(args.jobs)])]:
        record = dict(name=name, argv=list(map(str, command)), started_unix=time.time(), log=name + ".log")
        report["runs"].append(record)
        with (output / record["log"]).open("x") as log:
            result = subprocess.run(record["argv"], cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(output / record["log"]))
        (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode != 0:
            raise RuntimeError(f"Legacy exporter {name} failed; see {output / record['log']}")
    report["dependency_checkouts_unchanged"] = all(git(repository / r["path"], "rev-parse", "HEAD").decode().strip() == r["checkout_commit"] for r in records if r["path"] != ".")
    report["source_unchanged_except_instrumentation"] = all(sha(source / p) == digest for p, digest in original.items() if p not in changed)
    report["exporter_sha256"] = sha(build / "bin/emit_legacy_tile")
    report["sized_exporter_sha256"] = sha(build / "bin/emit_legacy_tile_sized")
    if not report["dependency_checkouts_unchanged"] or not report["source_unchanged_except_instrumentation"]:
        raise RuntimeError("Baseline provenance changed during the build")
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(build / "bin/emit_legacy_tile")


if __name__ == "__main__":
    main()
