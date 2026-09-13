"""Render audited Markdown; package only after an explicit --package invocation.

This script never executes captured kernels, compilers, or audit programs. The
source and evidence archives must be independently reviewed before publication.
"""

import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import shutil
import tarfile

OUT = Path(__file__).resolve().parent
EDGES = ("copy-m0", "copy-m4", "mma-c0", "mma-c4", "simplified-m4-c4")
ARMS = ("m0-c0", "m0-c4", "m4-c0", "m4-c4", "m4-c4-simplified")


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(name, value):
    with (OUT / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_results(raw):
    main = read_json(raw / "offline-audit-1.stdout")
    reference = read_json(raw / "reference-offline-audit-1.stdout")
    assert main["status"] == reference["status"] == "passed"
    assert [main[x] for x in ("captures", "pair_replays", "visits", "samples", "commands_checked")] == [30, 30, 360, 2520, 240]
    assert [reference[x] for x in ("preparations", "paired_replays", "visits", "samples", "protocol_tests", "native_selftest_checks")] == [3, 6, 72, 504, 22, 24]
    assert len(main["comparisons"]) == 30 and len(reference["comparisons"]) == 6
    cases = [row["case"] for row in main["captures_checked"]]
    assert len(cases) == len(set(cases)) == 6
    assert {(r["case"], r["edge"]) for r in main["comparisons"]} == {(c, e) for c in cases for e in EDGES}
    for row in main["comparisons"]:
        assert len(row["candidate_over_baseline"]["pairs"]) == 6
    for row in reference["comparisons"]:
        assert len(row["reference_over_tile"]["pairs"]) == 6
    return main, reference, cases


def render_table(raw):
    main, reference, cases = audit_results(raw)
    by_edge = {(r["case"], r["edge"]): r for r in main["comparisons"]}
    lines = [
        "# Attention joint copy/MMA：完整审计表",
        "",
        "由 `package.py --render-table-only RAW` 从两份已通过的离线审计 JSON 生成；不重跑计时、不选最快臂。解释与限制见 [notes.md](notes.md)。",
        "",
        "## 六个 case × 五条配对边",
        "",
        "下表为 candidate/baseline 的六个相邻 AB/BA 配对比值的中位数。**小于 1 表示候选耗时更短**。每格属于独立 cohort，不能把跨格比值相乘或反推未测过的直接比较。",
        "",
        "| Case | Copy，M0 | Copy，M4 | MMA，C0 | MMA，C4 | Simplified，M4C4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        ratios = [by_edge[case, edge]["candidate_over_baseline"]["median"] for edge in EDGES]
        lines.append("| " + case + " | " + " | ".join(f"{v:.4f}" for v in ratios) + " |")
    lines += [
        "", "## 全部 30 条边", "",
        "时间单位 µs。每臂时间是六个 visit 中位数的中位数；每个 visit 有七个样本。比值是六个配对比值的中位数，不一定等于表中两列时间相除。范围是六对的实际 min–max，**不是置信区间**。",
        "",
        "| Case | 边：baseline → candidate | Baseline µs | Candidate µs | 配对比值 | 配对 min–max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        for edge in EDGES:
            row = by_edge[case, edge]
            ratio = row["candidate_over_baseline"]
            baseline, candidate = ratio["baseline"], ratio["candidate"]
            lines.append(f"| {case} | {baseline} → {candidate} | {row['summary_us'][baseline]:.3f} | {row['summary_us'][candidate]:.3f} | {ratio['median']:.4f} | {ratio['minimum']:.4f}–{ratio['maximum']:.4f} |")
    lines += [
        "", "## 固定 Tile 臂与 NEON / Accelerate 的六条补充对照", "",
        "Tile 始终固定为预声明的 `m4-c4-simplified`，没有按 case 挑选最快臂。这里比值方向为 **reference/Tile，小于 1 表示 Tile 更慢**。补充 cohort 与上面的优化矩阵独立，不能拼接时间或相乘速度比。",
        "",
        "| Case | Reference | Tile µs | Reference µs | Reference / Tile | 配对 min–max | 最大绝对误差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in reference["comparisons"]:
        ratio = row["reference_over_tile"]
        lines.append(f"| {row['case']} | {row['variant']} | {row['tile_us']:.3f} | {row['reference_us']:.3f} | {ratio['median']:.4f} | {ratio['minimum']:.4f}–{ratio['maximum']:.4f} | {row['max_abs_error']:.3e} |")
    lines += [
        "", "## 五臂实际资源与 full-packet 准入", "",
        "Snapshot 单位为每个逻辑 worker 的字节，allocation / interleaved 均为数量；workspace 是所捕获入口的字节需求。clone 列记录实际生成数量，不用 requested flag 代替。Source/candidate 是准入用 LLVM 指令数（目标 O1/O2 前），不是机器指令数；`ineligible` 的 0 不代表函数为空。",
        "",
        "| Case | 臂 | Snapshot B / alloc | Interleaved | Workspace B | ABI | Clone | Source → candidate | 准入结果 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in main["captures_checked"]:
        for arm in ARMS:
            entry = case["variants"][arm]
            p = entry["physical"]
            lines.append(f"| {case['case']} | {arm} | {p['static_snapshot_bytes_per_worker']} / {p['static_snapshot_allocations']} | {p['interleaved_private_arrays']} | {entry['workspace_bytes']} | {entry['abi']} | {p['full_packet_specializations']} | {p['full_packet_source_instructions']} → {p['full_packet_candidate_instructions']} | {entry['specialization_decision']} |")
    lines += [
        "", "## 数据身份", "",
        f"- 主审计 JSON SHA-256：`{sha(raw / 'offline-audit-1.stdout')}`。",
        f"- 主 plan SHA-256：`{main['plan_sha256']}`。",
        f"- 补充审计 JSON SHA-256：`{sha(raw / 'reference-offline-audit-1.stdout')}`。",
        f"- 补充 plan SHA-256：`{reference['plan_sha256']}`。",
        "- 原始样本、六个配对比值、完整输出与对象身份均保留在证据中；本表仅为四舍五入展示。",
        "",
    ]
    with (OUT / "table.md").open("x") as stream:
        stream.write("\n".join(lines))


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("refusing symlink: " + str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("refusing nonregular evidence: " + str(path))
        result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def tar_inventory(path, mode):
    result = {}
    with tarfile.open(path, mode) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or member.name in result or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe/nonregular/duplicate archive member")
            result[member.name] = dict(bytes=member.size, sha256=hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest())
    return result


def package(raw):
    audit_results(raw)
    outputs = ("evidence.tar.xz", "sources.tar.gz", "provenance.json", "audit.json", "reference-audit.json", "package-inventory.json", "SHA256SUMS")
    if any((OUT / name).exists() for name in outputs):
        raise ValueError("package outputs already exist; preserve earlier attempts")
    if not all((OUT / name).is_file() for name in ("notes.md", "table.md")):
        raise ValueError("reviewed Markdown reports are required before packaging")
    before = inventory(raw)
    provenance = read_json(raw / "provenance.json")
    source = before["sources.tar.gz"]
    source_members = tar_inventory(raw / "sources.tar.gz", "r:gz")
    if source["sha256"] != provenance["source_archive_sha256"] or {n: v["sha256"] for n, v in source_members.items()} != provenance["source_sha256"]:
        raise ValueError("source archive does not match its frozen provenance")
    evidence = {n: v for n, v in before.items() if n != "sources.tar.gz"}
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 0, "dict_size": 128 * 1024 * 1024}]
    with (OUT / "evidence.tar.xz").open("xb") as destination:
        with lzma.LZMAFile(destination, "w", filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, metadata in evidence.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = metadata["bytes"], 0o644
                    with (raw / name).open("rb") as stream:
                        archive.addfile(info, stream)
    for origin, destination in (("sources.tar.gz", "sources.tar.gz"), ("provenance.json", "provenance.json"), ("offline-audit-1.stdout", "audit.json"), ("reference-offline-audit-1.stdout", "reference-audit.json")):
        shutil.copyfile(raw / origin, OUT / destination)
    if inventory(raw) != before:
        raise ValueError("raw changed while packaging; preserve failed output")
    if tar_inventory(OUT / "evidence.tar.xz", "r:xz") != evidence or sha(OUT / "sources.tar.gz") != source["sha256"]:
        raise ValueError("archive readback differs from original evidence")
    write_json("package-inventory.json", dict(
        format="attention-joint-repack-evidence-v1", original_raw=str(raw),
        raw_unchanged_after_packaging=True, source_archive=source,
        source_member_count=len(source_members), excluded_from_evidence=["sources.tar.gz"],
        evidence_compression=dict(format="xz", codec="lzma2", preset=0, dictionary_bytes=128 * 1024 * 1024, threads=1),
        evidence_files=evidence, evidence_file_count=len(evidence),
        evidence_uncompressed_bytes=sum(v["bytes"] for v in evidence.values()),
        evidence_sha256=sha(OUT / "evidence.tar.xz"),
        boundary="Original regular raw files preserved; source archive stored separately once. Internal hash consistency, not loader-closure attestation or a reproduced build. No new native execution."))
    with (OUT / "SHA256SUMS").open("x") as stream:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(sha(path) + "  " + path.name + "\n")
    print(json.dumps(dict(status="packaged", evidence_files=len(evidence), evidence_bytes=(OUT / "evidence.tar.xz").stat().st_size, source_members=len(source_members), raw_unchanged=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render-table-only", action="store_true")
    mode.add_argument("--package", action="store_true", help="requires an explicit packaging approval; performs full hashing/compression/readback")
    args = parser.parse_args()
    raw = args.raw.resolve(strict=True)
    if raw == OUT or OUT.is_relative_to(raw) or raw.is_relative_to(OUT):
        raise ValueError("source and package must be disjoint")
    if args.render_table_only:
        render_table(raw)
    else:
        package(raw)
