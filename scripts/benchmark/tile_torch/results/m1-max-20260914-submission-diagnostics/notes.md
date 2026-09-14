# Submission 诊断原始证据：2026-09-14

这是小型增量归档，不是性能报告，也不是成功准入。原始 benchmark 与 full6 数据见[前一归档](../m1-max-20260914-program-team/notes.md)，这里不重复其约31MiB内容。

## 覆盖与终态

| 原始诊断 | 保留结果 | 解释边界 |
|---|---|---|
| Metal copy-only v3 | 30秒 cap，退出-15，Error | 保存源码、runner、plan、compile/link产物和日志、native收据、两次sample。不能据此宣称GPU恢复或copy数值正确。 |
| Metal copy-only v4 | 30秒 cap，退出-15，Error | 额外保留event/tail-pulse观察与evidence-report；`synchronize_returned=false`、`full_numeric_checks_passed=false`。event=1只覆盖前四次upload，绝不是16次copy数值检查通过。 |
| full7 | 完整build通过，unit SIMD为12/13通过、1失败 | 失败是新增uniform read-lane相关结构断言；保留原始完整stdout/stderr和收据，不把其他通过项拼成全绿。另含八项syntax/clang-tidy及八项format收据。 |
| full8 diagnostic | 完整build通过，单独codegen诊断失败 | 保存更详细的IR诊断及失败；不把诊断编译通过等同于正确性通过。 |

这两次 Metal 运行是独立的 Runtime submission 诊断，不作为 Tile kernel 回归结果；本目录不生成GPU性能数据、不执行native，也不再次触发提交。

## 源码与复现边界

`diagnostics.tar.gz` 原样包含 `source-uniform-read-lane.tar.gz`（full7）和 `source-uniform-read-lane-diagnostic.tar.gz`（full8），各23条真实源码路径逐项匹配自己的freeze SHA。可能存在的AppleDouble `._*` 元数据保留并单列，不计为源码。归档程序不读取正在变化的工作区源码，也不从当前selected树重建旧版本。

同时保留23项 `owned.json`、17项 `owned-through-full6.json`、run/toolchain/source-clone与clone收据。两个源码tar只是明确的overlay，不是完整Git快照；不包含完整系统、SDK、LLVM、Runtime依赖闭包。Metal probe自身的实际object/executable包含在其raw目录中。

仅纳入明确已终态的v3/v4/full7/full8文件；不含live full9、未来CPU ablation、pycache或机器缓存。`manifest.json` 的 `status=passed` 只表示归档完整性通过，`terminal_diagnostics` 内仍保留各项真实Error状态。

## 校验

`SHA256SUMS` 给出顶层文件hash；`manifest.json` 给出压缩包及每个member的SHA-256、大小、原始路径、源码freeze匹配和终态状态。`archive.py` 是一次性机械归档程序：不修改旧收据、不重新运行测试、不改变selected源码。打包后已逐member回读验证，并检查原始文件在归档期间未变化。
