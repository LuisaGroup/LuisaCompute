# Joint copy/MMA 实验未完成：保留编译超时，不报告新加速比

2026-09-13，Apple M1 Max / macOS 26.6.2，FP32，XIR → SIMD → LLVM。

预声明为六个 attention case × 五个实现：`m0-c0`、`m0-c4`、`m4-c0`、`m4-c4`、`m4-c4-simplified`。`m`/`c` 分别表示 native MMA/copy 向量宽度；最后一臂另启用 provisional full-packet clone 的局部简化。

**本轮只有 26 个 capture + prepare 成功，1 个 Error，3 个 NotRun；没有运行任何竞争性 ABBA replay。** 原计划的 30 对 replay、360 visits、2520 samples 均未执行。本目录没有新原生性能比值，也不能据此宣称超过 Torch/MPS/BLAS。成功 capture 自带的 synchronized Runtime 诊断计时完整留档，但不冒充预声明的单线程原生入口配对结果。

| Case | M0 C0 | M0 C4 | M4 C0 | M4 C4 | M4 C4 simplified |
|---|---|---|---|---|---|
| decode-mha-d64 | OK | **Error：180s capture 超时** | NotRun | NotRun | NotRun |
| decode-gqa-d80 | OK | OK | OK | OK | OK |
| decode-long-kv | OK | OK | OK | OK | OK |
| prefill-q4 | OK | OK | OK | OK | OK |
| prefill-q8 | OK | OK | OK | OK | OK |
| batch-gqa-q4 | OK | OK | OK | OK | OK |

`OK` 仅表示已有成功 capture、完整输出及 prepared 原生对象；不是完整 cohort 的性能审计通过。失败后的三个 MHA 实现未启动，不代表它们失败。其他五个 case 的五臂 capture 校验由冻结 driver 完成，回执保留在 `captures.json`；本次 failure audit 不重新计算 FP64 oracle。

## 超时证据与边界

失败配置为 `[B,Hq,Hkv,Q,KV,D,Dv]=[1,8,8,1,2048,64,64]`，block `1×16`，packet W8，MMA output block 4，MMA unroll cap 0、native MMA 0、native copy 4。超时监督终止整个隔离进程组，退出码 `-9`；原始 stdout/stderr 为空，输入与 FP64 oracle 已导出，但无输出、LLVM 文件、对象或 prepare。

`m0-c4-compile-sample.txt` 的主线程 727 次观察都处在 `LLVMJIT::lookup` 下的机器码生成路径，726 次进入 `llvm::LiveVariables::analyze`，多数时间在 `runOnBlock`；采样报告记录 physical footprint 2.0G、峰值 3.1G。它证明问题位于 LLVM 机器级活跃变量分析，而非 GPU 执行或 runtime 同步。

**180 秒是 capture 总时限，不是单次生产 JIT 的实测时长。** 此次启用了 assembly capture：冻结的 `simd_compiler.cpp` 先 `emit_assembly_copy`（优化 + 机器码生成），再 `add_module`/ORC `lookup`（再次优化 + 机器码生成）。采样已在第二遍，第一遍已返回。不能把双编译超时改写为“生产单次 JIT 必然超过 180 秒”，也不能把超时当作极慢 kernel 时间。

## 已确认的静态机制与尚未证明的部分

冻结源码中，`_find_interleaved_private_arrays` 只允许闭合的 `alloca → GEP → scalar load/store` 地址树；typed copy 的完整数组引用会排除 K/V 的 packet-interleaved 排布。普通 MMA 消费者随之从连续 packet vector load 退回 masked gather。cap 0 的 D64 与小输出继续完全展开，形成大量 gather 投影。

同一轮成功的 MHA `m0-c0` 元数据显示：4 块 interleaved private arrays、2050 个静态 contiguous-private read sites、0 个 rolled MMA、原始 packet body 42529 条 LLVM 指令。按冻结 MMA 投影结构，K/V 对应约 2048 个读取站点；copy 引用破坏交错排布与机器码膨胀构成强假设，但失败臂没有落盘的 pre/post-opt IR，**不能宣称已经量化它实际生成的 gather、机器指令或活跃区间数量**。

一般性启示是共同选择 producer copy 排布、consumer 访问与展开，而非为 copy 和 MMA 各加固定收益。后续修复应在新源码、新 cohort 中验证；本目录保留原失败版本，不回填新结果。

## 构建、身份和复核

冻结计划使用 `full-build-2.json` 的完整 selected-tree 构建及 `regression-2.json` 的 11 项 CTest，全通过；CTest 总耗时 146.96 秒。先前构建、语法与回归尝试全部保留。`sources.tar.gz` 是冻结的选定源导出，不是完整 HEAD，也不是动态加载依赖闭包证明；provenance 中的 HEAD 只标识当时工作树基线。后续主树修改不改变这些源、对象或指纹。

- `evidence.tar.xz`：全部原始 regular evidence（包括旧 `audit.py`、失败命令、采样、成功对象及 prepared 文件），仅排除另存的 `sources.tar.gz`，避免重复打包源归档。
- `sources.tar.gz`、`provenance.json`：原文件逐字保留。证据包也包含原 provenance，便于独立提取。
- `package-inventory.json`、`SHA256SUMS`：逐成员大小/哈希和顶层文件校验。
- `failure_audit.py`、`failure-audit.json`：专门核对 incomplete 状态、27 次 capture、26 次 prepare、130 条内部工具命令，共 183 条命令回执，源/对象身份及无 replay。`incomplete_evidence_consistent` 仅说明失败证据一致，绝不表示完整实验通过。

证据包采用单线程 xz / LZMA2 preset 0、128 MiB 字典，以去除跨臂及 raw/prepared 的重复大数组；没有删除或降低任何证据精度。打包器完整读回压缩流，逐成员重新核对哈希，并确认原 raw 在打包前后不变。

离线复核（无 native 执行）：

```sh
mkdir extracted
tar -xf evidence.tar.xz -C extracted
python3 failure_audit.py extracted --sources sources.tar.gz
```

不要修改或运行旧完整实验 `audit.py` 来制造 pass；它要求完整 cohort，当前证据本来就不满足。
