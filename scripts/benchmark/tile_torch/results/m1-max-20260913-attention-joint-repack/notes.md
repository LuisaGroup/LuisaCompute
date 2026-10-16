# Attention：copy / MMA 联合对照与 full-packet 简化

2026-09-13，Apple M1 Max，64 GiB，macOS 26.6.2，FP32，XIR → SIMD → LLVM。

这轮完成了预声明的六个 case、五个固定臂、30 条独立配对边：30 次 capture、30 次 ABBA replay、360 个 visits、2,520 个样本；离线审计通过，核对 240 条命令收据。另完成三组固定 Tile 臂与手写 NEON / Accelerate 的六条参照边，72 visits、504 样本。全部正、负结果见 [完整表格](table.md)，没有删掉慢臂或选择每个 case 的最快 baseline。

结论：连续 snapshot copy 与 native MMA 必须联合规划。三组 decode 中，单开 native MMA 反而慢 46.9%–60.6%，有 copy 后 MMA 才分别降低耗时 21.0%–28.2%；这不是一项独立的、可以给所有 MMA 套用的速度折扣。Q4/Q8 的 MMA 边仍有退步，batch 则两种 copy 设置下都受益。与固定手写参照相比仍明显落后，**没有完成超越 BLAS / Torch / MPS 的总体目标**。

## 固定实验与计时边界

尺寸为 `[B, Hq, Hkv, Q, KV, D, Dv]`。`M0/M4`、`C0/C4` 分别代表 native MMA / copy 向量宽度 0（禁用）或 4。

| Case | 尺寸 | BQ × BK | MMA K 展开 cap | 请求 2D blocking |
|---|---|---|---:|---|
| decode-mha-d64 | 1, 8, 8, 1, 2048, 64, 64 | 1 × 16 | 0 | 否 |
| decode-gqa-d80 | 1, 8, 2, 1, 2053, 80, 96 | 1 × 16 | 8 | 否 |
| decode-long-kv | 1, 16, 4, 1, 8193, 128, 128 | 1 × 16 | 8 | 否 |
| prefill-q4 | 1, 4, 2, 32, 65, 32, 32 | 4 × 16 | 0 | 是 |
| prefill-q8 | 1, 4, 2, 64, 129, 32, 48 | 8 × 16 | 0 | 是 |
| batch-gqa-q4 | 2, 6, 2, 17, 67, 40, 48 | 4 × 16 | 8 | 是 |

五臂是 `m0-c0`、`m0-c4`、`m4-c0`、`m4-c4`、`m4-c4-simplified`。其余控制固定：R4、packet W8、32 workers/block、local lanes 1、全局 tile 展开阈值 64、region-work 4096、QK/PV 均为 MMA、`fast_math=false`。每条边运行三个 ABBA cycle；每个 visit 七个样本、30 ms warmup、15 ms 目标样本时长。每对固定 root order、task grain 等控制；MMA 开关允许改变 snapshots 和入口 ABI，而不是伪装成固定资源的指令替换。

主指标是 `single_thread_native_entry_host_wall_us`：直接重放捕获的实际 ORC 机器码对象，不由 LLVM 文本重新编译。包含整个 native entry、launch 状态重置、block 遍历以及内部 helper；不含 Python、Luisa Runtime dispatch、JIT、调用方分配/拷贝和验证。这里是单线程主机墙钟时间，不是 GPU kernel 时间，也不是 Runtime 多线程成绩；realization 字符串中的可用 CPU workers 数不改变这个测量口径。

表中时间为六个 visit 中位数的中位数；主比值是六个相邻 AB/BA 比值的中位数。每条边属于独立 cohort，**不能跨边相乘速度比或拼出未测过的直接对照**。范围是实测 min–max，不是置信区间；没有 CPU affinity 或后台活动隔离。

## 这轮证据说明了什么

- Copy 在 M0 的六组全部降低耗时；在 M4 的五组有效，Q8 却慢了 2.62%（六对范围 1.0191–1.0365），负结果保留。
- 三组 decode 的 MMA 边在 C0 是 1.535 / 1.469 / 1.606，在 C4 变为 0.718 / 0.790 / 0.734。这里只做方向上的联合依赖判断，不把独立 cohort 当作严格交互效应估计。
- Simplified full-packet 在 Q4 / Q8 分别降低耗时 25.9% / 13.1%。Q4 的准入从 source 6,495 条变为简化后 3,182 条；Q8 从 4,146 条变为 2,425 条，均恢复实际 clone。Q8 原先 C0 有 clone、C4 无 clone，说明 code-size admission 会改变 copy 的最终收益。
- MHA 也从无 clone 变为一个 clone（4,460 → 2,168 条），但耗时比为 0.9998，不能声称实际加速。batch 的 full-packet 始终 `ineligible`，请求简化并没有生成 clone，其接近 1 的结果是负对照。其余简化边的范围跨过 1，不宣称稳定收益。
- 同 MMA 的 copy / simplified 边保持完整 FP32 输出逐 bit 一致、snapshot 字节与 allocation 数不变。跨 MMA 的资源不同，例如 MHA 从 8,320 B / 4 allocations 变为 9,216 B / 9 allocations；完整资源与实际 clone 数均列入表格。

这里的 LLVM 指令数来自目标 O1/O2 前 dump；被接受的 candidate 已经过局部简化，不包含所有 helper 的总代码量，更不是实际机器指令数或执行周期。当前 `native_mma_cost`、`native_copy_cost` 仍为 `unmodeled`；这些证据支持改进联合 realization 和准入模型，**不代表求解器已经自动选择正确组合**。新开关继续 opt-in，没有根据六组数据修改默认策略。

简化保留 clone 也有代价：Q4 的实际对象从 121,952 B 增至 211,752 B，capture 编译从 3,498.50 ms 增至 6,489.06 ms；Q8 从 35,976 B 增至 63,544 B，443.01 ms 增至 1,074.93 ms。这两组运行更快，但代码与编译开销更高。这里编译包含启用 assembly 导出的双编译过程，不是 production 单次 JIT 成本，不能直接拿来拟合求解器的编译成本。

## MHA 编译 Error 的修复边界

上一轮独立 [incomplete cohort](../m1-max-20260913-attention-joint-incomplete/notes.md) 保留了 26 个成功 capture、1 个 Error、3 个未执行、零 replay。MHA `m0-c4` 在 180 秒 capture 限时内没有返回，采样主要落在 LLVM `LiveVariables`；该失败没有被改写成通过。

本轮的通用修复允许已验证 contiguous-copy 的 destination 成为 private-array interleaving 的封闭使用：每个活跃 program 仍读取连续 source 向量，但按布局 stride 写入各自的 private 元素。普通 MMA 消费者得以继续使用 interleaved 存储；native MMA 的数组引用仍保留其原来的非交错约束。没有 attention 名称识别、隐式改变数学语义或调高展开预算。

这次 MHA `m0-c4` 成功返回，capture 的 `compile_ms=4588.591791`；实际记录仍有 4 个 interleaved arrays、2,050 个 contiguous private reads，snapshot 仍为 8,320 B / 4 allocations。完整构建、11 个 CTests（151.16 秒）与选定语法检查通过。

`compile_ms` 是当前 capture 的完整编译路径：既生成 assembly 副本，也再次走 ORC 优化/机器码 lookup，即**双编译边界**；不是单次 LLVM pass 时间。旧 180 秒是进程限时截断值，不是精确耗时，不能据此报告“编译加速 39 倍”。静态布局机制与成功结果相符，但旧失败没有最终 post-O2 / 机器码供完整逐指令对照。

独立只读 object 核查确认：C4 实际生成 128-bit 连续源读取、按 W8 间隔写入目标；没有把 private consumer 改为逐 program 连续布局。原始入口仍有 64 次 masked gather，机器码中仍能看到逐 lane 分支/加载，因此不能称为整体消除了 gather。Q4/Q8 的 candidate 均经三轮局部清理满足 4096 条预算，最终对象同时保留 fallback。完整说明、六条静态工具命令收据和反汇编保存在证据包的 `code-review/`；静态核查没有执行 native 对象，也不估计热点的时间占比。

## 固定参照：仍有结构性差距

参照实验事先固定 `decode-mha-d64`、`decode-long-kv`、`prefill-q8`，Tile 始终使用 `m4-c4-simplified`，不是逐 case 选择最快臂。使用相同 C++ native timer，完成 22 项协议测试、24 项 native self-test、3 次 preparation、6 条配对 replay；离线审计通过。

完整六条表的比值方向是 **reference / Tile，小于 1 表示 Tile 更慢**：

| Case | online NEON / Tile | dense Accelerate / Tile |
|---|---:|---:|
| decode-mha-d64 | 0.4346 | 0.4060 |
| decode-long-kv | 0.4886 | 0.3378 |
| prefill-q8 | 0.4875 | 0.2369 |

三个固定 Tile 臂全部仍慢于两种手写参照。`online_neon` 是显式四 lane QK 树；`dense_accelerate` 使用 BLAS 并物化 dense scores，允许其重排/FMA。二者是同数学问题的手写参照，不是已经证明合法的严格 MMA 编译器改写。BLAS 内部 packing / 分配计入时间，调用方 workspace 预分配不计；没有把更换算法或数值许可隐藏成单一 lowering 优化。

所有参照请求 OMP / OPENBLAS / VECLIB 为单线程，且在执行线程上核对 `BLASSetThreading` 返回 0、`BLASGetThreading` 模式为 1（visits 前后）。这是同线程 API acknowledgement，**不是独立 profiling 证明内部没有其他 worker**。本轮没有新增 Torch、MPS、GPU 或多线程 Runtime 比较；不跨补充 cohort 与主矩阵乘速度比。

## 正确性、源码身份与复查

全矩阵使用保留的 FP32 输入，独立 FP64 dense causal GQA oracle（NumPy `einsum(optimize=False)`、FP32 scale、bottom-right causal mask），完整检查 shape、有限值、所有输出及 guards，`atol=rtol=5e-5`。主矩阵最大绝对误差约 `2.010e-7`；补充参照最大约 `1.602e-7`。同 MMA 边逐 bit 相等；跨 MMA 不要求逐 bit 相同，但 replay 必须复现自己的捕获结果。输入/guard/oracle 在 preflight、warmup、calibration 与每个 sample batch 后核验，不是在每次计时内部调用后核验。没有由有限用例宣称所有 FP32 溢出、全 mask 行都已覆盖。

本轮 plan 在 `1789313005.468516` 冻结，source archive 在 `1789313006.551321` 冻结，capture 尚未开始。provenance 记录主 HEAD 为 `b2540c68b0aa4b083d7206aea4fb79a499c0b115`；随后提交的 `3621c4dd0` 不应追记成捕获时 HEAD。真正控制证据身份的是选定 source archive（546 个成员、52 个 owned files）、build/gate 收据、准备阶段记录的实际对象与输入 hashes，而不是声称整个工作树都参加了构建。

Source archive SHA-256：`b7f76a442e3cc9e2547d82bd813475efd41f50872d74eb115fe1b44fb3bd0c25`。两份审计与 plan 的完整 hash 见 [表格末尾](table.md#数据身份)。归档脚本只对冻结 raw 做读取、压缩与完整读回验证，不调用编译器或 native 对象。源码/工具 hashes 是内部可复查收据，**不等于完整动态 loader closure 证明或另一台机器复现构建**。

`package.py --render-table-only RAW` 只生成表；`package.py --package RAW` 必须在 raw 最终冻结、获得打包确认后执行。原始 LLVM/对象、完整输出、所有样本、命令与验证日志均保留；源码压缩包单独存放避免重复。独立主审计和 reference 审计仍分别对应各自的预声明与测量口径。

离线复核时，在本报告目录执行以下命令；需要 Python 与 NumPy，原始构建目录和 native 动态库加载都不需要。两份 audit 只读取解包后的证据并重新计算 oracle、配对统计和 hashes，不重测性能。结果输出写到新临时目录，保留包内原始收据不变。

```sh
shasum -a 256 -c SHA256SUMS
attention_audit_dir=$(mktemp -d)
tar -xf evidence.tar.xz -C "$attention_audit_dir"
cp sources.tar.gz "$attention_audit_dir/sources.tar.gz"
python3 "$attention_audit_dir/audit.py" > "$attention_audit_dir/main-reaudit.json"
python3 "$attention_audit_dir/reference/audit.py" > "$attention_audit_dir/reference-reaudit.json"
```
