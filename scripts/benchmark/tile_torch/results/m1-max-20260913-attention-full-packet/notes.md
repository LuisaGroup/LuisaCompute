# Attention：满 packet 特化的纯 native-entry 验证

日期：2026-09-13；Apple M1 Max，LLVM 22.1.8，FP32 precise。

## 结论与比较边界

已有的通用 `LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION` 候选在 attention 上有显著收益：本轮五个实际执行满包特化的配置，配对时间减少 **24%–72%**；其余配置保留为尾包／未触发控制。**这不是新写的 attention 专用 kernel，不是自动 planner 选出的成绩，也不是新的 Torch/MPS 对照。** 开关仍然 opt-in，没有修改生产默认。

每个 case 使用同一二进制、DSL、输入和 FP64 oracle，QK/PV 都为 MMA；仅开关 full-packet specialization。实际映射为 W8、block=32、local=1；不改变归约顺序、fast-math、fusion、私有数组策略。十个 case 包括先验四个诊断点，以及看过诊断结果后事先列出的六个扩展验证点；后者没有用于调整阈值或默认策略。

计时直接链接 Runtime 导出的 ORC `.o`，共用既有 C++ replay helper；单个 CPU 线程计入 native call、launch-record 重置、block 遍历和编译器产生的内部调用，排除 Runtime、Python、JIT、线程池、调用方分配、数据拷贝与验证。这是 **native-entry host-wall**，不是硬件 cycle counter，也不能与多线程 Runtime E2E 混排。

每 case 三轮 ABBA、每 visit 七个样本，warmup 30 ms、每样本目标 15 ms；大于目标的单次 kernel 不截断。表中时间为 visit 中位数的中位数，比例为六组配对比例的中位数，不能用展示时间相除替代。原始 capture 的三样本 Runtime 时间仅用于取对象／正确性检查，不作为比较成绩。

## 完整结果

shape 顺序 `B,Hq,Hkv,Q,K,D,Dv`，时间单位 µs；比例越小越好。

| case / shape | Q×KV block | Off | On | 配对 On/Off | 比例范围 | 实际特化 |
|---|---|---:|---:|---:|---:|---|
| decode：1,8,2,1,2053,80,96 | 1×16 | 3157.990 | 2021.526 | 0.6405 | 0.6322–0.6458 | 1 个满 packet |
| prefill：1,4,2,32,65,32,32 | 4×16 | 140.772 | 140.868 | 1.0004 | 0.9913–1.0159 | 未生成 clone |
| decode-tail：1,6,2,1,2053,80,96 | 1×16 | 3034.393 | 3015.703 | 0.9937 | 0.9886–0.9984 | 生成但没有满 packet |
| small-triangular：1,2,2,5,5,5,3 | 2×3 | 1.392 | 1.394 | 1.0016 | 0.9901–1.0126 | 未生成 clone |
| MHA：1,8,8,1,2048,64,64 | 1×16 | 1138.546 | 1134.951 | 0.9971 | 0.9763–1.0232 | 未生成 clone |
| batch/MQA：2,8,1,1,2053,80,96 | 1×16 | 4656.099 | 1286.591 | 0.2764 | 0.2722–0.2813 | 2 个满 packet |
| mixed-tail：1,10,2,1,2053,80,96 | 1×16 | 5722.417 | 4372.870 | 0.7643 | 0.7551–0.7744 | 1 满包 + 2 lanes 尾包 |
| long-KV：1,16,4,1,8193,128,128 | 1×16 | 28451.958 | 16861.583 | 0.5880 | 0.5816–0.6650 | 2 个满 packet |
| small-Q：1,8,2,3,67,80,96 | 1×16 | 364.369 | 209.811 | 0.5766 | 0.5686–0.5785 | 3 个满 packet |
| triangular：1,4,2,65,65,64,64 | 1×16 | 1534.730 | 1527.085 | 0.9938 | 0.9879–0.9988 | 未生成 clone |

五个有效特化 case 的 30 个配对全部获益；未触发／只有尾包的结果不算优化成功。桌面存在其他负载，扩展 cohort 的 load average 从约 9.29 上升至 18.16；范围不是置信区间，尤其长 KV 的首对波动保留，不挑选最快一轮。

## 对编译与规划的启发

这项开关不是“跳过空 packet”：原 wrapper 已经跳过它们。差别是将满包入口的 active-lane count 固定为 W，使 LLVM 能简化动态 mask；尾包继续使用原始函数，内部越界 mask 与 predicated memory 不能被当成全真。

decode 的优化前函数为 3343 条 LLVM 指令，满足现有 4096 条 clone 构造量上限。实际 `.o` 与汇编都保留；特化后出现共享 `llm_attention.full_packet`。私有 snapshots 容量没有改变；机器码拓扑、mask、寄存器分配与内联同时可能变化，不能把所有收益归因于某一项静态计数。

MHA 的 D=64 与 ragged D=80 出现不同的编译形态：小于现有展开阈值的循环会展开，函数可能超过 clone 预算。该预算是防止代码增长的约束，不是盈利模型。**不能因为模型只看到更多算术／内存工作，就断定更大的 shape 一定更慢，或放大 clone 上限就一定更快。** 本轮没有调整这个阈值。

后端 cost policy 至少需要区分满包、尾包 active count、残留内部 mask、实际 memory realization 和有界代码增长；并与 local mapping、输出分块、CPU task grain 联合考虑。当前 arithmetic/memory work prior 在开关两边相同，本轮没有伪装成已实现校准。GQA 的组内重复 K/V 地址、PV 连续输出维与 QK 贡献维的访问差异仍需通用分析；不能按 `attention` 名字套固定常数。

## 证据与范围

`evidence.tar.xz` 保留全部 20 份 capture、实际 ORC objects/汇编、prepared dylibs/helper、输入、FP64 oracle、120 个完整 replay 输出、全部样本、命令和执行记录。`audit.py` 独立验证数据与配对计算；`provenance.json` 说明隔离构建与源码身份，`sources.tar.gz` 保存实际 capture 的相关源码，`SHA256SUMS` 校验归档。审计通过 20 个独立 NumPy FP64 oracle、120 个 replay 输出及 840 个时间样本；所有配对输出逐 byte 相同，四个未触发控制的 LLVM／ORC 对象也逐 byte 相同。

新增三组通用代码生成回归覆盖 predicated memory、1D block 配合 2D/3D dispatch、block coalescing 的组合。首轮测试未达到循环内条件访存的特化准入，失败日志保留；修正为默认 standalone diamond 与显式启用现有 predicated-memory-effects 的 counted loops 后，完整相关四项 CTest 通过（142.14 s，包括同时开发的独立 MMA 候选测试）。这是覆盖构造的修正，未放宽 `clone == 1`、有效地址或数值断言。

完整输出之外，guard／输入不变／workspace 检查由 C++ helper 在执行时验证，其 guard payload 不单独归档，不冒充可离线重算的原始数据。全部时间属于独立 CPU 实验；此前发生 GPU hang 的 cohort 仍然无效，本轮没有恢复 Metal 计时、测 Torch，或宣称完成全算子性能目标。

复现入口是归档内 `run.py` 的 capture/replay 两阶段（默认四个诊断点，`--heldout` 为六个扩展点）；需要将 ROOT、BUILD、OUT 改为本机新目录，保留命令与新产物。原始路径仅作 provenance，不应覆盖旧结果。
