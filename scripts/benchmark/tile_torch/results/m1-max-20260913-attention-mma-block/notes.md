# 通用 MMA 输出寄存器分块：实现、验证与负面交互

2026-09-13，Apple M1 Max，LLVM 22.1.8，FP32 precise。**有界 opt-in 候选已实现；没有改默认，也没有声称自动 planner 或 Torch/MPS 性能目标完成。**

## 实现与语义

`PlannerOptions::mma_output_block` / `LowerOptions::mma_output_block` 提供当前有界候选族 1／2／4，默认 1 保留原路径。它是编译器候选集合，不是 DSL 语义或硬件宽度限制。`ExecutionPlan` 将请求值提供给 backend admission/cost policy；SIMD 和 Metal4 都从选中 plan 传给 XIR lowering，metadata 分别报告 requested width、实际 `blocked_mmas` 和 `mma_blocking_cost=unmodeled`。

准入只看 typed contraction 的输出最内非 unit 维：一侧输入对此维广播，另一侧在逻辑 Tile 中 stride=1；不检查算子名、轴名或 GQA head 数。PV 和 row-major GEMM 可符合，QK 的 key 输出方向通常有通道步幅，保留原路径。当前候选仅支持 complete-program lanes；显式完全展开的诊断路径、布局不符或单块展开预算不足均回退 reference。

每次沿输出方向维护 R 个独立 accumulators，K 仍按原贡献维顺序展开或循环。每个 K 只读一次广播项，更新 R 个连续输出；保留原 lhs/rhs 乘数位置及逐输出 MUL→ADD，不做 K 分片、归约树修改、额外 FMA 或精度改变。小结果仍走 SSA／原有 definition snapshot，大结果仍只分配原结果 snapshot。静态资源分析与 emitter 使用同一 admission plan，没有按“少 load 应更快”修改成本先验。

## 纯 native-entry 结果：不能独立选择分块与特化

两种固定 shape：decode `1,8,2,1,2053,80,96`，block 1×16；prefill `1,4,2,32,65,32,32`，block 4×16。QK/PV 均为 MMA，W8/block32/local1、融合、数学策略和输入完全固定。P 为 full-packet specialization 的**请求开关**，不是保证 clone 已生成。

12 captures，8 个 R2／R4 对 R1 的独立 ABBA 实验；每实验 3 cycles、12 visits、每 visit 7 样本，warmup 30 ms、样本目标 15 ms。单线程真实 ORC-entry host-wall，计入入口／launch reset／block 遍历／编译器内部调用，排除 Runtime、Python、JIT、线程池、调用方分配与验证。与上一批一样不是硬件 cycle counter，也没有新测 Torch。

单位 µs，时间是 visit p50 的中位数；比率为六组配对比率的中位数。每行重新测 R1，不跨行拼比率。

| case | P | candidate | R1 | candidate | 配对 candidate/R1 | 六对范围 |
|---|---|---|---:|---:|---:|---:|
| decode | off | R2 | 3181.776 | 3037.479 | 0.9533 | 0.9491–0.9586 |
| decode | off | R4 | 3188.930 | 3146.102 | 0.9839 | 0.9692–1.0346 |
| decode | on | R2 | 2038.232 | 2033.742 | 0.9957 | 0.9699–1.0164 |
| decode | on | R4 | 2042.531 | 3140.073 | 1.5392 | 1.5195–1.5572 |
| prefill | off | R2 | 142.084 | 141.091 | 0.9903 | 0.9823–1.0020 |
| prefill | off | R4 | 141.760 | 140.626 | 0.9919 | 0.9775–1.0014 |
| prefill | on | R2 | 141.640 | 140.738 | 0.9938 | 0.9859–1.0068 |
| prefill | on | R4 | 141.687 | 140.670 | 0.9958 | 0.9691–0.9982 |

R2 仅在未特化 decode 上表现出约 4.7% 的一致改善；加入满包特化后几乎没有额外收益。Prefill 的变化约 1%，不作为稳定优化结论。R4 + P-on 的约 **54% 回退**保留，不能只展示 R2 的正面数据。

原因的直接证据是准入变化：decode R1 clone=3343 条优化前 LLVM 指令，R2=3740；R4 原函数为 4560 条，超过既有 4096 clone 预算，因此 P-on 也回退动态 mask。三者静态 snapshot 都是 12864 bytes/worker、8 allocations、102912 bytes packet workspace；不是增加 snapshot 容量导致。没有放大预算来挽救结果。指令数量／mask／代码拓扑说明通用缺口，但不是各项耗时的动态 profile 分解。

这支持**联合候选选择**而非预设单调成本：`output block × full/tail realization × bounded code growth × memory/layout`。现有 target policy 能看到 requested block，但默认 analytic prior 保持原工作量，未校准或自动搜索这些选项；不把实验的 4.7% 当通用系数。首次只测两种 attention shape，尚无非 attention 性能泛化证明。

## 正确性与实现验证

完整构建成功，相关四项 CTest 全通过（142.14 s）：SIMD LLVM codegen、XIR target info、SIMD Runtime、SIMD LLM。不是整仓所有测试通过；两个既有 emitted-SSA budget recoverability 测试的问题未在本次修复。

- 新 host tests：广播左／右输入、不连续 QK fallback、动态小输出读取，覆盖 `(M,N,K)=(2,5,3),(2,35,17),(1,7,65)`；R1/2/4 的实际 alloca、资源分析、planner 一致，容量与 prior 不变；默认／fallback XIR 一致；预算不足回退；非法参数及 BF16 accumulator recoverably 拒绝。
- 新 Runtime tests：三 shape × 交换乘数两种 × R1/2/4，共 18 次编译／dispatch。先 load A/C，再覆盖其 views，验证旧 Tile snapshot 仍正确；与明确非融合的 FP32 顺序 oracle **逐 bit 比对**，检查三个 buffer 的 guards、B 不变和 A 的显式写入。
- 补强 FMA 反例后再次完整构建并运行 Runtime suite，通过（61.09 s）：`(1+2^-23)*(1-2^-23) + (-1)` 的 separate-MUL→ADD 为 0，FMA 为 `-2^-46`。其余贡献为零，避免后续值掩盖差异；两个操作数顺序均保留逐位断言。该改动仅在测试输入，不改变已冻结的性能对象。
- 本轮 12 captures 和 96 native replay visits 对完整 FP64 oracle 检查。核心代码经独立只读审查，未发现确认的语义／分配问题。尚缺 zero-K、trailing-unit、全尾输出块和动态小输出 extract 的专门 Runtime 执行覆盖，不能说测试穷尽。
- 八个变更 C++ translation units 的 clangd/tidy 检查均无 error；保留 warning 日志，不声称零告警。格式检查及仓库 no-throw 检查通过。Metal4 option forwarding 已编译，GPU 运行／性能 **NotRun**；之前 hang 的队列健康尚未确认。

## 复现与证据

Benchmark 控制：`LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK=1|2|4`，配合既有 full-packet enable／disable 开关。仅 `benchmark_tile_xir` 的 llm/rank diagnostic 入口读取此环境变量；正常 API 通过 PlannerOptions 配置，不在生产 planner 中按 kernel 名或环境特判。

`evidence.tar.xz` 保留 captures、真实 `.o`／汇编／dylib、全部输入／oracle／输出、计时样本、runner、测试和诊断日志；`audit.py` 独立重算数值、身份和配对结果。`sources.tar.gz` 与 `provenance.json` 记录实际冻结源和工具／二进制身份。Guard payload 未单独存档，guard 与输入不变是 C++ 执行记录，不冒充可离线重算的 payload。

独立审计通过 12 个 FP64 oracle、96 个 replay 输出、672 个计时样本和 691 个带证据链文件；全部 R2/R4 输出与 R1 逐 byte 相同。四个 R1 的 LLVM／ORC 对象也与前一批冻结 baseline 逐 byte 相同，没有混入未记录的默认优化。审计命令为 `python audit.py NEW_EXTRACTED_DIRECTORY PREVIOUS_FULL_PACKET_EXTRACTED_DIRECTORY`；前一目录来自相邻 [full-packet 归档](../m1-max-20260913-attention-full-packet/notes.md)，审计不运行 native code。

原始目录 `/tmp/luisa-attention-mma-block.PNrrvz`；归档 `run.py` 提供 capture／replay 两阶段。复现须改用新 ROOT/BUILD/OUT 路径，不能覆盖原始数据。桌面共活动未控制到静默状态，1 分钟 load average 从约 16.28 降至 13.02；六对范围不是置信区间。
