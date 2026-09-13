# Attention execution mapping：现状、缺口与有界实验

更新日期：2026-09-14。范围：源码静态审查与已归档实验；初稿为只读审查，后续 CPU 实验与模型修正见第 8–20 节，Metal 复测与 reduction 诊断见第 21 节。**未宣称已完成自动生产优化**。
它是实现侧工作记录，不替代既有设计文档；不改动受保护的 matrix-initializer WIP。

## 1. 先分清三种状态

| 状态 | 已有内容与边界 | 源码入口 |
|---|---|---|
| 已实现 | TIRx 可把自动 parallel program 映射到 cooperative group；矩阵候选枚举 subgroup 整数因子，验证覆盖、fragment/shared 容量，并按可替换 cost policy 打分。它是有限候选族的求解器，不是任意 phase 的完整映射求解器。 | [planner.h](../../include/luisa/tile/bridge/tirx/planner.h)、[plan_group](bridge/tirx/planner.cpp) |
| 已实现 | 支持有数值许可的 closed reduction、只读输入 forwarding、部分 barrier/accumulator 优化；这些能力不意味着未匹配的 contraction 已有贡献维搜索。 | [cooperative.cpp](bridge/tirx/cooperative.cpp)、[reduction.cpp](bridge/tirx/reduction.cpp) |
| 已实现 | XIR bridge 的 target info / cost policy 可由后端提供；当前 local 分布候选仍受 packet-local 合法性限制。SIMD 的 whole-program 向量化不等于单个 attention program 内 QK/PV 的联合分布。 | [XIR planner.h](../../include/luisa/tile/bridge/xir/planner.h)、[planner.cpp](bridge/xir/planner.cpp)、[lower.cpp](bridge/xir/lower.cpp) |
| 受保护 WIP | `materialize_matrix_initializers` 把纯的非平凡 MMA 初值变成独立 C tile，使普通 matcher 有机会接收 `acc * alpha`；计入元素工作与 shared 存储，失败后重试原表示。不是新数学许可，也不是已验证收益。 | [matrix.cpp](bridge/tirx/matrix.cpp) 的 `MatrixInitializerMaterializer`；[cooperative.cpp](bridge/tirx/cooperative.cpp) 的候选重试；[tests](../tests/unit/tile/bridge/test_tirx_matrix.cpp) |
| benchmark-only，PV 尚无 GPU 性能结果 | QK、PV 可独立选择 `mma` 或 `reduce`；默认二者仍为 `mma`。PV probe 是 `acc * alpha + reduce(probability * value, n, add)`，不是生产 planner 决策。 | [fixture](../tests/common/tile_llm_test_utils.h)、[benchmark entry](../tests/benchmark/benchmark_tile_tirx.cpp)、[driver](../../scripts/benchmark/tile_torch/compare_llm.py) |

当前 fixture 是 FP32、连续 KV、bottom-right causal GQA：`kh = h / (Hq/Hkv)`，合法 key 为 `k <= q + K - Q`。
同一 online-softmax 程序覆盖 prefill/decode；`row_max/row_sum/acc` 是跨 KV block 的循环携带状态。

## 2. 一个 phase 图就能看出的结构问题

```text
parallel(B, Hq, query_block): load Q，初始化在线状态
  └─ pipeline(KV blocks): load K/V → QK [输出 q×k；贡献 d]
                                          ↓ layout 转换 / mask
                                     max → exp → sum
                                          ↓ layout 转换 / α 广播
                                     PV [输出 q×dv；贡献 k]
                                          └─ 更新 (max, sum, acc) → 下一 KV block
```

箭头不仅表示数据依赖，还可能需要 shared 写读、广播、fence 和等待；`pipeline` 声明不自动证明硬件重叠。
QK 的 K 通常沿 `d` 连续，PV 的 V 通常沿 `dv` 连续；两者相同的 contraction 语义不要求相同的 lane 方向。

## 3. 量化 prior：历史证据，不是当前 revision 的成绩

下表均为 Apple M1 Max 的 FP32、bottom-right causal GQA，Tile 路径是 TIRx→MSL→Metal；单位 µs，shape 顺序为 `B,Hq,Hkv,Q,K,D,Dv`。
**计时是无 counter 的 command-buffer GPU interval / invocation（吞吐口径）**，含 buffer 内空隙，不含 host 编码；不是纯 kernel 时间。
时间为各 round 中位数的中位数，比值为同 round Tile/Torch 比值的中位数，不用表中两个中位数相除替代。

| 历史 cohort / 配置 | shape | Tile µs | Torch µs | Tile/Torch |
|---|---|---:|---:|---:|
| 09-07 cooperative prefill，block 8×16 | 1,4,2,64,128,64,64 | 80.612 | 27.876 | 2.894× |
| 09-08 QK-reduce probe，block 1×32，1024 threads | 1,8,2,1,2048,64,64 | 363.985 | 39.372 | 9.245× |
| 同上，KV 尾块与不同通道宽度 | 1,8,2,1,2053,80,96 | 519.987 | 136.924 | 3.798× |
| 同上，更长 KV / 更宽通道 | 1,16,4,1,4096,128,128 | 730.416 | 75.771 | 9.640× |

Prefill 是 6 个平衡顺序 round；decode 每配置 4 rounds，各配置之间未交错，不能作配对因果 A/B。
Torch 是 functional SDPA，包含输出/内部临时分配，mask 构造在计时外；Tile 输出预分配。不是独立 MPS/MPP 或 compiled-Torch 对照。
历史 Tile 使用 `fast_math=false` 与 `unordered_tree`；Torch 使用默认数学策略，不能假定两者浮点求值顺序相同。
原始数据：[prefill/results.json](../../scripts/benchmark/tile_torch/results/m1-max-20260907-cooperative-programs/prefill/results.json)、[decode/results.json](../../scripts/benchmark/tile_torch/results/m1-max-20260908-composed-reduction/qk-reduce-1024/results.json)。
协议与审计：[09-07 notes](../../scripts/benchmark/tile_torch/results/m1-max-20260907-cooperative-programs/notes.md)、[09-08 notes](../../scripts/benchmark/tile_torch/results/m1-max-20260908-composed-reduction/notes.md)、[audit.json](../../scripts/benchmark/tile_torch/results/m1-max-20260908-composed-reduction/audit.json)。

历史 SIMD attention 只有另一批 Runtime E2E 对照：`1,4,2,16,32,16,16` 为 42.067→42.945 µs，`1,4,1,64,128,32,32` 为 108.037→111.292 µs。
这是 cohort-private off/on、local=1，配对比约 1.022/1.030，范围跨过 1；不是稳定收益，也没有该批 attention 的纯 native-entry/Torch 对照。
见 [SIMD tables](../../scripts/benchmark/tile_torch/results/m1-max-20260909-cohort-private/tables.md)。不能将上述数值跨硬件路径、计时口径或 cohort 排名。

## 4. 源码指向的通用映射缺口

1. `matrix_extent()` 要求正的 8 倍数 domain；decode 的 Q=1 不能使用此 atom matcher。未匹配 contraction 进入普通元素路径。
2. `GroupWorkloadAnalysis` 对未匹配域累计 `domain.count × executions`；这没有完整计入其内部串行贡献维工作。`plan_group()` 在 matrices 为空时保留参考宽度，不进行矩阵族的联合候选搜索。
3. 历史 [decode Metal source](../../scripts/benchmark/tile_torch/results/m1-max-20260908-composed-reduction/qk-reduce-1024/attention-1x8x2x1x2048x64x64-r0-native.metal) 中，QK 使用 32 个 subgroup 分摊 32 个输出，每 lane 处理 2 个 channel 后 `simd_sum`；PV 仍仅 64 个线程各自串行扫 32 个 key。
4. 同一 source 的 mask/softmax/标量阶段只用 32 或 1 个线程，却保留多处 shared 临时与全 group barrier。它解释“为什么需要 phase 成本”，不是已测出的 bank-conflict、occupancy 或 cache 根因。
5. 历史 QK probe 在 1024 threads 描述性改善、在 64 threads 全部变慢：每输出固定一个 32-lane subgroup 时，64 threads 要分 16 批完成 32 个输出。仅调全局线程数或仅奖励连续访问解释不了整个程序的权衡。

上述结论来自 typed contraction、域、访问 layout 和依赖分析；无需识别 `llm_attention` 名字，也不能把所有 `mma` 机械替换为 reduction。

## 5. 待实现的候选与成本表达

每 phase 的候选可写成 `q_s = (output_partition, reduction_partition, vector_width, ownership_layout, resource_choice)`。
先用 `o_s × r_s <= T` 描述参与线程预算，再细化输出分批、每线程局部累积及完整覆盖；这里是设计提案，不是现有 API 或完备性证明。
同一程序允许 QK/PV 选择不同 `q_s`，但共享 group 大小、峰值 live state、资源容量与依赖约束。

- 合法性：依赖 primitive 的独立性契约，验证覆盖/边界/别名/同步可实现性；归约重排必须满足数值策略。`unordered_tree` 允许重排，不代表 FP32 精确结合。
- 服务成本：贡献维 issue rounds、输出批次数、collective 次数、实际 access composition 的读取宽度/步幅、live temporary 与同步；IR bytes 不能冒充 DRAM bytes，逻辑 live scalar 不能冒充物理寄存器数。
- 转换成本：`q_s → q_(s+1)` 的寄存器交换/shared 往返、广播、barrier 和 pipeline slot 生命周期；不把每个 phase 独立最优的结果直接拼接。
- 初期无重叠目标：`min Σ service_s(q_s) + Σ transition_(s,s+1)(q_s,q_(s+1))`，保留合法参考候选；有实际重叠后改求资源受限依赖图 makespan，不能先假定所有 phase 并发。
- 可先有限枚举并按资源/成本支配剪枝；系数留给 backend cost policy 校准，solver 与 target info 解耦。`r=1/2/4/8/16/32` 是候选研究方向，subwarp emitter 尚不能当作现成能力。

GQA 还暴露跨 query-head 的 KV 复用机会，但语义上的同一 KV head 不自动带来 cache 命中或零成本广播。
长 KV 的 split-KV 是另一层候选：必须实现 online 状态 `(max, sum, acc)` 的合法合并、额外存储/launch 成本；当前记录不把它算作已支持优化。

## 6. 两条单变量实验，不混进生产默认

**Prefill：initializer WIP off/on。** 同一冻结构建，固定 64 threads、block 8×16、QK/PV 都为 MMA，其余 view/reduction/prefetch 策略完全一致。
先检查 generated source 是否确实多一个 PV tensorized contraction，再检查新增 initializer/shared 流量和容量 fallback；“成功匹配”不等于“执行更快”。

**Decode：QK/PV 独立分解 probe。** 在固定 threads、block、forwarding、closed-collective 和数学策略下，先固定 QK，只切 PV；随后补齐 `(mma,mma)/(reduce,mma)/(mma,reduce)/(reduce,reduce)` 四格检查交互。
初稿时 `--attention-pv reduce` 刚加入、没有性能结果；同日 CPU 结果见第 8 节，GPU arm 仍未完成。默认仍走 MMA，metadata 必须确认开关确实生效，不能只信命令行。
补充执行状态：四个新增 SIMD 数值检查已通过；同日长 KV 的既有 QK-reduce/PV-MMA control 出现两次超时和 MPS command-buffer GPU Hang，整轮性能不接受，PV-reduce GPU arm 未运行。见 [独立执行检查点](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-pv/notes.md)，不能用部分 PASS 消除驱动错误。
先看 PV 是否成为贡献维 collective、product 是否被内联而非巨大中间 tile，再测完整程序。
若 probe 变快，说明候选空间值得扩展，但不能证明所有 contraction 都适用；若变慢，区分未触发、访存步幅恶化、输出分批过多、重分布/同步增加。
若两种表示生成等价映射，则本轮只验证了表示等价，没有检验新的物理分布；失败或无收益也必须留档。

## 7. 固定验证集与 held-out 矩阵

下列是**计划矩阵**，不是已完成测试清单。先跑小型正确性，再逐级扩大；held-out 不参与最初规则选择。

| 角色 | shape / 变体 | 主要排除的过拟合 |
|---|---|---|
| Prefill 诊断点 | 1,4,2,64,128,64,64；block 8×16 | initializer 与 PV tensorization |
| Prefill held-out | 1,4,2,65,131,64,64；1,8,2,256,1024,64,64 | Q/K 尾块、较大 workload |
| Decode 历史锚点 | 表中三个 Q=1 shape；block 1×32 | 标准、ragged、不同 D/Dv |
| 小 Q / mask held-out | 1,8,2,3,67,80,96；1,4,2,65,65,64,64 | bottom-right 偏移、真正 triangular mask、尾块 |
| GQA / batch held-out | 2,8,1,1,2053,80,96；1,8,8,1,2048,64,64 | MQA、MHA、batch，不只固定 Hq/Hkv=4 |
| 长 KV held-out | 1,16,4,1,8193,128,128；1,8,2,1,32768,64,96 | 循环长度、尾块、带宽/状态成本外推 |
| 非 attention 反例 | dot/GEMV、小矩阵、转置/带步幅 contraction | 以 domain/access 为规则，而非 kernel 名字 |

当前 helper 只提供 bottom-right causal mask；noncausal、任意 mask、全 masked row、paged KV 需先扩展语义与独立 oracle，不能冒充已覆盖。
每格保留完整 FP64 oracle 输出检查、guard、数值策略、源码/二进制/依赖指纹；Error/NotRun 不删，不能用通过的子集补完整矩阵。
计时分列无插桩 CB throughput、插桩 dispatch/compute-pass 诊断、Runtime E2E batch/latency；SIMD 另列 native-entry，绝不跨口径替代。
实验前确认 GPU/队列健康且无并行计时干扰；采用同构建交错 A/B，不以最佳单次或不同日期的两个数宣布收益。

## 8. 同日后续：native-entry 探针与 collector 审查

[SIMD 纯 native-entry 实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-native/notes.md)完成两种尺寸的四种 QK/PV 表示及 72 个 ABBA visits。PV→reduce 在两个尺寸均变慢，不能作为生产统一改写。PV 两边 snapshot 总字节相同；现有数据未证明回退是容量增加导致，仍需指令/访存组织分析。GPU arm 仍未重跑。

进一步定位：`GroupWorkloadAnalysis` 在 `cooperative.cpp`，未匹配 MMA 只累计 output-domain × 外层 executions；进入贡献循环时 lane-depth 已增加，K 信息不再上报。`plan_group()` 接收的聚合字段无法反推 K，且 matrices 为空时直接返回 reference。即使修正 aggregate，SIMDGROUP_REFERENCE 的公共 element term 对线程候选也只是同一常数，不能冒充新的映射搜索。

可分两步：先为保留 MMA annotation 的 canonical scalar contraction 记录 `O × E × (1 + product(K))` 初始化/更新工作 prior，同时保持输出并行度为 O，防止 double-count 与溢出；再引入真正的分阶段输出/贡献维候选及 transition cost。第一步只修成本信息，不应声称改变 emitter 或达到性能提升。接入点与现有 protected WIP 重叠，本轮未改。

初稿只读；本节补充同日独立实验，生产代码与 protected WIP 仍保持不动。

## 9. SIMD：实际 mask realization 比抽象 work count 更重要

[十配置纯 native-entry 实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-full-packet/notes.md)把已有 full-packet 特化候选独立应用于 attention：固定 DSL、QK/PV MMA、W8/block32/local1，五个有效特化配置的配对时间减少 24%–72%，包括 8193 KV、batch/MQA 和 mixed tail。其余五个未生成 clone／没有满包的控制也完整保留，没有新的 Torch/MPS 对照。

这是代码生成候选的效果，不是自动 planner 优化：当前 prior 无法区分 P0/P1。未来需把 `full/tail packet × residual mask × memory realization × code-size budget` 接入 target policy，不能只按 Tile 算术和逻辑流量排名，也不能把满包误认为内部访问均无 bounds mask。

下一项通用候选是沿连续输出维的 MMA 寄存器分块：若一个输入对该维广播、另一个输入连续，交错多个独立输出的累积可共享广播项；保持每个输出的贡献顺序与 MUL→ADD，不等同于归约重排。PV 与普通 row-major GEMM 可符合，QK 的输出 key 通常有通道步幅，应由访问分析决定而非函数名。其同日实现与实验见下一节。

## 10. 输出分块已落，但收益不能独立相加

[实现与 96 个 native-entry visits](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-block/notes.md)提供 opt-in R1/2/4，默认 R1 不变。共享 admission／resource plan，逐输出 K 顺序和原 snapshot 不变；选择字段传入 `ExecutionPlan`，backend cost policy 可查看，默认 prior 暂不降价。SIMD 已执行，Metal4 forwarding 已编译但 GPU 尚未测量。

R2 在未特化 decode 上约快 4.7%，特化后增益近于零；R4 使函数超过原 clone 预算，P-on 也不能生成满包路径，整程序反而慢约 54%。静态 snapshot 容量未增加。这是实测的候选交互，不是算子特判或完备 cost model 的证明：求解器需要同时评价代码规模、mask realization 和数据访问，不应把多个局部收益系数相乘。

## 11. 贡献循环与数据表示必须联合决定

[独立 MMA cap 实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-roll/notes.md)固定全局 Tile 阈值64，仅额外限制 MMA 的 K 展开。cap8 恢复 R4 decode 的满包 clone，P-on 时间减少约36%；但 MHA D64 退化到约4.8倍，prefill约慢43%。这不是新的最优 decode 成绩：之前 R1 满包路径已经约2 ms。所有负例保留，默认 cap0 不变。

一个直接的 representation 缺口是：新保留的动态 K 循环仍从旧的小 SSA Tile 读取，可能生成线性 SELECT 链；MHA/PV 的小输出域又可展开成很多独立循环。静态代码支持这些机制，但不能把全部耗时分别归因给某一机制，也不能用原先不变的 snapshot 容量推断物理寄存器或 spill 不变。

因此合法候选不只是 `K_unroll`，而是 `(K_loop, output_block, operand_representation, packet_realization)`：动态访问需要选择可索引 snapshot 或其他合法实现；新增数组在 SSA 定义处捕获，不能延迟到消费端重新读外部 memory。资源容量先准入，随后由 backend policy 评价存储访问、循环/代码增长及满包/尾包路径的组合成本。相同 primitive 与 access layout 的普通 GEMV/GEMM 也适用，不需要 attention 名字或额外 DSL 实体。

[联合 snapshot 实现与后续实测](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-indexable/notes.md)已落到共享plan：MHA每worker新增Q64的256 B（P已有snapshot），默认六个LLVM/ORC对象逐字节不变。新72个visits中，MHA cap8只比同批默认慢约7%–10%，prefill仍慢约43%；两批不构成同轮V1/V2配对。下一步是已有输出分组的合法候选扩展：broadcast-LHS配上strided-RHS也能保持各输出K顺序，步幅应进入成本而不是被当作不合法。任何候选仍须以完整程序测量，而非只看循环数量。

[带步幅RHS输出分组](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-strided/notes.md)随后实现并通过四项回归。72个新visits显示R4相对同批R1：ragged decode约少3.5%时间，cap0 prefill约少4.3%；cap8 prefill约少21%，但仍慢于展开版本，MHA增益很小。两边容量和逐bit结果一致，不给未校准cost prior虚构折扣。

这些小改进也指向后续的结构性缺口，但不构成性能上限证明：当前 `packet_local_program` 只准入有限的一维程序族，不支持任意MMA phase ownership；`_mma`在bridge中降为标量MUL/ADD循环，后端只剩SIMT/SSA实现。应在这一步之前，以typed contraction保留输出/贡献域、广播/步幅、输入及accumulator类型和 `MmaPolicy`，再选择phase mapping。默认MMA允许reassociation但不允许偷偷降精度；严格策略保留ordered fallback。硬件/BLAS候选还必须计入packing、snapshot alias、转换及调用开销，不能用外部库名字代替合法性和成本分析。

## 12. 成本提取与生成方案对齐，不把工作量下降当作实测加速

[MMA work checkpoint](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-cost/notes.md)修正了 XIR planner 仍按旧展开路径计费的问题：现在读取与容量分析共用相同 representation options，并使用实际准入的 `mma_emission_plan`，而非直接相信 requested block。

设一次逻辑程序内输出数为 O、贡献数为 K、沿输出内轴的长度为 N、准入分块宽度为 R，则非空分组的组数 `G=(O/N)×ceil_div(N,R)`。广播侧的投影读取为 `G×K`，另一侧为 `O×K`，乘加仍为 `O×K`；尾组不能用 `O/R` 截断。未准入分组时 G=O，空输出时 G=0。外层 scope 的执行次数乘入所有动态计数；新增快照写入按 SSA 定义执行次数计，不按 consumer 数重复收取。

cap 导致的小 Tile 索引快照现在计入定义写入和动态读取。常量索引仍优先使用原 SSA 元素，即使该值同时有 snapshot；MMA 的动态性按操作数实际依赖的输出/贡献轴区分，不能因为输出循环是动态的，就向与输出轴无关的广播项收取数组读取。小 CONSTANT 列表仍可能在动态索引下生成 SELECT 链，不能与 large-Tile SPLAT 混淆。

`ExecutionWork::mma_per_packet` 向 backend cost policy 提供不带系数的乘加、左右输入/seed 投影、runtime K 循环调用与迭代计数。CPU 单 worker、多 worker 调度均保留这些字段；它们不是硬件 load/DRAM transaction、静态指令数或 ns，也不应和现有 weighted prior 双重收费。普通 GEMV/GEMM 同样适用，没有 attention 名字匹配或新 DSL 实体。

本轮是模型一致性修复，**没有新增性能测量，也没有开启自动 R/cap 搜索**。四项 CTest 与七项精确筛选的 host 回归通过；新测试含 441 个 planner 配置、零 K/零输出、交换/转置/尾组、重复 scope、常量 SELECT 和 budget fallback。原来的两个 host SSA-budget fatal 问题未在本轮解决，不能声称全仓测试绿色。

下一候选可在不切分 K、最多四个累加器的约束下比较每个 MMA 的 `1×4` 与 `2×2` 输出微块。若两个输出轴分别由左右输入使用、另一侧广播，完整 `2×2` 微块每 K 可用四次投影服务四个输出，`1×4` 则为五次；decode Q=1 仍需要一维候选。此处仅提出候选：小结果必须按原 flat index 回填，保留 carry/alias 与逐输出 K 顺序；还需验证代码规模、mask、真实 stride 和完整 kernel 的时间，不能直接凭投影数自动选定胜者。

## 13. 二维输出分块：已验证的通用候选，尚非自动选择策略

[二维 MMA 实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-2d/notes.md)已实现上述候选：`enable_mma_2d_blocking` 默认关闭，在 requested R4 的四累加器预算内，允许两个互补广播输出轴使用 `2×2`，否则保留原一维方案。R1/R2 不受影响；不识别 attention/GEMM 名字，不增加 DSL primitive，也不切分或重排 K。前导 batch 和中间/尾部 singleton 不是新的分块轴。

```text
同一 typed contraction、逐输出相同 K 次序
                 │
      输出轴 / 广播 / expansion budget 准入
                 │
       ┌─────────┴─────────┐
       │                   │
     1×4                 2×2
  一行四个输出        两行 × 两列输出
       │                   │
       └─────────┬─────────┘
                 │
  相同 flat 输出 ABI、carry 与 definition-time snapshot
                 │
     backend 代码生成 → 完整 native-entry 验证
```

设输出为 `B×M×N`，贡献数为 K，二维候选的动态组数为
`G=B×ceil_div(M,2)×ceil_div(N,2)`；行操作数投影为
`B×M×ceil_div(N,2)×K`，列操作数为
`B×N×ceil_div(M,2)×K`。交换操作数时交换左右计数，乘加数仍为
`B×M×N×K`。工作提取与 lowering 共用准入结果，包含尾块；不把请求值当作实际生成方案。

完整微块相对 `1×4` 的投影总数降低20%，但行侧读取增加、列侧减少，访问成本未必对称；静态循环层次和代码生成也不同。因而分别保留 lhs/rhs 计数，不用统一的20%系数预测 kernel 加速，更不把 snapshot 字节当成物理寄存器数。

六组预声明 attention 形状、12 captures、72 ABBA visits、504 samples 全部通过独立 FP64 与 A/B 逐 bit 输出检查。同批固定 R4/cap/P 下，五组 prefill/batch/GQA 的配对时间减少约2.2%–10.6%；decode 对照未准入二维，LLVM/ORC object 逐字节相同，比值范围跨1。五组 prefill 的满包 clone 均未生成，decode 两边均生成，因此本批收益不是由新增满包特化解释。这里是单线程实际 native-entry host-wall，不是 Runtime E2E 或新的 Torch/MPS/Metal4 对照。

四项相关 CTest 和七项精确名称 host 回归通过；新增96组 plan/lower 配置与60个严格数值 runtime dispatch（含负控制，16次实际准入二维），覆盖交换/转置、奇数尾块、零K、budget fallback、small-SSA carry 和较大 snapshot 输出。Metal4 forwarding 通过构建，GPU 执行仍未验证。

默认关闭且未加入自动 R/cap/二维搜索。后续应把这些候选纳入有资源约束、代码量/访存/phase 转换成本的联合选择，并用更多非 attention contraction 和 held-out 尺寸验证；这些局部收益不替代第5节的贡献维 ownership、split-KV 和跨 phase 映射能力。

## 14. 原生参照量化了更大的 phase realization 缺口

本节记录原生参照实验结束时的诊断与提案；后续编译器候选见第15节。

[六尺寸原生参照实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-native-reference/notes.md)将当前实际 Tile ORC 对象与两种**手写、benchmark-only** CPU 实现放入同一原生计时器：KV16 online NEON，以及整头 dense Accelerate GEMV/GEMM。144 visits / 1008 samples 全量保留，输入、独立 FP64 oracle、guard 和线程协议逐次检查。它们不是新的 Tile lowering，也不是 Torch/MPS 成绩。

三组 decode 的 NEON/Tile 配对时间比为0.189、0.151、0.237；Accelerate/Tile 为0.172、0.114、0.168，含 KV=8193。三组 prefill/batch 的差距也存在，完整数值见实验表。仅从这些数据不能把总差距归因于某个局部循环：两种参照都改变数据表示和求值顺序，online 还跳过全 masked 工作，dense 则物化完整 score。主机仍有用户/系统背景活动，不能用这批数值直接拟合微小成本系数。

### 14.1 保留 contraction，才能选择不同的 phase 实现

目前 [`_mma`](bridge/xir/lower.cpp) 仍掌握 typed axes、贡献域和 operand projection，随后降成 `_fold` / `_fold_many` 的标量乘加；到了 SIMD Schedule 再猜矩阵意图已太晚。现有小型 `MATRIX_LINALG_MUL` 也不是任意尺寸 BLAS 接口。应在这个边界选择实现，而不是识别 attention 名称。

```text
                typed MMA + math permission + access layout
                               │
             shared admission / resource / realization plan
                    ┌──────────┴──────────┐
             scalar/vector fallback   backend-owned MMA leaf
                    │                 QK: vectorize contribution D
                    │                 PV: vectorize output Dv
                    └──────────┬──────────┘
                    snapshot/layout transition costs
                               │
                       complete phase-graph choice
```

拟议的每 MMA descriptor 保留 batch/M/N/K 轴、dtype/accumulator type、各 operand stride/projection、math permission、结果 ownership、snapshot/packing 容量。后端可选择合法实现，默认保留当前 fallback；resource analysis、lowering 与 cost policy 消费同一个计划，不能 emitter 临时加数组。首版可限制静态 FP32、一个贡献轴、`local_lanes=1` 和同步单线程调用，之后扩展的是 realization 能力，不是 DSL primitive 数量。

普通 XIR `CallInst` 当前不被 [SIMD schedule lowering](../backends/simd/schedule/xir_to_schedule.cpp) 接收。已有 launch-record callback 可借鉴，但不等于已支持 BLAS。若走 leaf-call，应使用 compiler-owned typed registry 和 backend/shader-owned descriptor 生命周期，仅准入已注册签名，其余调用继续拒绝；不要开放任意未解析外部符号。

### 14.2 数学权限与物理 snapshot 是两道独立约束

- 每个 MMA 必须直接检查 `MmaPolicy`。`allow_reassociation=false` 的候选不能偷偷改为 BLAS/FMA/tree，且任何策略都不能降输入精度。目前 kernel-wide fast-math guard 的 `OrderedReductionAnalysis` 只检查 REDUCE，不能用它代替 MMA 权限验证。探针的输入仅是受控、完整数值验证通过的样本；有限 FP32 仍可能 dot 溢出，不能据此证明所有输入等价。
- leaf 读取定义时捕获的 SSA snapshot，不能在消费端重读可能已被修改的用户 buffer；seed 只读，输出写新鲜 storage，零贡献域保留 seed。现有 alias/snapshot 测试必须继续适用。
- 现有 SIMD private-array interleave 要求封闭的 GEP/load/store 地址使用；把地址传入调用会改变准入。逐 program 连续数组与跨 program interleave 是不同物理 layout，packing、额外 live state 以及其它消费者的访问退化都必须计入，不能假设传一个指针就免费兼容。

### 14.3 求解器要比较实现及边界成本，而不是给 BLAS 一个奖励系数

对 phase `s` 的候选 `q_s`，扩展第5节的目标为：

```text
C_s(q_s) = calls_s · call_cost(q_s)
         + pack_cost(bytes_s, source_layout, leaf_layout)
         + compute_cost(shape_s, dtype_s, math_s, realization_s)
         + local_state_cost(q_s)

min Σ_s C_s(q_s) + Σ_(s→t) transition(q_s, q_t)
s.t. coverage / dependencies / math permissions / live-memory capacities
```

这是待实现的成本分解，不是已校准公式；pack 与 transition 的归属须唯一，避免重复计数。backend policy 提供实现能力与系数，solver 只负责合法候选的枚举/剪枝/组合。现有 `ExecutionMmaWork` 可继续提供工作量，但需补每 MMA 的 shape/stride、call 数、packing bytes、实际实现 ID。

尤其不能把本实验的“整头 BLAS attention”比率直接赋给每个 KV16 小 MMA：后者调用更频繁，输入/seed/snapshot 边界更多，可能完全抵消算术收益。下一步是先实现窄范围、可拒绝的 typed leaf/vector candidate，验证普通 GEMV/strided contraction 与 attention 的 held-out 尺寸，再测完整 attention 的实际 native-entry；自动选择与 Metal phase 映射仍是后续工作。

## 15. 每 program 的 native MMA 向量候选

首版实现采用同一 LLVM module 内的私有 helper，而不是 Runtime callback、外部 BLAS 符号或 whole-attention rewrite。DSL 仍使用现有 `mma`；选择依据是输出轴、贡献轴、快照 stride 和局部数学权限，不识别 attention、QK、PV 或 dimension 名称。

```text
Tile MMA：axes + MmaPolicy
            │
     native_mma_plan（共享、纯分析）
       ┌────┴───────────────────┐
     不准入                    准入
  原 _fold/_fold_many      定义时 A/B/seed snapshots + 新 output
                               │
               typed XIR external + StridedMmaMD
                               │
               Schedule 内持有 descriptor 副本
                               │
          同 module 私有 LLVM helper；逐 active program 调用
                  ┌────────────┴────────────┐
            连续输出轴向量化           连续贡献轴向量化
            保持每输出 K 次序         要求允许 reassociation
```

候选暂限静态 FP32、一个贡献轴、`local_lanes=1`，由 backend target info 明确接受内部向量宽度。SIMD 首版提供2/4/8，默认 `native_mma_vector_width=0` 关闭；它不是 program packet width，也不是执行 hierarchy 的固定 lane 上限。该选项为固定实验候选，不是已经校准、自动求解出的最优选择。

输出轴需要一个输入广播、另一个及输出 unit stride；优先采用此方向，保持各输出 ascending K、separate MUL/ADD。否则在两个输入均 unit K stride 且 `MmaPolicy.allow_reassociation=true` 时沿贡献轴向量化，分组部分和再合并。部分和以真实乘积初始化，不能凭空插入改变 signed zero 的 `+0`。不启用 FMA、`nnan/ninf/nsz` 或低精度输入。贡献为空、尺寸太小或布局不匹配则回退。

快照仍在 SSA 定义处物化，不延迟重读用户内存；强制 storage 会覆盖 constant splat/deferred recipe 的选择，并进入同一资源分析。小结果在 helper 完成后读取为 SSA，保留现有 implicit carry 与常量投影路径；大结果保留 bounded storage。输出不得与输入别名。调用引用使对应数组退出 packet-interleaved private-array 准入，所以这项布局代价是真实候选的一部分，不是假定免费传入指针。

`StridedMmaMD` 是必需语义而非可忽略 hint：只允许位于 external function，clone、文本和 bitcode 保留所有字段；未知/丢失 metadata、错误模式、容量、引用类型或非本地 root allocation 均拒绝。Schedule 持有副本，不依赖原 XIR module 的生命周期。普通 external call 仍不被 SIMD 打开。`StrictMmaAnalysis` 与 REDUCE 分析独立，使后端的全 kernel fast-math 开关不能覆盖局部 strict MMA。

工作提取新增 native call、输出向量组和贡献向量组计数，保留 scalar-equivalent 乘加数及左右独立投影；输出标量尾部与首组乘积初始化单独影响循环计数。当前 prior **没有**标定 helper call、水平合并、private-layout 转换或代码膨胀成本；不能从这些工作量声称预测了 native 周期，更不能把第14节 BLAS 整体比率作为奖励系数。

完整选定构建、11项CTest和23个修改的C++ translation units的syntax检查通过。[六组实际编译器候选实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-native-mma/notes.md)的12 captures、72 ABBA visits、504 samples均通过完整FP64、输入/guard和本臂capture逐bit检查；两个臂之间不强求逐bit相同。

结果没有形成普遍的加速：batch GQA配对时间减少25.8%，其他五组增加6.2%–60.4%。所有on候选快照增加、packet-interleaved arrays减少；prefill-q8还从无满包clone变为一个。已生成正确的向量MUL/ADD，不代表完整phase的layout转换和快照代价可忽略。这是完整realization对照，不是单独算术方向的因果实验；各开销占比仍需profile，不能凭容量相关性当作已证明瓶颈。

因此继续保持默认关闭及`native_mma_cost=unmodeled`。下一步应联合比较phase realization与physical layout，明确producer/consumer转换、定义时snapshot、call与代码量的成本归属；重点验证能否保留packet布局或让相邻phase共用布局，再用非attention及held-out尺寸检查泛化性。此次没有新的Torch/MPS/Metal性能比较，也没有完成自动求解或整体性能目标。

MHA-on的实际反汇编已确认helper内联，kernel body内无调用指令；但16-output QK循环仍逐score重载同一Q。由此得到更具体的候选：把贡献向量化与有寄存器预算的输出分组组合，同时保留快照/布局边界的成本。仅凭私有函数出现在优化前LLVM中，不能把慢归因于调用；仅凭最终重复load，也不能跳过profile就宣称它解释了全部差距。

## 16. 原生采样：先优化搬运的执行映射，而不只优化 MMA

本节保留实现前的诊断检查点；连续快照复制的后续实现与测量见第17节。

[独立采样记录](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-phase-profile/notes.md)复用第15节冻结的实际 ORC object/dylib，没有重编译 LLVM 文本，也没有修改 kernel 或数学策略。MHA decode、长 KV decode、batch GQA prefill 各采样 off/on；每次均在独立进程内检查完整输出、FP64 reference、guard、输入和 launch metadata，要求逐 bit 复现本臂 capture。采样是热点诊断，不替换之前的 ABBA 性能数据。

MHA-on 的 K/V 定义快照搬运是明确的热点。关联到完整生成代码的两个循环分别为每个active program拷贝16×64个 FP32 元素，先沿 program packet 构造地址，执行逐 lane 的掩码标量 gather，再写逐 program 连续的快照。即使源 tile 沿 feature 连续，当前搬运也没有沿这个方向做连续向量访问。CPU 时间采样不是内存带宽/cache计数器，不能把热点直接称作 DRAM 带宽瓶颈；精确比例、未匹配样本及反汇编区间见记录。

长KV-on 的实际满包函数也保留了沿 element→program 的 K/V scalar copy。满包特化能消除这里的 lane mask，却不会自动把循环改成沿 tile 内连续元素向量搬运；两项候选应组合评估，而非把连续copy当作原满包特化的替代品。

### 16.1 相同的 Memory，搬运阶段也可以选择不同的 Execution

令 `p` 为独立 program，`e` 为该 program 内的 tile 元素。本例 source 中 `e` 连续、不同 `p` 的地址间距大；native MMA 要求的 snapshot 是 `[p][e]`。

```text
当前搬运：for e                         候选搬运：for p（保持 active 条件）
            packet(p): masked gather                 for contiguous e-vector
                       masked scatter                    guarded vector load
                              │                          vector snapshot store
                         snapshot[p][e]  ◀───────────────┘
                              │
                 QK / PV 的 per-program native vectors
```

这是 `(program, element)` 执行方向及访存实现的选择，不是让 memory 决定逻辑 hierarchy。沿 program 的 packet SIMD 对某些算术/交错快照很好，但不能机械套到每个 producer/consumer 边界；同一语义快照允许不同搬运计划。GQA 多个 program 共享 KV 还可能增加广播/复用候选，不能假定 cache 已免费完成。

下一项优先候选是**封闭 view-load → 定义时 snapshot 的连续块搬运**。它依据域、stride、bounds、dtype 和使用点，不识别 attention 名称；可适用于 GEMV、转置读取中的连续内层以及其他要物化 tile 的算子。没有合法连续片段则保留原逐元素实现，不新增用户 DSL primitive。随后再组合 QK 的有限输出寄存器分组；不能用 QK 局部收益代替完整 kernel 计时。

### 16.2 语义和准入必须一起保留

- `parallel` 已提供 program 间独立性契约，不再要求用户额外证明这一点。但跨 phase 移动读写仍须尊重同一 program 的 effect 顺序；此次候选在原 load 定义点完成，不把读取延迟到 MMA。
- destination 是该 SSA 值的新鲜私有 snapshot。输入 buffer 可以与其他用户参数别名；不由此给所有输入加 `noalias`。之后覆写用户 buffer 不得改变已捕获的值。
- 每个 active program 及每个有效元素只观察原来的读；inactive program 不访问用户地址。边界零填充、ragged row、非连续外层和小尾块必须保持，不能凭内部私有 padding 允许越界读取用户 buffer。
- 首版仅选择类型/表示一致的非 volatile 读取与可证明连续片段；量化转换、动态 gather、不匹配 layout 保留 fallback。复制不授予新的浮点重结合/FMA权限。
- 如用 typed XIR bulk-transfer 描述保存信息，必须是 compiler-owned 的可验证语义，保留 snapshot 定义点、完整 bounds 和类型；不能按函数名称匹配、任意开放 external call，或在后端临时增加未计入资源计划的数组。该描述尚未实现。

### 16.3 对 cost policy / solver 的直接要求

候选的成本要取决于实际访问实现，而不仅是逻辑元素数：

```text
transfer plan = (program grouping, element grouping, source/destination layout,
                 residual masks, vector width, full/tail handling)

cost = address/mask issue + scalar/vector load/store service
     + layout conversion + live-state/code-size effects
```

共享计划至少需区分 logical bytes、生成的标量/向量访存组、残留mask与循环次数；它们都不是已测的DRAM流量或物理寄存器数。将 transfer 归属到 producer 节点或转换边，二者不可重复计费。backend policy 提供目标能力和成本，solver 联合选择 load/compute/consumer 的实现；不要从一次 MHA 采样拟合一个全局系数。

验证次序是：先在封闭拷贝上检查有/无mask、尾块、inactive program、别名与定义时快照，再检查实际对象是否出现连续访问，最后冻结未采样的完整 attention 与非attention对照。保留现有快照容量、程序次序和MMA算法作为控制；若实际packet/clone/ABI随候选变化，报告必须显式披露。**当前只有定位与设计依据，没有新编译器加速、自动policy或新的MPS/Torch/BLAS胜利。**

## 17. 连续定义快照搬运候选

第16节的首个实现采用静态 FP32、完整 program、已有私有快照的窄范围方案。`native_copy_vector_width=0` 默认关闭；SIMD target info 独立接受2/4/8，其他后端默认拒绝非零请求。用户 DSL、TileIR 的 `VIEW_LOAD` 语义以及 snapshot allocation plan 均不变。load/reduction producer fusion 优先；它没有被重复物化来凑一个 copy 候选。

### 17.1 准入是布局等式，不是 attention 特判

设源 shape 为 `s`、tile shape 为 `t`，二者的 row-major strides 为 `S_i=∏_{j>i}s_j`、`T_i=∏_{j>i}t_j`。首版要求每个非单位 tile 轴满足 `S_i=T_i`，源/tile 容积有限且不会溢出。单位轴没有变化坐标，因而不要求它的 stride 相等。

于是对于所有 tile 内坐标 `e`：

```text
source_address(o + e) = source_address(o) + Σ e_i S_i
                      = source_address(o) + Σ e_i T_i
                      = source_address(o) + tile_flat(e)
```

这给出一个完整连续区间。原 load 定义处再检查 `0 ≤ o_i ≤ s_i−t_i`；使用减法上界避免 `o_i+t_i` 溢出。所有轴都满足时才使用 fast path，不能只检查 flatten 后的 buffer 容量。边界不满足时，仍对原 tile 做逐元素 bounds/fill，包括负 origin、跨行与 ragged 尾块。第一版不切分任意非连续外层，相关 view 保留原路径。

```text
VIEW_LOAD 定义点 ── 既有 allocation/fusion plan
                         │ snapshot 且连续
                  整个逻辑 view 有效？
                    /              \
             typed bulk copy    原逐元素 bounds/fill
                    \              /
                     同一个 snapshot
                            │
                 后续 effects / MMA / consumers
```

### 17.2 必需的中间层语义

XIR `ContiguousCopyMD` 附在 compiler-owned external declaration，保存 element count 与内部向量宽度；三个实参分别为 typed `buffer<float>` resource、`uint64` element offset、完整 root-local `array<float>` destination reference。名字仅用于调试。clone、text、bitcode、verifier 以及 Schedule 的拥有型副本都要保留此语义；普通 external calls 仍拒绝，错误类型/容量/目标存储或混合 semantic tags 均不可默默接受。

LLVM helper 在同一 module 中，以整数位模式连续搬运 FP32，向量 full chunks 之后是精确 scalar tail。每个 active program 独立调用，inactive program 不求值用户内存访问。它沿用 buffer read 的有效资源区间前提，不为非法宿主指针编造零值，也不声明所有用户资源 `noalias`。新鲜私有 destination 与资源不别名，但源 buffer 仍可与其他用户参数别名。整个 copy 在定义处完成，小 tile 从已完成的 snapshot 读取为 SSA，不能延迟到 consumer 或改变 carry ABI。

### 17.3 成本边界

`ExecutionNativeCopyWork` 向 backend policy 提供每完整 program 的候选调用数、full-path 向量访存组、scalar tail 元素与 fallback 元素。这两条路径互斥，不能把它们相加当作动态工作；逻辑 bytes 和 snapshot capacity 也不能重复收费。默认 prior 暂不提供 native copy 加速奖励，保留 `native_copy_cost=unmodeled`。

完整 attention 对照固定 native MMA width4、W8/block32/local1、MMA block/unroll 策略，只改变 copy width0/4。两臂要求输入、数学策略、snapshot capacity 和输出 bit pattern 相同；interleaving、native code、clone admission、workspace/entry ABI 仍须逐项记录。

### 17.4 实测：搬运映射有收益，但必须联合代码预算选择

[六尺寸连续复制实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-native-copy/notes.md)完成12次实际ORC capture、72次ABBA visits和504个样本。三组decode（包括KV=8193）的配对时间减少70.2%–78.8%，prefill-q4减少25.8%，batch GQA减少50.4%；prefill-q8增加2.6%。全部输出通过完整FP64、guard、输入不变与A/B逐bit检查。

这次六对的snapshot容量、allocation数、交错数组数、workspace、root order、task grain及入口ABI均不变。实际MHA/long对象确认K/V是连续NEON搬运且helper已内联；prefill-q8的原始entry却从3786增至4146条LLVM指令，跨过4096门槛，从一个full-packet clone变成没有clone，其他五组的clone存在性不变。因此完整实现虽只开关copy候选，最终代码特化仍会改变，不能把时间差都归因于单独一条向量load/store。

这给联合求解提出了实际要求：`copy choice × compute choice × physical layout × specialization budget`不能只靠各自独立的加速系数相加。原始逻辑工作相同不代表搬运服务成本相同；连续复制的guard/fallback也会增加特化前IR，必须使用实际生成代码预算验证候选组合。满包收益消失的具体耗时仍需独立控制实验，不能把本次相关性当作已分解的因果成本。

完整选定构建、11项CTest和25个C++ TU的syntax检查通过。该实现不识别算子名称，但性能泛化仍需更大prefill和非attention留出集；目前默认仍关闭、cost仍明确unmodeled。这一轮基线固定native MMA开启，不是前一轮MMA关闭的更快基线，也没有新的Torch/MPS/BLAS配对，整体性能目标尚未完成。

## 18. 特化后预算与联合候选

原有 full-packet 准入只查看尚未代入 `active_lane_count=W` 的原始 LLVM entry。这个计数是复制工作的上界，却不等于特化后需要保留的代码：全包能消掉某些入口 lane mask，不能消掉内部数据依赖、逻辑 view 边界或 causal mask。因此，把原始 entry 的 4096 条门槛直接当作最终候选成本，会在搬运实现改变时产生不必要的准入跳变。

新增独立 opt-in `LUISA_SIMD_ENABLE_SIMPLIFIED_FULL_PACKET_SPECIALIZATION`，仍须显式开启原有 full-packet 候选，原 disable 开关优先。默认行为不变。新路径只在原始 entry 不超过8192条时创建临时候选，固定入口有效lane数；最多四轮本地常量折叠、死指令和不可达块清理，然后检查候选是否仍不超过4096条。没有提高保留代码的预算，也没有加入 inlining、循环展开或新的浮点重结合权限。失败时删除临时函数；原始尾包路径和所有helper均不被修改。

```text
语义/执行几何准入
      ↓
原始代码预算 ── 超限 → 原路径
      ↓
临时特化 + 有限局部清理
      ↓
候选代码预算 ── 超限 → 删除候选，原路径
      ↓
候选可用 ≠ 候选盈利 → 后续目标成本评价/实测选择
```

`full_packet_specialization_decision` 区分未请求、显式禁用、结构不适用、原始预算超限、候选预算超限与两种实际选中方式；另提供原始/候选指令数和清理轮数。旧 `full_packet_cloned_instructions` 仍表示成功复制的原始指令数，新路径中可能大于4096。新计数仅描述特化前后的entry，不包含helper、目标O1/O2或机器指令，不能解释为cycle成本。

应求解的对象是联合 realization `r=(storage, transfer, compute, packet specialization)`。先满足 Tile 语义和目标能力，再满足资源与编译预算，最后最小化实际服务时间；编译预算不能替代服务时间目标。native MMA会改变操作数是否需要可索引snapshot，copy会改变代码和mask实现，特化又影响这些路径的执行开销，所以不能用三个独立固定折扣相乘进行选择。当前实现提供候选与准入诊断，尚未提供校准后的自动联合选择策略。

## 19. 搬运不能强迫消费者改变 private layout

六尺寸、五个联合候选的首轮捕获暴露了新的组合缺陷。MHA decode 的普通 MMA + native copy 候选达到预先设置的180秒编译超时；另26份捕获成功，余下三份未执行。整轮未通过完整捕获门禁，**没有进入 native replay，也没有新的性能结论**。超时包括汇编捕获与实际 ORC 两遍机器码编译；调用栈采样定位到第二遍的 `LiveVariables` 分析，不能据此声称生产路径的单遍 JIT 必定超过180秒。

静态原因链是：copy 的整数组 reference 原先使 destination 退出 packet-interleaved 准入；普通 MMA 的标量投影随之从相邻 packet vector load 退到 masked gather。此 MHA 基线已生成42529条原始 entry 指令、2050处连续 private load，而且未使用 rolled MMA。因此，这不是减少 copy 自身指令就能保证盈利的组合。采样证明机器码活跃变量分析开销异常，但没有取得失败臂的完整 post-opt IR，不能把预计的 gather 数称作已经测得的目标指令数。

修复将已验证的 typed copy destination 纳入封闭地址使用集合；未知调用、引用逃逸和 native MMA 仍不准入交错布局。设 packet lane 为 `p`、program 内元素为 `e`：

```text
外部 source[p][e]（e 连续）
           │  每个 active program：连续整数向量 load
           ├── native MMA consumer → destination[p][e]，连续 vector store
           └── packet consumer    → destination[e][p]，固定步长 scalar stores
                                             │
                                   后续相邻 packet vector load
```

交错目的地址为 `base + (e·W+p)·sizeof(float)`。root handle 已含 `p·sizeof(float)`，copy helper 只把元素步长设为 `W`；不是再加一次 lane 偏移。向量 chunk 只读取完整有效区间，余数逐元素处理；每次只写当前 active program 的字，不覆盖其他 lane。现有 snapshot 的容量、定义时机、外部 alias 契约与 carry ABI 均不变，也不需要中转数组。即便没有启用相邻 private 访问优化，GEP/gather 仍使用同一物理双射。

这仍是通用实现候选，不是已经完成的自动成本模型。后续 policy 需区分 `source vector load + destination vector store` 与 `source vector load + strided scalar stores`，并联合考虑消费者连续读/gather以及展开规模；不能继续给二者相同的向量访存折扣。编译预算也需估算 legalization 后的展开压力，而不只统计优化前 LLVM 指令。此次修复保留兼容布局，尚未实现上述成本校准或全局 solver。

修复后的完整选定构建、11项CTest（151.16秒）及六个相关C++ TU语法检查通过。copy测试包含54组独立编译配置和8组copy/native-MMA混合配置：packet W2/4/8/16、source R2/4/8、交错开/关、空/奇/偶/全runtime mask、所有active count、源保护页、精确FP32位模式及逐word workspace检查。混合测试使用独立有限数值oracle，不把MMA算术误当作NaN payload复制；不支持的R16仍显式拒绝。该回归证明这些边界测试通过，不替代完整attention的性能复测。

## 20. 联合实验：收益方向取决于相邻实现

[修复后的完整实验](../../scripts/benchmark/tile_torch/results/m1-max-20260913-attention-joint-repack/notes.md)完成30份实际ORC capture、30组独立配对、360 visits与2520 samples，离线审计通过全部240条编译/准备/计时命令及完整FP64/位模式/guard检查。失败旧轮单独保留，未把其中成功的26份对象混入新轮。此前MHA普通MMA+copy的180秒捕获超时，本次相同双编译流程的编译阶段为4588.59 ms；交错数组仍为4块、连续private读取仍为2050处。

下表各格是独立ABBA边的候选/基线耗时比，小于1为候选更快。M0/M4表示普通/native MMA，C0/C4表示原/native copy；不是从不同时段的表格时间相除。

| case | copy：M0下 C4/C0 | copy：M4下 C4/C0 | MMA：C0下 M4/M0 | MMA：C4下 M4/M0 | M4C4：简化特化/原特化 |
|---|---:|---:|---:|---:|---:|
| MHA decode | 0.614 | 0.285 | 1.535 | 0.718 | 1.000 |
| GQA D80 decode | 0.396 | 0.213 | 1.469 | 0.790 | 0.998 |
| KV8193 decode | 0.656 | 0.300 | 1.606 | 0.734 | 0.991 |
| prefill-q4 | 0.865 | 0.742 | 1.420 | 1.222 | 0.741 |
| prefill-q8 | 0.885 | 1.026 | 1.064 | 1.233 | 0.869 |
| batch GQA | 0.629 | 0.496 | 0.741 | 0.585 | 0.995 |

三个decode的native MMA在原copy下更慢，在native copy下更快，支持联合选择而非独立乘固定折扣。这里不能对不同边继续做比值之比并称为因果交互系数。prefill-q4/q8的简化特化分别减少25.9%/13.1%耗时：原始entry为6495/4146条，简化后的候选为3182/2425条，均在原4096条保留预算内。MHA虽然也得到clone，耗时基本不变；因此“能特化”仍不等于“值得特化”。这是有限五臂实验，未穷举所有组合，也未证明某臂全局最优。

运行收益并非没有代价：q4/q8的实际对象分别由121952/35976 B增至211752/63544 B；捕获中的编译阶段分别由3498.50/443.01 ms增至6489.06/1074.93 ms。它们仍是包含汇编导出的双编译流程，不能直接拟合生产单次JIT成本。未来若优化端到端生命周期，目标还需包含真实生产编译成本及预期调用次数，而不能只最小化单次kernel时间。4096只是局部候选的LLVM指令准入预算，不是最终机器码字节上限。

另一个预声明补充实验固定`M4C4-simplified`，对MHA、KV8193、prefill-q8分别比较手写online NEON与dense Accelerate BLAS，共72 visits、504 samples。22项协议测试、24项validation-only检查及补充离线审计均通过。**目标仍未达成**：固定Tile候选约为NEON耗时的2.0–2.3倍、BLAS的2.5–4.2倍。参照使用FP32但允许不同的求和顺序；BLAS还允许内部FMA和dense score物化，不能直接当作strict MMA的合法替换。它们使用相同原生计时器并确认BLAS同线程模式；不包含Runtime/Python/JIT/调用者分配，却包含完整入口、launch reset、block traversal和BLAS内部工作。没有新的Torch、MPS、Metal或多线程Runtime结果。

后续优化应先profile当前copy-on产物，重新确认剩余copy、private traffic、Q复用与exp/reduction占比；不能沿用copy-off时的热点百分比。再扩展通用phase候选：保留正确effect顺序的snapshot消除/复用、受寄存器预算约束的输出分组，以及跨相邻阶段的布局选择。大prefill和非attention留出集仍需补测。当前已实现候选与合法性/编译预算门禁，默认仍opt-in，自动成本校准与联合solver尚未完成。

## 21. Metal：问题是 contribution 的协作映射，不只是 intrinsic

[Metal attention 复测](../../scripts/benchmark/tile_torch/results/m1-max-20260914-metal-attention/notes.md)保留两条路径的不同结论。TIRx/Torch-MPS 小尺寸对照数值通过，但相邻 round 的 GPU control 时间比从 3.391 变成 0.205，拒绝性能验收。XIR→Metal4 的固定批次完整 attention 中，插桩 dispatch 达到约 1.9–2.0 ms，确认存在大量 kernel 内工作；两种计时口径及不同 cohort 不混为跨路径排名。本轮使用有指纹的 selected build，不包含受保护 TIRx WIP，也不声称代表完整当前 HEAD。

### 21.1 已确认的 reduction 缺口

这批 XIR attention 的实际计划都是 `local_lanes=1`。32/64/256 threads/group 描述独立 programs 的打包，不表示一个 program 的 reduction 已由这么多线程协作。

- [`packet_local_program`](bridge/xir/representation.h) 的准入范围是有限的一维 map/closed-reduction 程序族；它尚不接收 attention 的 pipeline、MMA 和跨 phase ownership。这个 guard 表达 emitter 的能力边界，不是在否认 `parallel` 的无依赖语义。直接删除 guard 不能生成缺失的数据重分布与 carry。
- [`reduction_emission_plan`](bridge/xir/representation.h) 只有实际分布到多 lane 时才进入跨 lane 分支。`unordered_reduction_partitions=4` 是线程内独立 partial chains 的选项，**不是四个 warp，也不保证每个 reduction 都启用它**。本次 BK16、展开阈值64、local1 下，max/sum 走普通线程内 carry loop；默认 unordered 许可允许这种求值顺序，但没有自动得到并行 tree。
- [`_partial_reduction`](bridge/xir/lower.cpp) 已有 `WARP_READ_LANE` butterfly 与 root broadcast，服务于准入的分布式 closed reduction。缺口不是整个 bridge 没写 shuffle，而是当前完整 attention 到不了这个实现。
- QK/PV 也由 [`_mma`](bridge/xir/lower.cpp) 的普通标量 MUL/ADD contraction 路径处理，而不是本轮已经接上 GPU 协作矩阵 atom。max/sum、dot/contraction 和跨 KV 的 online carry 不能混称成一个 reduction 热点。静态源码不能给出三者实际耗时占比。

TIRx 不能套用上述“没有 collective”的结论：第4节的历史生成 Metal 已有 `simd_sum`，同时仍有串行 PV、少量 active lanes 和 shared/barrier 往返。**使用 warp intrinsic 是实现条件之一，不是完整程序高性能的充分条件。**

### 21.2 粒度实验支持结构诊断，但不证明归因

两种形状分别做两组 ABBA，共16 visits；每 visit 固定8次吞吐 dispatch、3个样本，BK16、local1、QK/PV MMA、FP32与数学策略固定。BQ4→BQ1 改变完整 realization：

| shape `B,Hq,Hkv,Q,K,D,Dv` | dispatch threads：BQ4→BQ1 | threads/group：BQ4→BQ1 | 静态 snapshot B/worker：BQ4→BQ1 | 四个相邻配对的插桩吞吐时间比 BQ1/BQ4 |
|---|---:|---:|---:|---|
| 1,4,2,16,33,32,32 | 16→64 | 32→64 | 6688→4224 | 0.415、0.433、0.418、0.434 |
| 1,4,2,64,128,64,64 | 64→256 | 64→256 | 12832→8320 | 0.715、1.287、1.394、0.990 |

小尺寸有一致的 dispatch 改善；较大点的插桩时间方向不稳定，尽管其 feedback-only 吞吐控制全部改善。不能合并不同口径宣称稳定的纯 kernel 收益。BQ 改动同时影响程序数、group 宽度、SSA/数组表示和静态临时容量，不能把小点收益全部归因于 reduction、寄存器占用或某一个 layout。静态 snapshot 字节也不是实测 spill/寄存器数；本轮没有据此改变生产默认或拟合 cost 系数。

实际 timestamp 记录还揭示四种 realization **都只有一个 threadgroup**：`ceil_div(dispatch.x, block.x)=1`。BQ1 增加的是组内独立 programs，不是已增加跨 group 并行。当前 [`MetalTileCostPolicy`](../backends/metal4/tile/metal_tile.cpp) 使用总 packet arithmetic/memory work 加 block dispatch prior；固定相同 packet work 时，减少 block 数只会减少 dispatch 项，没有同时建模可并发 group、资源驻留与完成时间。这是可确认的目标函数缺项，不是测得的 GPU occupancy 百分比。后续必须同时测固定 BQ 下的 block32/64/128/256，先确认改变实际 group 数的效果；不能把较大 block 当作普遍更高的并行度。

### 21.3 下一步应修一般性映射能力

沿用第5节的分阶段计划，先为 row-wise reduction 保留两个独立域：输出域 `O` 与贡献域 `R`，由候选把它们映射到 programs/subgroups/lanes，而不是把全 kernel 的 `local_lanes` 当成唯一开关。
同时，group 划分的成本应近似资源受限完成时间，而不只是总工作量加正的 group 数惩罚；所需并发/驻留信息由 backend policy 提供。固定 BQ 的 group-size 对照和输出/贡献维的协作扩展是两项独立实验，不应合成一次无法归因的变更。

```text
QK output ownership → 按 row 的局部 partial + subgroup tree
                                       ↓ max/sum 的广播或重分布
                               PV output/contribution ownership
                                       ↓
                            同步更新 online (m, l, acc)
```

该图是待实现目标，不是已存在的 arbitrary-phase emitter。候选需要共同规划参与 lane、每 lane 元素、尾部有效性、结果 ownership 和相邻 phase 的转换；保留用户初值只合并一次、ordered fallback、snapshot/effect 时机等语义。沿 KV 的 online recurrence 也不能仅凭内部 max/sum 是 unordered 就任意重排；split-KV 需要完整状态的合法合并。

先以 softmax/RMSNorm/dot 的独立输出域、非32倍数贡献长度和 ordered 反例验证，再组合 attention 的 QK→softmax→PV。性能诊断应分离“输出分布不变，只换 reduction 实现”和“整程序联合重新映射”，分别记录实际 collective、layout transition、barrier 与容量。TIRx 的 initializer/PV tensorization 隔离实验仍是独立候选；它不替代 XIR 的上述结构修复，也不能混入一次无法归因的整体开关。
