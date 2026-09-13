# Attention execution mapping：现状、缺口与有界实验

记录日期：2026-09-13。范围：当前源码静态审查与已归档实验；**本记录没有新性能测量，也不宣称已完成生产优化**。
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
当前 `--attention-pv reduce` 刚加入，**没有新性能结果**；默认仍走 MMA，metadata 必须确认开关确实生效，不能只信命令行。
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
