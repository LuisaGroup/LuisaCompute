# 私有访问成本特征：最小实现备忘

状态：**提案，尚未实现、尚未校准**。日期：2026-09-14。
本文记录现有代码的计价缺口及下一步实现范围，不报告新性能结果，不把相对工作量视为周期。
不新增 DSL 实体，不改变 `parallel` 的无冲突源程序约定；访问合法性仍由 compiler/backend 保证。

## 1. 当前代码与计价缺口

| 代码锚点（行号为审查时位置，函数名为稳定定位） | 当前行为 / 缺口 |
| --- | --- |
| [planner.cpp:393](bridge/xir/planner.cpp#L393), `ProgramTeamWork::_private_access/_read/_snapshot` | 所有私有 load/store 统一收取 `gathered_lane * packet_width`；跨 owner 读取只额外收取粗略的四个算术工作单位。 |
| [planner.cpp:137](bridge/xir/planner.cpp#L137), `read_work/measure` | 旧 local1/common-axis 的 materialized read、snapshot、carry 也使用同一 gather prior，不能只修 program-team 一侧。 |
| [lower.cpp:180](bridge/xir/lower.cpp#L180), `_storage_index` | 已保留相同 traversal 的 slot、cyclic uniform quotient；这些真实 realization 事实尚未进入成本特征。 |
| [lower.cpp:612](bridge/xir/lower.cpp#L612), `_read` | owner-local 读与 guarded source-owner load → reconverge → `WARP_READ_LANE` 是不同的路径。 |
| [emitter_memory.cpp:146](../backends/simd/llvm/llvm_schedule_emitter_memory.cpp#L146), `_find_interleaved_private_arrays` | 闭合 typed 地址树才可 interleave；uniform slot 才能进一步生成连续向量访问。 |
| [emitter_memory.cpp:251](../backends/simd/llvm/llvm_schedule_emitter_memory.cpp#L251), `_local_load/_local_store` | 连续读为 vector load + mask；连续写为 preserve-load + select + vector store，不是单次 store。否则仍为 masked gather/scatter。 |
| [emitter_collectives.cpp:175](../backends/simd/llvm/llvm_schedule_emitter_collectives.cpp#L175), `WARP_READ_LANE` | 同 cohort 的 source index 可以取标量再 splat；不能跨 collective epoch 复用该 uniform 事实。 |

CPU 的私有 interleaved 存储是 `(slot * W + lane) * sizeof(T)`。
即使只有一个 owner active，当前连续读仍读取 W 个存储元素；不能按一个标量字节数计价。
bool 计算为 i1，存储为 i8；新 bool/int8/uint8 准入不意味着 FP16、FP8 或其他量化类型自动获得该实现。
Metal 的 per-thread private storage 也不能直接套用 CPU 的连续向量数组成本。

## 2. lowering 前能知道什么

现有 [representation.h](bridge/xir/representation.h) 的 allocation/fusion 计划能说明哪些 SSA 定义需要 snapshot、哪些读可消除、native MMA reference snapshot 和 carry current/next 分配。
[ProgramTeamPlan](bridge/xir/program_plan.h) 提供 producer 的 replicated/cyclic 布局、local extents、consumer phase 和投影；[ValueLayout](bridge/xir/program_team.h) 提供精确 owner/slot 几何。
静态循环域、收缩域、local traversal 次数及 padded slots 也可计算。

但是，**相同 owner 不等于相同 slot，replicated 不等于 index uniform**。
还缺少共享的 slot 表达式分类、生成的 GEP 与访问的 cohort/epoch 关系，以及该 allocation 是否必须保留 native reference ABI。
Schedule 当前只把 warp-uniform index，或 GEP 与 load/store 同 block 的 cohort-uniform index，准入连续私有路径。
未来 Tile 成本分析不能用布局维度名称替代这项检查。

## 3. 最小 patch 范围

建议新增内部 `SnapshotAccessPlan`（拟议名称），由资源分析、成本提取与 lowering 共用，不把 TileIR 指针塞进公开 `ExecutionPlan`：

- 存储：elided / closed scalar array / reference-ABI array；synthetic carry current/next 也有记录。
- slot：same traversal slot / cohort-uniform projected slot / varying-or-unknown slot。
- 通信：owner-local / source-owner load-and-broadcast。
- mask：可证明 full 或 guarded；有效元素与 padded lane-slots 分开。
- 标量存储类型、字节宽度和生成访问是否保留 same-block address snapshot。

它描述选定 realization 的事件，不替代后端地址逃逸检查。无法证明的地址保守归类，不能靠便宜的成本反向取得合法性。
`lower.cpp::_storage_index/_read/_store_local/_copy` 消费同一计划，避免再复制一套只在 planner 中存在的地址分析。
最终 Schedule 分类仍是后端事实；若预测与实际不一致，应保留诊断和回归反例，而不是放宽 codegen 检查。

在 [ExecutionWork](../../include/luisa/tile/bridge/xir/planner.h#L168) 加一个未加权的 per-packet 子结构，最少保留：

| 特征 | 计数单位 |
| --- | --- |
| 连续 private load、RMW store、gather、scatter | 按存储字节宽度分组的逻辑动态访问次数；另存有效 lane-elements，不能拿它替代物理向量流量。 |
| source-owner broadcast | 每个消费位置的通信次数；内存读与通信各计一次。 |
| loop invocation / iteration | 实际 rolled traversal 的入口与迭代次数，不用 kernel 名称估算。 |
| 地址计算、guard/mask | 非重叠的工作分类；bool 存储转换也需区分。 |
| 有效元素 / padded lane-slots | tail 的数学工作与实际执行范围分开，不假定 masked lane 没有指令开销。 |

静态事件数应与动态 multiplicity 分开保存：前者可与生成 IR 的结构核对，后者乘静态循环次数用于评分。
LLVM 后续 CSE、scalarization、寄存器分配仍可改变最终机器指令，不宣称这些事件就是机器指令数。

后端接口只需新增保守默认的 private-realization capability 描述（可 interleave 的标量类型、cohort-private 与 uniform-read-lane 能力、masked-store 实现），与本次 compile flags 一致。
SIMD 在 [SIMDTileTargetInfo](../backends/simd/runtime/simd_tile.cpp#L26) 提供它；泛型/Metal 默认不采用 CPU 假设。
现有 `ExecutionCostPolicy::evaluate()` 接收新特征后即可覆盖权重，不需要新 solver 或 kernel-name 分支。

实施文件：公开 `planner.h`、内部 access-plan header、`planner.cpp` 两条测量路径、`lower.cpp`，必要时 `resources.h` 接入同一分配描述，以及 SIMD target info；测试落在已有 target-info / SIMD codegen suites。
不添加兼容 shim。第一阶段先提取并核对特征，再启用显式的校准 policy，不能把尚未校准的权重宣传成自动最优映射。

## 4. 两组独立 oracle 测试

### A. 同族短行 / 长行 reduction 的选择

固定 root programs、block、worker，分别用 width7 与1024，并枚举 local1/W。
host oracle 根据域、snapshot 计划和实际循环结构独立计算 load/store、RMW、loop 与 padding；cyclic local volume 按各维 local extents 的乘积计算，不能一概用 `ceil(total_elements/W)`。
用公开、明确的合成系数分别算两个候选分数，检查 planner 选择最小值；至少构造两组确实产生不同赢家的系数。
policy 不允许读取 kernel 名称或直接对 `candidate.local_lanes` 给偏好分数。
另外检查相同访问计划生成的 XIR/Schedule 分类，避免特征提取与 emitter 两边共享同一个错误 oracle。

### B. 跨 phase 的 reduction → broadcast → MMA

复用 ragged 输出小于 packet、输入 source owner 在该输出 lane 无有效元素的 fixture。
独立检查 W 候选的 owner broadcast 与 guard 计数，且广播必须保留输入 owner，不得使用输出 valid mask 截断通信。
local1 若启用 native MMA，其 reference arrays 不得误分类为 closed interleaved arrays；切换 native MMA 后应重新计算存储 ABI。
固定其他系数，提高通信价格应按上述计数影响排序；改 kernel 名字不得改变结果。
数值 oracle 仍覆盖完整输出、输入和输出 guard、snapshot/carry；结构 oracle 对照实际 load/store/collective，另检查跨 epoch index 不被当成全局 uniform。

这两组测试证明“评分输入与求解选择一致”，不证明合成系数对应真实硬件最佳性能。

## 5. 集成风险与不可重复收费

1. 新 private / loop / mask 项要替换已有相应加权项，不能在旧 `memory_per_packet/arithmetic_per_packet` 上直接再加一次。可保留非重叠的 legacy remainder，明示覆盖范围。
2. native-copy fastpath/fallback 是运行时二选一；不能相加。native MMA 自身 reference 读写与 snapshot 生产/消费要分清边界，不能重复记。
3. carry 初始化、next staging、next→current 提交是不同复制；必须保留 simultaneous snapshot 语义，不因成本高就删掉必要副本。
4. [planner.cpp:758](bridge/xir/planner.cpp#L758) 的旧 `work.memory *= 2` 不能作用到新物理字节/guard 统计；root tail 应单列，而非全部私有流量翻倍。
5. 新字段必须完整通过 `distribute_thread_pool_work()` 的聚合重建以及 `info.schedule()`。local1/W 的 packet 数与 per-program/per-packet 单位不能混用。
6. 配置变化必须重算 realization 特征：native MMA/copy、fusion、byte interleave、cohort-private、uniform-read-lane 都可能改变实现。不能沿用另一编译配置的缓存评分。
7. 不把当前访问 ABI / emitter 准入能力上升成 Tile 语言限制，也不要求用户提交无冲突证明。后续能力扩展仍须由共享计划和 backend validation 闭合。
