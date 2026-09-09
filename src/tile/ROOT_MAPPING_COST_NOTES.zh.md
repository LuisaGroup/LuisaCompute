# Root 执行映射与缓存成本模型：设计笔记

## 结论与状态

本文讨论如何让 XIR bridge 根据通用的资源访问关系，自动选择 root
`parallel` 的执行顺序。目标不是为 GEMM 增加名字匹配或默认参数表，而是求解：

```text
独立 root programs + 资源访问关系
                 │
       合法、规范化的执行映射候选
                 │
     packet / task 内的访问与重复间隔
                 │
  target policy：issue、cache、TLB、调度成本
                 │
       有预算的候选搜索与验证
```

截至 2026 年 9 月 9 日，`root_axis_tiles` 的显式执行映射已实现，
但本文的资源访问摘要、缓存重用成本及自动分块搜索**尚未实现**。
当前 planner 仍明确记录 `root_temporal_cache_cost=unmodeled`；
显式约束不能写成自动调优成果。

固定源 kernel 的 4096³ FP32 GEMM 诊断显示：root `32×32` 映射有改善，
但没有解决性能问题。两次独立单样本记录的同步 host-wall throughput 分别为
13.111 s 和 8.753 s。它们不是 ABBA 实验，不是纯 native-entry 计时，
也不是原归档矩阵的完整计时协议，不能作为稳定加速比或已修复 timeout 的证据。
映射后的诊断进程本身仍耗时 70.38 s。

因此，容量 footprint 只能解释一个候选为什么值得测试，不能直接等同于
cache 命中、TLB 命中或最终时间。默认优化还必须保留 gather/native issue
成本，并补齐 profile、不同尺寸及无关算子的验证。

## 1. 范围：只改变独立 program 的枚举顺序

设 rank 为 r 的 root 域为 `P = [0,E0) × … × [0,E_(r-1))`，候选映射为双射
`σ: [0,|P|) → P`。初版候选来自静态整除 split、轴顺序和组合；
每个 program 内的 pipeline、serial、MMA、carry、load/store 次序保持不变。

这不是 memory layout 变换，不是 K-loop interchange，不是 register blocking，
也不自动生成 packed microkernel。普通 `parallel` 已承诺实例之间独立；
重排这些实例不需要重新证明用户是否允许并行。需要验证的是新映射本身的
双射、范围和溢出条件，以及具体 lowering 对该映射的支持。

实现入口以符号定位，避免随开发移动的行号失效：

- [`detail::root_mapping`、`RootMapping`](bridge/xir/root_mapping.h)：mixed-radix
  描述，去除 unit digits、合并相邻可合并 digits、识别恒等映射。
- [`Lowerer` 的 root parallel lowering](bridge/xir/lower.cpp)：
  root 坐标绑定；内层执行不应随 root 重排而改变。
- [`solve`、`SpatialAxis`、`measure`、`distribute_work`](bridge/xir/planner.cpp)：
  候选、packet 空间访问 prior 和 Runtime task 负载估计。
- [`ExecutionCostPolicy`](../../include/luisa/tile/bridge/xir/planner.h)：
  后端可替换成本，不能替换合法性判定。

### 1.1 等价枚举序列必须得到相同成本

令两个候选满足对所有合法 `j` 都有 `σ1(j) = σ2(j)`。
在相同 packet 宽度、task 划分和内部 lowering 下，它们必须得到相同访问摘要
和相同成本。不能因为一个 factorization 的名义 tile 更小，就赠送缓存收益。

候选应按规范化后的映射去重；footprint 分析只读取真正的 `σ`，
不以未经规范化的 `root_axis_tiles` 字段作为 locality 特征。
全 1、全 extent、单轴或 unit-axis 分解等恒等情形必须保留原代码路径。

## 2. IR 已有足够的普通 View 信息，但没有完整缓存模型

普通 XIR View 访问可从现有 IR 提取：

- `VIEW_LOAD/VIEW_STORE` 的 operand 0 是资源，随后是 origin；
  `domain()` 存在时给出 subtile 形状，否则是 scalar 访问。
- View 的 `Type::index_space()` 给出完整逻辑尺寸，`scalar_type_size()`
  给出存储字节数；这与 accumulator 的计算精度分开。
- origin 是可检查的 Scalar SSA，不是不可见的 C++ callback。
- region 域、block arguments 和词法顺序给出 root、pipeline、serial、
  reduce 及 tile element 的迭代结构。
- `bounds_mode()` 和 masked scalar load 的 predicate/fallback 显式存在。

相关定义见 [`Type`、`Operation`](../../include/luisa/tile/ir.h)、
[`IRBuilder::create_tile_load/create_tile_store/create_view_load`](ir.cpp)、
[`_capture_view_access`、`_view_element`](bridge/xir/lower.cpp)。
当前 XIR 只接受直接 buffer argument，并在 `_view_element` 中按 row-major
规则展开地址。因此第一版可以复用这份确切规则，而不推测任意 Tensor layout。

注意：`Operation::memory_layout()` 是 `MEMORY_ALLOC` 的 allocation-local
映射，不是普通 View 的物理 stride 元数据。当前 XIR 拒绝手工 Memory；
不能把普通 View 的 cache 分析建立在一个不存在的 `memory_layout` 上。
将来支持更多 View layout 时，应组合真正的物理 IndexMap，无法识别则回退。

### 2.1 最小访问摘要

对资源 `r`、root 坐标 `p`、内层执行坐标 `k`、tile 坐标 `t`，先支持：

```text
byte_offset_r(p,k,t) = c_r + A_r·p + B_r·k + D_r·t
```

摘要保留资源 identity、元素字节数、整数系数、各坐标域、访问词法位置、
read/write effect、bounds/predicate 状态，以及是否完整表示了访问集合。
这里的系数是 checked-integer 结果，不是浮点导数。

最小 recognizer 支持常量、region index、加减、常数乘和已验证不丢信息的 cast。
溢出、变量乘法、未知 mask、数据相关 gather/reindex、不可识别的循环 carry
和副作用都应产生 `unknown`，不能把未知轴错误地当作系数 0。
可逐步扩展到明确受控的 floor-div/mod 分段关系。

现有 `slope()` 仅用 double 估计一个轴上的变化率，适合空间访问 prior，
不应直接充当范围、集合相等、共享性或 alias 证明。
未证明的边界 clipping 不允许增加复用信用；初版可只分析已证明 in-bounds
的访问／窗口，把尾部保留为保守路径。

## 3. 共享只读性可以局部推导，但不是全参数 noalias

`Usage::READ` 只说明通过某个 IR 资源身份观察到读取，不说明另一个参数
不能指向相同存储。缓存分析不能由不同参数名推导不相交。

不过，普通 `parallel` 的契约能支持下面这个更小的推论。

### 3.1 共享读取 footprint 不可被 root 实例写入

设 `R(p)`、`W(p)` 分别为一个 root 实例实际读、写的**字节集合**。
普通独立 parallel 的合法执行满足不同实例间没有 ordinary read/write 冲突：

```text
p ≠ q  ⇒  W(p) ∩ (R(q) ∪ W(q)) = ∅
```

若某个字节 `b` 被至少两个不同实例 `p`、`q` 读取，则没有任何 root 实例
`z` 能合法写入 `b`：`z` 不可能同时等于 `p` 和 `q`，因此至少会与其中一个
不同实例的读取冲突。这只推出**实际共享读取的 footprint**不可被该 root 写入，
不推出整个 buffer、所有参数或后续 scope 全局 immutable/noalias。

一个可实现的充分条件是：

1. root 是这里定义的 ordinary independent parallel；不是允许冲突的 atomic
   或其他特殊 collective 语义。
2. 至少一个 root 轴具有两个实例；该访问的地址、是否执行、predicate、
   内层迭代域和访问域均对这个轴不变。
3. 访问确实发生且字节集合可确定；不能只看到地址系数 0，却忽略依赖该轴的 mask。
4. 分析范围是同一个 root 执行；没有未建模的外部并发修改或未知 effect。

固定 GEMM 的 A 对 gn 不变、B 对 gm 不变，因此可得到这种共享只读性。
这个条件同样适用于其他广播权重或只读表，不需要检查 `MMA` 或 kernel 名称。
A/B 之间仍可只读 alias；按不同资源分别计算 footprint 时，可保守地高估容量，
但不能凭空扣除重叠。

无法建立上述条件、也无法局部证明读写不相交时，相关复用信用回退为 0。
未来可参考 [`PointwiseRegion::alias_pairs`](bridge/xir/pointwise.h) 的运行时
版本化方式，但不能偷偷给已有 kernel 增加全参数 noalias 前提。

字节无冲突不等于 cache line 无干扰：邻接字节仍可能 false-share 一条 line，
未知 base alignment 也影响 line/page footprint。共享只读推论不是命中率保证，
写入污染和 coherence 风险仍需成本模型或诊断处理。

## 4. 固定 task 观察窗口，才能公平比较资源 footprint

设一个 Runtime task 对应连续 physical dispatch 区间 `I`。
使用候选真正的 packet/program 映射，将其切成最多 `H` 个 logical programs
的观察窗口；`H` 是统一的分析配置，不随候选 tile 面积变化。

```text
同一个 task 区间
  ├─ 固定 H 的窗口 0：按 σ 访问实际 root 坐标
  ├─ 固定 H 的窗口 1：按 σ 访问实际 root 坐标
  └─ 同样处理剩余窗口与尾部
```

窗口起点从 task 起点导出，不从候选自己的 tile 边界重新对齐。
比较不同 task grain 时，要分别覆盖各自真实 task 区间并按实际工作量计费；
不能只挑一个最有利的完整 tile。跨 task 不默认延续 cache 状态。
单 worker 时，Runtime 会执行整个范围，而不是请求的多个独立 chunks，
分析必须沿用这一例外。

Runtime 的稳定 home chunks 和 work stealing 见
[`SIMDThreadPool::_parallel_for/_worker_loop`](../backends/simd/runtime/simd_thread_pool.cpp)。
task 内执行顺序可建模；跨 task 同核、跨 dispatch 热缓存、共享 L2 的竞争
都不是这个调度器给出的保证。

为避免分析整个大矩阵，第一版可解析相同窗口的周期类及边界类，按出现次数加权。
解析或枚举超过预算时保守回退；若以后采用采样窗口，必须在 metadata 标明采样
覆盖范围，不能把采样 prior 表述成全域精确成本。

### 4.1 一个可核算的 GEMM 窗口

取 FP32 `M=N=K=4096`、源微块 `2×2×4`、W8、whole-program lanes，
且 packet 沿 N 方向排列。root 域是 `2048×2048`。
选择 task 内对齐的 `H=1024` 个 programs，比较相同 128 packets：

```text
identity：输出区域 M2 × N2048
  A payload footprint = 2 × 4096 × 4 B       = 32 KiB
  B payload footprint = 4096 × 2048 × 4 B    = 32 MiB
  C 写 payload        = 2 × 2048 × 4 B       = 16 KiB

root tiles (32,32)：输出区域 M64 × N64
  A payload footprint = 64 × 4096 × 4 B      = 1 MiB
  B payload footprint = 4096 × 64 × 4 B      = 1 MiB
  C 写 payload        = 64 × 64 × 4 B        = 16 KiB
```

这些是地址集合的 payload 字节数，不是实际 cache line 数、DRAM 流量或计时。
在当前已实现 A broadcast 的表示下，每个 packet 读取约 32 KiB A 和 256 KiB B，
因此两种顺序每窗的请求 payload 都约 36 MiB；差异是不同请求之间的重用。

identity 中某个 B 面板再次出现，通常要等完整 N 扫描，涉及约 64 MiB 的 B 流；
blocked 顺序只需经过 N64 的 B 面板，约 1 MiB，再加相邻 A 与写入流。
这是“值得测试”的结构性证据，不是“B 一定驻留 cache”的结论。

## 5. 重复面板和重用距离，比整个窗口是否装得下更可靠

只判断 `window_footprint < cache_capacity` 会漏掉大窗口中 A 的短距离重复，
也容易过度奖励另一个候选。最小改进是保留同一窗口内的 panel 顺序。

对每个 packet 的 load，把完整内层静态访问表示为一个面板。面板的 signature
由访问模板和代入根坐标后的 byte offset 决定。相同 signature 必须代表同一个
实际字节集合；相同 bounding box 不足以证明这一点。相同模板的精确平移关系
可先处理；不同 load 之间的部分重叠可以暂时不计信用。

若面板在窗口中再次出现，令 `D` 为前后两次出现之间所有读、写及 RFO 访问的
cache-line footprint union 上界。包含两端的完整面板是一个可用的保守上界，
不必展开到每个 scalar 访问；规则矩形／strided-box 摘要可覆盖内层长循环。

容量 prior 可以写为：

```text
reuse_admitted = shared_readonly_known
                && exact_previous_panel_in_same_window
                && footprint_bound_known
                && D <= η * effective_cache_capacity

score = arithmetic + address/issue + root_decode + dispatch/imbalance
      + hit_cost * requested_cache_lines
      + miss_extra_cost * predicted_missed_cache_lines
```

首次访问、未确认的重复、跨窗口／task 复用和未知访问保守按 miss 估计。
这里的 `D` 是容量筛选输入，`reuse_admitted` 只是 **cost prior**；真实 cache
不是理想全相联 LRU，其他核、其他线程和系统负载也会干扰，不能把它用作程序
正确性、load 消除或同步消除的依据。

`η`、effective capacity、hit/miss 代价由 backend policy 提供并校准。
目前的 memory 系数不是纯 DRAM 成本，应拆开 issue 和数据移动成分，
避免一边按原 gathered-lane prior 完整计费，一边再重复计同一份访存成本。
反过来，也不能因为预测命中就删除 gather、地址计算、mask、寄存器溢出等成本。

当前接口缺少 per-resource 摘要、cache/TLB 参数、line/page 对齐信息和重复距离。
这些应作为 bridge 提取的候选特征传入 `ExecutionCostPolicy`，让后端重写硬件
信息和代价，不让求解器内嵌特定 CPU 的容量常数或算子名称。

## 6. 已有诊断只说明部分改善，尚未解释剩余瓶颈

本段数据来自同一台 M1 Max 上两个分开的诊断运行，均为 FP32、严格数学
`fast_math=false`、源 `2×2×4`、W8、8 CPU workers、block 1024，
map fusion 和 full-packet specialization 均关闭。

- identity：`throughput_us=[13111367.5]`，`latency_us=[12815684.875]`。
- 固定 root `(32,32)`：`throughput_us=[8753350.834]`，
  `latency_us=[9257475.583]`；metadata 明确为 fixed constraint，时间缓存成本未建模。
- 两者均完成两次全输出检查，每次 16,777,216 个元素和 34 个 guard 元素，
  最大绝对误差 0。

计时是 `synchronized_host_wall`，每个记录只有一个 throughput 和一个 latency
样本，且 `repetitions=1`。没有 ABBA、置信区间或跨时段稳定性证据。
这些数据不是当前旧／新 tile 正式矩阵的替换项，也不能和 map-fusion 的
single-worker pure-entry 数据混算。

两种 realization 都报告 `contiguous reads=0; broadcasts=8`。
root 重排没有改变 B 的 lane stride=2 gather；单个 packet 仍完整扫描 K。
这正说明仅优化 root 次序不会同时解决下面所有问题：

- **Gather/native issue。** 相同微块下的 gather、mask、整数地址运算和
  FP32 mul/add 成本仍存在；cache hit 不会把它们自动变成更好的微内核。
- **TLB。** B 的行字节步长为 `4096×4 = 16 KiB`；跨行访问会涉及多少 page、
  TLB 级别和 page walk，取决于实际 page size、base alignment 和硬件。
  未测量前不能指定一个看似精确的 TLB 容量或命中率。
- **Cache set 冲突。** 小于总容量的 footprint 仍可能集中到少数 sets。
  地址低位、associativity、replacement、物理地址/hash 均未建模。
- **资源竞争。** 多核共享 cache、带宽、coherence/false sharing、调度迁移、
  频率和热状态都会影响 wall time；理想单核容量模型不能直接外推多线程。
- **遍历与计算组织的区别。** `root_axis_tiles` 不做 K/cache blocking、packing
  或 register blocking；手工 `m8n1k4` 改变了计算和加载结构，不能把它的全部
  收益归因于 root mapping。

下一步 profile 应区分地址／gather issue、cache miss、TLB/page walk、spill
以及 worker 利用率。已有 stack sample 只能证明工作线程在执行 JIT 代码，
不能单靠这一点宣称是 DRAM、TLB 或某个 cache level 的瓶颈。
如果当前工具拿不到某些 counters，就明确保留未知，并用受控 stride、尺寸、
packet 宽度和微块实验缩小原因，而不填造计数器结论。

### 6.1 诊断证据定位

本地诊断根目录为 `/tmp/luisa-simd-gemm.dx2oxi/`：

- `baseline.log`、`baseline.source.txt`、`baseline.sample.txt`。
- `root32-4096.log`、`root32-4096.source.txt` 及同名前缀的 inputs/oracle/output。
- `root-v1/source/` 保存显式 root mapping 诊断使用的相关实现。

长期证据保存于
[`m1-max-20260909-simd-root-traversal`](../../scripts/benchmark/tile_torch/results/m1-max-20260909-simd-root-traversal/README.md)，
包含完整输入、oracle、输出、生成源码、实现快照和验证日志。
这些诊断没有事前完整二进制冻结或 ORC 纯入口重放，不能追补成受控实验；
原 60 s 超时记录仍应保留为 Error，不由成功的长时诊断覆盖。

## 7. 最小实现顺序与默认启用门槛

1. 增加独立的静态访问摘要分析，先覆盖普通 row-major View，明确 unknown
   和分析预算；与现有单轴 slope 分开。
2. 根据每资源对 root 轴的依赖／不依赖关系，生成少量合法整除分块候选。
   使用规范化映射去重，保留 identity；不检查 kernel 名、维度名或 MMA 名。
3. 按统一 H、真实 packet/task 边界提取面板及重复距离，写入可审查 metadata；
   对未知区域不给信用，不能为了得出 winner 而忽略分析失败。
4. 增加 backend cache policy，分别计 issue、数据移动和索引开销。
   TLB/set 冲突尚未建模时明确标记，不包装成完整硬件时间预测器。
5. 对有可信收益、超过新增索引开销与不确定性余量的候选进行测量验证；
   其他情况保留 identity。候选数、观察窗口和模型校准均须有预算。
6. 只有多尺寸、多算子、受控协议证明默认策略不造成显著回退后才默认启用。

必要验证包括：同序列不同 factorization 成本一致；非零 task 起点和尾部；
rank 1/2/3 与 unit axes；共享只读推论的成立及 mask 反例；RW/alias/dynamic
回退；严格 serial/fold 及 pipeline 数值次序不变；copy、transpose、广播
elementwise、reduction、scan 与不同 GEMM 尺寸。

性能实验应固定源 kernel、worker 数、packet 宽度和无关开关，交错执行
baseline/candidate，并分别保留纯 native-entry 与 Runtime 多线程 dispatch
结果。模型训练／校准样本与 held-out 验证样本分开；不能用同一组拟合结果
自证泛化，也不能把显式 probe 的获胜直接写成 solver 已找到默认最优值。

仍需回答的问题是：可用 profile 能否证明容量之外的主瓶颈；TLB/set 风险需要
怎样的最低特征；以及 root traversal、lane 分布、register/K blocking 是否应
作为同一个有预算候选空间联合求解。本文给出了可实施的第一层，不声称已经
得到完备或校准完成的硬件性能模型。

## 8. 今晚冻结与 CUDA 交接边界

2026 年 9 月 9 日收尾时，`next` 的 `b6b10e603` 是 map-fusion checkpoint，
不是本文自动 cache planner 的完成标记。显式 root `32×32` 的诊断改善仅为
同源 kernel 的 13.11 s → 8.75 s 单样本观察；**默认 4096³ SIMD 矩阵的
Error 尚未修复**，性能目标不得标记完成。

今晚交接 Maxwell 继续 CUDA 工作；本文及 SIMD 诊断先冻结，明天再继续：

- 不把 CPU root factors、cache 容量 prior 或 W8 lane 假设直接当作 CUDA 默认值。
- 共享的是独立 parallel 契约、资源访问摘要和 policy/solver 分离；CUDA 的
  执行约束、资源容量、通信与成本需由对应 backend 实现和验证。
- CUDA 新结果单独记录 target、lowering、精度、尺寸和计时口径，不和本文
  同步 host-wall 诊断或已有 pure-entry 结果混算。
- 保留未解决 Error 和原始证据；恢复 SIMD 工作时从明确的源码/二进制
  checkpoint 重建测试，不把临时 probe 的成功外推成默认路径已修复。
