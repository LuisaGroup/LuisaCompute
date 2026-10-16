# XIR/SIMD 有界 Tile 表示：从 Torch 机器码检查落实到实现

2026-09-08，Apple M1 Max / macOS 26.6.2。结论：**消除了整行静态展开的主要
编译障碍，归约有实测改善，大尺寸现已能完成；尚未达到 Torch 性能目标。**
这里的 SIMD 是 CPU ARM64/NEON，不是 Metal SIMD-group。本轮没有修改 Metal/TIRx
优化，也没有新增 MPS、BLAS 或 GPU 纯 kernel 达标结论。

## 1. 借鉴了什么，哪些已经实现

[Torch 检查](../m1-max-20260908-torch-simd-inspection/notes.md)的核心启发是有界循环、
行内连续 SIMD、有限的归约状态和分阶段物化。当前实现落实了其中的基础部分：

| 部分 | 本轮实现 | 尚未实现 |
|---|---|---|
| Tile 表示 | 大 Tile 用运行时循环；大常量用 splat；单消费者纯 elementwise 合入消费者 | 连续 feature 向量 atom、多种物化方案的寻优 |
| 快照与状态 | 外部 `.load()` 仍在定义处读取；大循环携带值使用同时更新的 current/next 副本 | 跨效果重读输入、基于峰值存活区间的复用 |
| 归约 | 闭合的 unordered ADD/MUL/MIN/MAX 可用 1–16 个部分累加器，默认 4 | contribution → SIMD lane 分布、水平向量归约 |
| CPU 资源 | 按 W 倍物理容量检查；大私有数组放线程独占工作区并复用 | 完整机器栈/寄存器压力成本与资源生命周期求解 |
| planner | 同步表示分类和相对工作量计数；保留 root order/block 的精确有限枚举 | 表示阈值、累加器数和 CPU 任务粒度的联合寻优；校准成本 |

没有按 RMSNorm/Softmax 名字选择实现，没有加入用户 Memory 注解或新的 DSL primitive。
因此是通用的表示与资源策略，但**不是已经泛化验证的最优调度器**。
默认 `max_unrolled_tile_elements=64` 仍是固定代码大小策略，小尺寸回退说明它不是最优阈值。

```text
Tile SSA / execution regions
          │
          ├─ 小值 ───────────────> 标量 SSA
          ├─ 大 load / shared op ─> 有界循环 + snapshot
          └─ 单消费者纯表达式 ──> 消费者的 scalar recipe
                                      │
                      root programs → SIMD packets
                                      │
                    Σ aligned(W × local array bytes)
                         /                      \
                    ≤ 64 KiB                  > 64 KiB
                    stack                 每个 CPU 线程的 workspace
```

工作区上限为 16 MiB/执行线程，按需增长并在 packet 返回后复用，64-byte 对齐；
它不能用于需要跨 packet 保留状态的 cooperative/嵌套 handler 路径。独立 bridge 的
逻辑 worker 默认预算仍为 256 KiB；Runtime 根据 `16 MiB / W` 传入资源容量，codegen
再次检查对齐后的实际总量。此预算不包含任意 LLVM 寄存器 spill，不能称为完整栈证明。

### 数值与效果边界

初次 4096 项测试失败的值为 0.56826895，与逐项 FP32 参考完全相同；FP64 为
0.5683016598632094。是长串行累加误差，不是悄悄放宽容差的理由。
现在仅在默认 `unordered_tree` 已许可的闭合纯归约上重组部分累加器；各部分从真实
贡献开始，source initial 只合入一次，不额外插入零/一。fold L/R、额外 carry 依赖、
效果或复杂 region 保留顺序。固定分组数不等于允许宣称浮点加法严格满足结合律。

前一个工作区版本在 64×16384 RMSNorm、64×4096 masked softmax 的 worker 内
SIGSEGV；17×16384 masked softmax 被旧逻辑容量拒绝。Softmax 4096 的生成 LLVM
仅五个私有数组就占 688128 字节/packet。原始失败、生成 IR 和
[LLDB 调用栈](validation/crash-lldb-2.log)保留在 [pre-workspace](pre-workspace/results.json)，
没有把这 6 个失败 visit 替换成成功。资源修复后的独立完整 cohort 全部通过。

## 2. 真正生成的原生入口：隔离 Runtime/Python

[原生入口结果](native/results.json)比较的是同一 64×256 RMSNorm、相同 FP32 输入，
**单 CPU 线程**下的 C++ native-entry host-wall 时间。Luisa 直接重放实际 ORC object
链接出的入口，Torch 直接重放 `torch.compile` 生成的一线程 C++ 入口；不是另写一个
“类似 Torch”的手工 kernel。库加载、编译、Python、Runtime、线程池和分配均在计时外。
Luisa 的可变 launch record 重置和两者的调用/循环开销仍在计时内，不是硬件周期计数。

| 原生实现 | 六轮 p50 的中位数（µs） |
|---|---:|
| 旧 XIR：已有索引快照，但整行静态展开 | 98.570 |
| 本轮有界 XIR | 28.836 |
| TorchInductor 一线程生成入口 | 8.995 |

新 XIR 比旧实现约快 3.42×，但耗时仍约为 Inductor 的 **3.21×**。
三种实现的六种排列全部执行，每 visit 100 ms warmup、7 个约 ≥30 ms 的样本；
18 个 visit 全输出对照 FP64，并检查 output/partial 的 68 个 guard。
Luisa 禁用 fast math；Inductor 使用该版本默认编译数值策略，完整结果均满足
`atol=rtol=5e-5`，不声称两者逐位等价。Torch 为 2.14.0，源码版本及库哈希在 JSON 中。

[replay_native.py](replay_native.py)检查实际导出符号、参数顺序、无未解析 ORC 符号、
无 OpenMP team 的 Torch 源码后才调用。该诊断保持单线程，**不得把这张表同下文请求
8 线程的 E2E 表混算**。不同链接地址/代码放置是独立 native replay 的限制。

实际 [ORC object](code/kernel.o.gz) 的 `__text` 从上一快照版本的 262772 缩至
11564 字节；本次函数序言的栈为 `160 + 4×4096 + 512 = 17056` 字节，之前为
37376 字节。这里仍有两份 256 项私有快照，尚未实现行内连续分布。
机器码不是纯 LLVM 源码大小估算；[反汇编](code/object.asm.gz)和
[LLVM](code/kernel.ll.gz)原样压缩保存。代码捕获的额外诊断耗时不用于下面的 JIT 表。

## 3. 与旧 XIR 的交错 Runtime A/B

W8/requested 8 CPU workers、AB/BA 两轮、每 visit 7 样本/30 ms、warmup 100 ms。
这是同步 Runtime command-list batch 的热 host-wall 时间，非纯 kernel；不含 JIT、
上传或初始分配。32/32 visit 完成两次 native 全输出/guard 和独立 Python FP64 检查。
表中耗时是轮内 p50 再取中位数；比率是逐轮 old/new 再取中位数。

| 算子 / rows×width | 旧 XIR µs | 本轮 µs | paired old/new |
|---|---:|---:|---:|
| RMSNorm 17×7 | 0.341 | 0.343 | 0.994× |
| RMSNorm 17×127 | 8.335 | 5.557 | 1.500× |
| RMSNorm 64×256 | 93.713 | 54.893 | 1.710× |
| RMSNorm 1024×256 | 383.438 | 157.826 | 2.430× |
| RMSNorm 64×513 | 145.240 | 69.983 | 2.077× |
| LayerNorm 64×256 | 112.560 | 70.842 | 1.590× |
| Masked softmax 17×65 | 3.878 | 8.390 | 0.462× |
| SwiGLU 17×65 | 4.373 | 5.644 | 0.775× |

**Softmax/SwiGLU 回退保留，不用综合平均数掩盖。** RMSNorm 17×7 的不足 1% 差别
不解释为实质回退。两轮只是描述性证据，不是置信区间；实现同时改变表示、部分累加器
和大数组资源策略，不能把速度提升单独归因于某一个参数。

普通 JIT 的中位数：RMSNorm 64×256 从 5806.4 → 80.7 ms；64×513 从
25722.1 → 33.3 ms；LayerNorm 64×256 从 8927.7 → 50.3 ms。
这是宿主编译成本，不与 kernel 时间相加作为加速比。详见 [A/B 原始记录](ab/report.json)。

## 4. 大尺寸与其他 LLM 算子：仍落后 Torch

[大尺寸结果](large/results.json)使用同输入、FP32、两种交错顺序、7 样本，44/44 visits
全部通过 FP64。下表是请求 8 线程的 Runtime/eager Torch E2E batch 时间；Torch 各算子
是否包含输出/中间分配在 `measurement.expression` 中记录。它不是 Inductor 排行榜。

| 算子 / 尺寸 | XIR µs | eager Torch µs | paired XIR/Torch |
|---|---:|---:|---:|
| RMSNorm 17×1537 | 65.216 | 18.359 | 3.55× |
| RMSNorm 1024×4096 | 3690.560 | 1155.979 | 3.19× |
| RMSNorm 64×16384 | 2094.917 | 363.749 | 5.77× |
| LayerNorm 64×4096 | 947.785 | 81.377 | 11.65× |
| LayerNorm 17×16384 | 1943.884 | 83.356 | 23.32× |
| Masked softmax 64×4096 | 1212.816 | 243.662 | 4.98× |
| Masked softmax 17×16384 | 2808.185 | 246.898 | 11.37× |
| SwiGLU 1024×4096 | 5222.071 | 1636.703 | 3.19× |
| GELU + residual 64×1537 | 319.612 | 214.014 | 1.49× |
| RoPE 64×4096 | 523.653 | 287.877 | 1.82× |
| causal GQA 2,4,2,16,32,32,32 | 239.762 | 63.829 | 3.76× |

GQA 维度顺序是 B,Hq,Hkv,Q,K,D,Dv，固定 query/key block=4/8、bottom-right causal
mask。它的普通 JIT 仍约 7.81 s：有界化大域并没有消除多个小域嵌套展开的所有成本。
其余此 cohort 的行算子 JIT 为 27.6–165.5 ms。旧版本的 1537/4096 timeout 和 16384
expansion-budget 失败见[前一报告](../m1-max-20260908-xir-indexable/notes.md)，不能为失败
填造耗时或计算“大尺寸相对旧版”的加速比。

### 下一步具体该解什么

1. **连续局部分布**：原生单线程仍差 3.21×，说明不是全部都能归咎于 Runtime。
   需要把贡献/元素维分配给硬件向量 lane，保留合法快照/别名处理；不是再补一条 exp intrinsic。
2. **独立 CPU 任务粒度**：当前这组行算子选 32 workers/block；17/64 个 root programs
   只有 1/2 个 Runtime block，线程池以 block 为任务，所以请求 8 线程并不等于用满 8 核。
   非 cooperative 的 packet 分派粒度应成为单独可求解的映射维度，不能被 GPU 风格 block
   约束钉死。此结论来自实际元数据和 Runtime 代码，不是只从耗时猜测。
3. **阶段物化和活跃资源**：去除不必要的 iota/布尔中间数组，比较 exp 保存与重算；
   不能把 `const` 当作 noalias 而直接重读可能被输出覆盖的输入。
4. **候选具备后再校准**：联合选择局部向量分布、展开/部分累加器、任务粒度、资源复用；
   将 kernel 服务时间、调度时间、编译预算分开建模，用 held-out 尺寸/算子验证 regret。
   本轮小 Softmax 回退正是必须加入成本/候选验证的反例。

## 5. 构建、验证与证据边界

使用隔离构建 `/tmp/luisa-reduction-checkpoint.gQUERp/build`：基础源码来自 `f5daf25e6`，
benchmark helper 为 `db52dbc59`，旧 XIR 生产修复为 `3b96c263d`，再覆盖本轮列出的源文件。
主工作区无关的未完成 TIRx/Metal matrix 修改未混入。二进制及相邻共享库在测试后冻结，
A/B 与大尺寸报告均核实运行前后哈希未变。源码身份见 [provenance](provenance.json)。

- 完整所选配置 build 成功后，35 个 Tile tests 加 SIMD LLVM/Runtime widths/thread-pool，
  共 **38/38 CTest 通过**；含宽度 16384 的 Runtime FP64/guards、工作区容量边界和 lane 隔离。
- 严格 fold 逐位、unordered 初始值/尾部/负零、非闭合递归 fallback、零次循环、交叉 carry
  交换、const/writable 别名和 67 个 program 的原地大转置均在回归覆盖内。
- 额外 W1/2/4/16 各自完整运行 Runtime 测试，均通过 9 tests / 1243 assertions；
  不是只选择一个容易通过的测试过滤器。
- 语法检查未报告 error；日志保留已有/新增静态分析 warning，不声称 warning-free。
- 初始数值失败、错误的测试 block 参数、工作区修复前崩溃和中途测试代码 API 拼写错误
  都有对应日志；没有把失败二进制用于最终性能结论。

[audit.py](audit.py)独立重算分组中位数/逐轮比率、输入身份、检查覆盖与失败保留，
并解析实际 Mach-O 大小；包含故意损坏报告的拒绝测试。它不替代测试执行或特殊浮点、
所有布局、所有 LLM 算子族的验证。文档仍归属既有 Sphinx internals/performance 结构，
没有新建第二套 Tile 网站。

Doxygen XML 已重新生成；fresh Sphinx `-W` 成功，50 个 HTML、4171 个本地链接/资源、
199 个兼容锚点通过。1280/390 像素页面检查及人工截图复核完成；宽表/代码块使用页面内
滚动容器。生成 XML/HTML 和 benchmark tensor 不提交；原生源码、汇编、哈希、JSON 与
验证 receipt 保留。
