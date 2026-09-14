# 首消费者归约融合：合法不等于更快

## 结论与提交策略

实现了一条不依赖算子名称的 TileIR→XIR load/reduction 融合规则，以及共享其
admission 的 planner work accounting。它保留 `.load()` 的快照语义，不要求
输入参数 noalias，不改变 reduce 的树、初值或浮点运算。然而，本轮四组 RMSNorm
实际机器码重放全部退化，不能作为新的默认优化：**最终默认关闭，显式 opt-in**。
默认 private-vector 优化和 whole-program mapping 保持上一 checkpoint 的选择。
本轮没有达成超越 Torch/MPS/BLAS 的总体目标，也没有测得新的 Metal 收益。

报告面向编译器开发者。下面先定义比较口径，再给出全部 native 对照和多算子
结果；最后说明模型遗漏、验证边界和下一步。源代码、逐次测量、完整正确性结果
及实际生成物均保留，见 [audit.json](audit.json)。

## 口径：两种时间不能混用

本机 Apple M1 Max，FP32，W8；LLVM 21.1.8、实际 Torch 2.14.0 Inductor。
baseline 和 candidate 都已启用上一轮的 interleaved private layout 与 contiguous
private access。只切换本轮融合；分别固定 local_lanes=1/8，不在测量中选最好 mapping。

- **Native-entry**：C++ 循环调用实际 ORC object 与实际 Inductor 动态库；1 CPU
  线程，排除 Runtime/Python/分配/线程池，保留 native call 与 Luisa launch-record
  reset。不是硬件 cycles。每组采用三种实现的全部六种顺序，每次预热 100 ms，
  七个约 30 ms 样本，共 72 次 visit。
- **Runtime E2E**：包含正常 dispatch/批提交/同步的 warm host-wall 时间；请求
  8 CPU workers，两个相反顺序，每次七个 20 ms 样本、75 ms 预热。15 个算子/尺寸
  × 两种 mapping × 两种开关 × 两个顺序，共 120 次 visit。单次 latency 原始值
  也保留，不与 throughput 混算。本轮这一组没有 Torch E2E arm。
- 时间为各轮 p50 的中位数，单位 µs；paired ratio 是同轮比值的中位数，不是
  表内两个中位数之比。范围仅描述观测，不是置信区间。没有丢弃慢样本。

原始记录分别位于 [native-64-l1](native-64-l1/results.json)、
[native-64-l8](native-64-l8/results.json)、[native-1024-l1](native-1024-l1/results.json)、
[native-1024-l8](native-1024-l8/results.json) 和 [matrix](matrix/results.json)。

## Native：四组融合版均变慢

下面的基线已包含上一轮 private-vector 改进。每组各自重测 Inductor，因此不能把
不同组的 Torch 数字当作同一时刻的配对结果。全部 24 个 candidate/baseline
同轮对照都更慢，融合后的 Torch 差距也更大。

| RMSNorm | Mapping | 未融合 µs | 融合 µs | Inductor µs | paired 融合/未融合 |
|---|---|---:|---:|---:|---:|
| 64×256 | whole-program | 11.087 | 13.521 | 8.869 | 1.219 |
| 64×256 | packet-local | 14.323 | 17.164 | 8.847 | 1.198 |
| 1024×4096 | whole-program | 3205.413 | 3749.899 | 2147.740 | 1.170 |
| 1024×4096 | packet-local | 3233.376 | 3967.917 | 2146.697 | 1.227 |

对应 paired 融合/Inductor 为 1.525、1.940、1.746、1.848。
正确性按全量 FP64 oracle 检查，每次同时检查 68 个保护元素；没有只验抽样元素。
实际 Torch 源码保存在各 native 目录的 `inductor.cpp`，二进制和汇编一起保存。
Torch 使用 reciprocal(sqrt) 再乘，Tile 保留原始除法；本轮不偷偷改变数值策略。

## E2E：存在局部收益，也存在明确负例

这张表用于精确查找同轮时间比值；小于 1 表示融合版更快。只在标记“触发”的
两类 norm 中才能讨论融合效果。其余算子没有触发该规则，观测波动不能归因于融合。
两轮数据不足以支持跨机器、跨尺寸的普遍性能结论。

| 算子与尺寸 | whole-program 融合/未融合 | packet-local 融合/未融合 | 规则触发 |
|---|---:|---:|---|
| RMSNorm 1×4096 | 0.828 | 1.145 | 是 |
| RMSNorm 17×65 | 1.007 | 0.960 | 是 |
| RMSNorm 17×16384 | 1.118 | 1.169 | 是 |
| RMSNorm 64×256 | 1.073 | 0.986 | 是 |
| RMSNorm 1024×4096 | 1.198 | 1.211 | 是 |
| LayerNorm 1×4096 | 0.936 | 1.126 | 是 |
| LayerNorm 17×16384 | 1.012 | 1.156 | 是 |
| LayerNorm 1024×4096 | 1.044 | 1.136 | 是 |
| masked softmax 64×4096 | 1.007 | 0.999 | 否 |
| SwiGLU 17×65 | 0.992 | 0.982 | 否 |
| SwiGLU 1024×4096 | 1.051 | 1.036 | 否 |
| GELU+residual 17×65 | 0.994 | 0.990 | 否 |
| GELU+residual 1024×4096 | 0.963 | 0.991 | 否 |
| RoPE 17×66 | 1.011 | 1.004 | 否 |
| RoPE 64×128 | 0.986 | 0.993 | 否 |

单行 whole-program RMSNorm 14.857→12.305 µs、LayerNorm 23.629→22.128 µs
说明该变换不是始终有害。但 larger/multirow 和 packet-local 的退化阻止其默认启用。
未触发的 SwiGLU 大尺寸也出现约 5% 的比值波动，提醒我们不能把几百分点差异
直接解释成编译器因果收益。全部原始时间、范围与 realization metadata 均保留。

## 合法性来自时序与逐点对应，不来自参数名字

定义 load 的首个实际消费者为：沿现有 single-use pure Tile recipe 的 use-def
链，到达第一个真正读取元素的 operation。融合要求：

1. 消费者是 bounded/distributed 的 closed scalar unordered reduction；允许穿过
   执行一次的单位 map 包装，因此库 `reduce()` 与手写 reduce nest 都适用。
2. 非单位维度的 identity/extent 一一对应，extract 使用相应 reduction coordinate。
   单位维度可以插入/投影，完整多维归约可以调换维度顺序。
3. load 到消费者之间不存在任何写入、stage 或未知 effect。不同参数也可能别名，
   因而不能因为参数名字不同或输入 `const` 就越过写入。
4. 首轮归约每个逻辑元素只读一次。存在后续消费者时，读取同时保存 snapshot；
   没有后续消费者时省掉 snapshot。后续 aliasing store 不改变已有 Tile 值。

```text
load 定义：保存 buffer / origin / fill / bounds 的 SSA 定义
                      │ 无写入/阶段边界
                      ▼
首个归约遍历：read → contribution → 原来的 partial reduction
                 └─ 后续还使用？是：保存 snapshot；否：不分配
```

view bounds/填充值共用原 emitter；pending plan 在每次 lowering 时消费掉，不把
上一轮 host-unrolled 代码的 SSA 定义带入下一轮。严格 fold、不匹配索引、nonunit
wrapper、跨执行 scope 的首次使用、multi-use math materialization 仍保守回退。
没有新增 DSL 实体，也没有把 `parallel` 的独立性重新当作用户必须证明的义务。

## Cost model 的反例：变换组合存在不连续效应

64×256 whole-program 的相同计划中，work prior 的 memory 部分
264192→198656，总分 297112→231576；实际 native 时间却增加约 22%。
这些是相对 work units，不是 bytes、指令数或 ns。

本轮共享 admission 消除了“planner 以为省了、lowerer 却没有实现”的不一致，
但模型仍漏掉了后续编译决策。实际 object 中，未融合 whole-program 只有 batch
entry；融合版保留了独立 `_llm_rows` body。原版被内联而新版没有，full-packet
常量向下传播的机会发生了变化。SIMD 的显式 inlined-loop 路径目前由至少 512-bit
fixed register / 32 registers 的 target query 控制，在本机走另一条调用策略。

[实际 ORC 反汇编检查](assembly-summary.json)显示：64×256 whole-program 的
整个 native text section 条件分支静态站点 76→148，packet-local 为 56→155。
这不是动态分支计数或 misprediction 测量。whole-program 的总静态指令反而减少，
因此不能简单归因于“代码更大”或声称已证明 register spilling 是唯一原因。

更合适的 formulation 是联合选择
`(execution distribution, phase fusion, materialization, partial count,
full-packet specialization, task grain)`，然后评估实现后的地址/掩码/控制流特征。
单独按少一次 private read 给 fusion 一个固定收益，会遗漏这些交互项。
本轮**没有**用这几组数据硬拟合常数或加入 RMSNorm/尺寸名称分支。

## 验证与复现边界

新增 60 次 Runtime A/B 编译/执行覆盖两参数 alias、归约前后写入、保留/消除
snapshot、65/256/4096 宽度、1/3 次重复及 zero-trip、stage 分界、填零边界、
多维/维度置换。结构测试另覆盖 W2/4/8/16、部分可融合输入、严格 fold、cross-scope
回退与 cost accounting。全部选定 38 个 Tile/SIMD CTests 通过；原始记录及
最终默认关闭后的复验分别见 `ctest.log` 与 `shipping-ctest.log`。

`provenance.json` / `source-overlay.patch.gz` 冻结**测量时的实验源码**；
`shipping.json` / `shipping.patch.gz` 记录最终默认关闭、显式开关及测试调整。
二者不能混称同一个二进制。最终 shipping capture 的 LLVM/ORC object 比对结果
单独记录，不把 smoke 时间加入上面的统计。

`measure.py` 是冻结的原始测量脚本；当前默认关闭后复现请用
`measure_current.py`（显式 enable，disable 优先）。`audit.py` 独立校验完整性、
输入 hash、时间边界、paired ratio 与实际对象，并拒绝八种损坏证据。
主目录已有无关 TIRx/第三方修改，构建仅使用记录的 isolated overlay；没有把
它们混进本轮源码或提交。报告沿用仓库 Sphinx 文档结构，没有第二个报告应用。

## 下一步与尚未回答的问题

优先做 bounded full-packet specialization/inlining 候选，并与 fusion/partial
count 正交对照；检查它是否能消除热循环里重复的 inactive-lane 分支，而不让
任意大 CFG 膨胀。再把实际 mask topology、live state 与 worker task grain 的
特征纳入可替换 cost policy，用未参与选择的算子/尺寸验证。

纯 reduction loss/dot 等可完全消除 snapshot 的图还只有本轮正确性证据，没有
同口径 Torch 性能比较；需要补测。masked softmax 的 multi-use score、任意
Tile redistribution、packed GEMM、跨 effect 的安全 reload 也尚未由此解决。
这些是后续工作，不是本轮已经完成的优化。
