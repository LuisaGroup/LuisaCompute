# SIMD native codegen：地址投影有通用候选，全部内联有反例

2026-09-09，Apple M1 Max，macOS 26.6.2（25G83），LLVM 21.1.8。
基线是 `2cfc804932a014cd7a07eb3896166d53db32b3eb`，已合入
`next@03a0f5158b53768abefea555f480f87ee5bc5a1e`。

## 结论：定位了两个 codegen 决策，尚未形成新的性能达标结果

这次完成两组各 24 个案例的实际 native-entry 实验，分别控制
packet-loop inlining（把每个 packet 的函数体内联到 block 循环）和
integer lane projection（只计算最终地址所需的整数向量分量）。
后者在 RoPE 地址路径消除了多余的向量整数运算/向量与标量寄存器搬运，
不识别算子名；前者减少调用、重复别名检查和栈帧，却在大 LayerNorm 上出现回退。
因此值得推进的是**带需求与成本约束的通用变换**，不是全部内联或按算子名选参数。

**两个完整计时 cohort 都从启动前就标为 diagnostic-only。** 桌面存在并发活动，
不能据此宣布追平 Torch、更新性能榜、启用默认策略或校准 cost model；也不挑选
“看起来安静”的轮次重新计入。现有已接受的
[24-case native 结果](../m1-max-20260909-native-rows/notes.md)不被替换。
工作分支的编译器没有改动；下面的两个 C++ 原型只在隔离源码中构建，作为实验补丁归档。

最重要的证据入口：

- [完整 48 行诊断表](tables.md)：含绝对时间、同轮次配对比、六轮范围及所有回退。
- [独立审计](audit.json)：384 份落盘输出复读、144 次 smoke、864 次 timed visits、8 类污染拒绝。
- [构建/代码/命令归档索引](provenance.json)：源码、实际 ORC object、反汇编、Torch 生成源码及 SHA-256。
- [115 项回归 receipt](projection-tests.json)：80 XIR/SIMD + 35 Tile，投影开关强制开启，无失败或跳过。

## 实验边界先固定：同输入、同 mapping，分别只动一个开关

六个 FP32 算子为 RMSNorm、LayerNorm、masked softmax、SwiGLU、GELU+residual、RoPE。
每个算子覆盖 `17×65`、`129×768`、`257×1538`、`1024×4097`；
RoPE 的奇数宽度补到偶数，即 `66/768/1538/4098`。这是既有固定测试集，
最大约 420 万元素，**不是新的 held-out 泛化集**。

| 决策 | Off | On | 两边均保持不变 |
|---|---|---|---|
| A：packet loop 内联 | M1 的既有 outlined 路径 | 强制尝试既有 inlined 路径 | 不运行整数投影；同一个 A 构建 |
| B：整数 lane 投影 | 原 O2 管线 | O2 后增加投影及局部清理 | packet 内联强制关闭；同一个 B 构建 |

两组均固定单 CPU 线程、packet W8、local lanes=8、block=32、
`unordered_reduction_partitions=4`、`fast_math=false`。
既有 full-packet specialization、predicated effects、cohort-private access 和
pointwise fusion 开启；load/reduction fusion、expression/reduction fusion 和
linear-1D block coalescing 关闭。完整环境在各阶段 capture manifest 中。
这些固定 opt-in 不是新 planner 的自动选择，不能归因为 solver 已经变好。

Torch 对照是实际 **TorchInductor 2.14.0** 生成函数，git
`08187d9e0fba026dc8217405802ab5381dc88d90`；沿用已冻结的 C++/library/input，
并重新核对指纹，不是 eager Torch、手写近似算子或把生成 C++ 再改写后计时。

### 测的是 native-entry 主机墙钟时间，不是 Runtime，也不是硬件周期

共同 C++ replay helper 调用从 ORC 截获的 object 所链接的实际 entry dylib，
或原始 Inductor entry。计时排除 Python、Runtime dispatch、JIT 和调用者分配；
保留必要的 native 遍历、callback、launch record 重置、编译器发出的 libc 调用及
entry 内部分配。Inductor masked-softmax 的宽度大小 scratch 分配仍在计时内，
没有人为移出。payload 对齐固定为 64 字节。

每个案例枚举 `off/on/inductor` 的全部六种顺序；每次 visit 七个样本，
100 ms warmup，单样本目标约 30 ms。先取各 visit 的 p50，再报告六个 p50 的中位数。
`On/Off` 是六个**同轮次比值**的中位数，不是显示时间中位数的商。
范围为观察到的最小/最大值，不是置信区间；计数也不表示统计显著性。
A、B 在不同时间执行，不能把 A-On 与 B-On 直接拼成新的配对比较。

## 地址标量需求未贯穿 vector lowering，是可复用的优化机会

以下是 RoPE `129×768` 中观察到的地址表达式的简写；它不是 kernel 数学计算：

```text
row_vector = zext(splat(row)) : <8 × i64>
base       = row_vector * splat(768)
columns    = splat(k * 8) | <0, 1, 2, 3, 4, 5, 6, 7>
address    = extract_lane_0(base + columns)
```

最终只需要一个标量地址，却仍经历向量整数计算与寄存器域搬运。
B 的原型把 `extract` 沿允许的逐分量整数表达式向上投影，再运行
InstCombine/EarlyCSE/DCE。实际 ORC 反汇编中的这部分工作消失，
FP32 向量乘加减仍保留，别名 fallback 的 `memcpy` 也没有被删除。
这是实际 object 的静态证据，**不是硬件 counter 对瓶颈贡献的定量归因**。

局部规则可写成：对于所接纳的逐分量运算 `f`，
`P_i(f(v, w)) = f(P_i(v), P_i(w))`。
它把“一个内存访问只需求布局表达式的哪些分量”带到后端，
与 execution/layout mapping 的需求信息有关，但只是有界的地址表达式规则，
不等于整个 execution hierarchy calculus 已完备或获得形式化验证。

原型接纳固定宽度整数向量的常量、合法 lane index，以及 add/sub/mul、
位运算、shift、整数 cast、select、常量 shuffle/insert；
不复制 load/call/PHI，不重写浮点算术，不跨越内存效应。
新建整数运算不继承原 overflow/exact 标志。设置了递归深度和局部缓存阈值，
但它们**还不是总 code-growth 或盈利性上界**。

生产化之前仍须处理/专测 poison/undef、动态/OOB index、深表达式、
共享向量的多用户，以及其他用户仍需求完整向量时的重复计算成本。
此处通过的是既有回归和有限输入实验，没有新增这些精确边界测试，
不能宣称原型已经是可默认开启的完备 LLVM 优化。

## 内联收益取决于工作摊销，LayerNorm 提供了反例

A 的 RoPE `129×768` Off 有独立的 `full_packet` helper；block wrapper 有七个静态调用点，
其中四个属于满 block 路径，其余处理部分 block。内联后 helper 调用消失，
别名检查移到 packet-loop preheader，**每个 block 一次，而非每次 dispatch 一次**。
观察到的 helper 和 wrapper 各约 6 KiB 栈帧也不再叠加为两层调用。

但是代码体积和 live-value 压力也会改变。下面只用于确定下一步实验优先级，
所有比值仍然是未接受的诊断；小 RoPE 相对自身控制变快，不等于已证明比 Torch 快。

| 诊断案例 | A 内联 On/Off | B 投影 On/Off | 对下一步的含义 |
|---|---:|---:|---|
| RoPE 17×66 | 0.450091 | 0.767042 | 调用/guard 摊销和地址工作都值得检查；联合候选尚未测 |
| RoPE 129×768 | 0.986768 | 0.916398 | 大于一个小任务时，两种决策的收益不同 |
| LayerNorm 257×1538 | 1.074727 | 0.884583 | 内联六轮全部回退；投影六轮全部下降 |
| LayerNorm 1024×4097 | 1.065236 | 0.873537 | 不能用“消掉调用一定好”当默认策略 |
| Masked softmax 1024×4097 | 1.000411 | 0.989797 | 地址/调用改进不足以解释或闭合整体差距 |

其余案例，包括 B 中 RMSNorm `129×768` 的 `1.006517` 和 `257×1538` 的 `1.005124`，
都保留在完整表。如此小的变化尤其不能在并发条件下作确定性的收益/回退结论。
Masked softmax 的 B-On/Inductor 诊断比值仍约 `1.36–1.61`（四尺寸），
继续指向 phase、mask 域和中间结果 materialization 的联合选择，而非只修地址。

### 两条没有采用的捷径

- 对 RoPE 离线尝试关掉 SLP；在该输入上有/无 SLP 的 O2 IR 相同，
  额外 InstCombine/EarlyCSE 也没有消掉目标地址工作。归档的这些 IR/assembly
  是 compiler-only probe，**不是被计时的 native entry**。
- 试探把 precise-exp 的一段舍入替换成 `llvm.roundeven`：M1 的 v4f32 是一条
  `frintn.4s`，baseline x86-64 却变成四次 `roundevenf@PLT` 标量调用。
  因此撤回无条件共享实现的草稿。归档的草稿内虽有拟议测试，
  **那些测试没有编译或执行**；实际执行的只有
  [跨 target 代码生成 probe](roundeven-portability.json)。未全局开启 fast math 或 FMA。

## 证据闭合：两组实验不混用源码和构建产物

两组都从递归固定 Git archive 出发：19 个仓库、19,177 个固定输入文件。
A 只有 `simd_compiler.cpp` 的临时开关；结束后冻结整个 A source 和 15 个本地 binary closure
文件，记录 [重定位及指纹](stage-a-frozen.json)。随后才复用 source/build 进行 B。
B 增加 `llvm_jit.cpp` 的临时投影 pass，A 的开关保持关闭。
两份实验补丁、源码哈希表和各自 full-build receipt 分别归档，不能用 B 的当前目录
声称复读了 A 的源码。24/24 个 A-Off 与 B-Off ORC object 字节完全相同；
B 的 Off/On **pre-O2 LLVM** 也 24/24 相同，差异发生在后续 JIT passes。

完整配置为 SIMD、legacy Metal、DSL、Tile TIRx、tests 开启，Metal4 关闭；
外部 TVM 库沿用本地安装，不是重新构建。A 完成新目录全构建，B 完成同一完整配置的
增量全构建 gate，不只是单 target。B 初次快照误把两个 CMake 生成的 Metal C++
BYPRODUCT 当成额外输入；确认它们与冻结 A 版本逐字节相同后单列 generated 哈希，
重新通过 gate 才执行测试。此前一次 capture 因 gate receipt 尚不存在而拒绝启动，
没有用空输出冒充成功。

校验计数如下，来源为 [审计](audit.json) 和 [测试 receipt](projection-tests.json)：

| 校验 | 每组 | 两组合计 |
|---|---:|---:|
| 捕获阶段完整 FP64 输出 | 48 | 96 |
| 正确性 smoke native visits | 72 | 144 |
| 六种顺序的 timed native visits | 432 | 864 |
| 落盘输出独立复读 | 192 | 384 |
| Off/On bitwise-equal 案例 | 24/24 | 48/48 |

每次 native visit 检查输出、输入不变、workspace guards 和输出哈希稳定性。
审计复读的是保留下来的输出，并检查当时的 guard receipt，**不是事后复读已释放的 guards**。
FP64 容差为 `atol=rtol=5e-5`，使用同一组确定性有限输入；Off/On bitwise 相等
不等于 Tile 与 Torch bitwise 相等，也不等于全实数/NaN/Inf 域准确性证明。
Tile LayerNorm 的 centered-square 与部分 Inductor Welford/cascade reduction，
以及各自 transcendental math 的差异仍然存在。

独立审计还拒绝了零耗时、错误 median、guard 失败、输入变化、输出 hash 错误、
NaN 输出、尾元素损坏、输出截短共八类污染。计时均保留但不接受：周期性进程快照
不能证明硬件独占或恒定频率。公开记录去掉其他应用的名字、路径和 PID，保留
CPU 聚合观察、原始文件 hash 与预先声明的 diagnostic 标志；原始观察只在本地保留。

## 下一步：让 backend 提供成本，solver 组合决策，而不是搬回尺寸特判

这轮没有重写 planner、拟合参数或新增 automatic mapping。
capture metadata 中仍是固定单候选、`custom_cost_policy=false`。
下面是从代码证据得到的**待实现与待验证方案**：

1. 将 lane projection 收紧为需求驱动的地址/索引表达式变换。对仍有向量消费者的
   DAG 计算增量成本；为 code growth、效果边界和特殊值增加独立测试。
2. 把 inlining 与 projection 作为独立但可组合的 realization 维度，记录
   hot/cold 频率、guard/call 次数、向量整数与 scalar 地址成本、寄存器搬运、
   live-value/stack 压力；不要只看总 LLVM instruction count。
3. 保持 backend-owned cost policy：目标后端提供成本/资源信息，通用 solver
   在合法候选中搜索。可使用分相成本的增量比较，但本轮没有校准这些系数，
   也没有证明静态近似足以预测 spilling 或实际频率。
4. 对 inline×projection 联合候选做固定版本、平衡顺序、受控计时；加入未见过的
   算子表达式、layout 和大小，而不把目前 24 个已知案例再叫 holdout。
5. 并行的目标仍包括 masked-softmax phase/materialization，Metal 大矩阵和
   attention 的分阶段 contraction distribution。CPU codegen 候选不是 MPS/BLAS
   或低精度全 LLM parity 的证据；这轮没有新的 Metal 性能结果。

证据优先回答两个开放问题：同一 projection 在完整向量仍被使用时是否盈利？
内联、投影、fusion 和寄存器压力的交互能否由 backend cost policy 在 held-out
程序上正确排序？这比把小 RoPE 的诊断收益直接变成全局默认更接近当前目标。

## 复现与文档归属

这是现有 Tile 性能文档的明细附件，不是另建一套设计文档树。正文入口为
`docs/source/performance/tile/{index,results,validation}.md`，完整诊断数值放本目录，
避免污染已接受性能表。结构采用技术摘要、明确比较边界、代码证据、完整数据、
验证与后续问题；精确映射/比较使用小表，不再增设 dashboard。

`drivers/` 保留实际运行脚本；`inline/`、`projection/` 分别保留补丁、
capture/replay、逐案例实际 object/pre-O2 IR/反汇编与链接命令。
`compiler-only/` 单列未参与计时的输出和撤回草稿。
`provenance.json` 给出内容 hash 与本地来源；大型输入/输出数组、linked dylib、
依赖安装和导出的源码树留在原始本地路径，没有假装归档是可直接跨平台执行的包。
原始 driver 需要显式重定位路径及依赖。重新运行应使用新目录，不能覆盖冻结结果：

```text
从 pinned-repositories.json 的 commit 导出源码
应用所选阶段 experimental.patch.gz，按 configure-command.json 配置
执行完整 configured build gate
capture → verify → 预先明确资格的 replay → 独立 audit
```

用户原有未完成文件及依赖 checkout 的保留指纹在 provenance 中。
本轮提交的是可审计诊断及文档，不是生产编译器补丁、默认开关变化或新的性能达标声明。
