# 从 Torch 的标量基址到通用 SIMD 私有向量访存

## 结论：明显缩小 native 差距，但还没有追平 Inductor

2026 年 9 月 8 日，Apple M1 Max、FP32、W8。对照实际 TorchInductor
C++/ARM64 代码后，本轮修复了 SIMD 后端的一项通用实现缺口：已经采用
`[slot][lane]` 布局、并且当前参与 lane 访问共同 slot 的私有数组，现在
保留标量分配基址，生成连续向量读和保留 inactive bits 的向量写。

固定整程序映射时，64×256 RMSNorm 的单线程 native 入口由 **28.623 降至
11.511 µs**，1024×4096 由 **7380.091 降至 3336.237 µs**。
对应 Inductor 仍为 9.157 / 2241.350 µs：逐轮配对的耗时比约为
1.26 / 1.50，**没有达到性能目标**。这是后端访存 realization 的改进，
不是新 planner/solver 或成本校准；执行映射、私有布局和前端 kernel 都没改。

## 范围与计时边界

- Native：实际 Luisa ORC object 与实际 Torch 2.14.0 Inductor 入口；一个
  CPU 线程。三种入口的六种排列全部执行，每次预热 100 ms、7 个目标
  30 ms 的样本，共四组、72 次访问。C++ 计时循环排除 Python、Runtime、
  线程池和分配，但包含 native call、Luisa launch-record 的少量重置。
  这是 native-entry host wall time，**不是 CPU cycle counter**。
- Runtime E2E：15 个算子/尺寸，分别固定 whole-program (`local_lanes=1`)
  和 packet-local (`local_lanes=8`)，每种映射只切换新的访存实现；四种
  组合按正序/逆序各跑一轮，共 120 次访问。W8、请求 8 个 CPU workers；
  每次预热 75 ms、7 个目标 20 ms 的样本。排除 JIT/上传/预分配，包含
  dispatch、任务调度和同步。请求 workers 不代表每种尺寸都会使用满 8 核。
- 单位均为 µs，展示值是各访问 p50 的中位数。配对比值独立逐轮计算，
  不用展示中位数相除冒充配对估计。范围是描述性波动，不是置信区间。
- Runtime 在计时前后检查全量 FP64 oracle 和每次 34 个哨兵，Python 再
  全量复核；每次 native 访问检查全量输出和 68 个哨兵。容差保持
  `atol=rtol=5e-5`，Tile 全局 fast-math 关闭。Torch 使用本机默认编译配置。

## 纯 native：四组固定映射开关对照

下表每一行是独立的三入口排列实验，不把不同行的 Torch 波动解读为性能变化。
`program` 为每个 lane 执行完整程序；`local` 为一个 packet 协作执行一个程序。
所有 Luisa 列都采用相同的 interleaved 私有布局。

| RMSNorm | 映射 | gather/scatter 对照 | 新私有向量访存 | TorchInductor |
|---|---|---:|---:|---:|
| 64×256 | program | 28.623 | 11.511 | 9.157 |
| 64×256 | local | 29.569 | 14.338 | 8.881 |
| 1024×4096 | program | 7380.091 | 3336.237 | 2241.350 |
| 1024×4096 | local | 7438.575 | 3235.974 | 2147.953 |

两种执行映射都受益，但不能因此把 packet-local 设成默认：小 RMSNorm 的
local native 仍更慢，宽行 Runtime 的收益又混入了 CPU task 粒度的变化。
默认保持 `local_lanes=1`；联合映射搜索依然是实验性选择。

## Runtime E2E：全部 15 个 case，包括没有收益的项

`旧/新` 只表示关闭/开启共同 slot 的连续访存，布局均为 interleaved。
这张表不是新一轮 Torch E2E 排名，更不能与上面的单线程 native 时间混用。

| 算子/尺寸 | program 旧 | program 新（默认） | local 旧 | local 新 |
|---|---:|---:|---:|---:|
| RMSNorm 1×4096 | 33.395 | 14.443 | 7.326 | 3.277 |
| RMSNorm 17×65 | 2.892 | 1.296 | 34.353 | 34.312 |
| RMSNorm 17×16384 | 609.760 | 279.103 | 170.295 | 103.263 |
| RMSNorm 64×256 | 50.176 | 33.532 | 59.350 | 42.551 |
| RMSNorm 1024×4096 | 1056.937 | 519.427 | 1071.512 | 515.554 |
| LayerNorm 1×4096 | 60.986 | 22.984 | 13.120 | 4.974 |
| LayerNorm 17×16384 | 1127.431 | 428.225 | 265.932 | 134.231 |
| LayerNorm 1024×4096 | 1934.415 | 781.371 | 1911.667 | 752.391 |
| Masked softmax 64×4096 | 834.136 | 525.549 | 257.206 | 163.488 |
| SwiGLU 17×65 | 4.957 | 3.474 | 34.214 | 33.665 |
| SwiGLU 1024×4096 | 1832.169 | 1279.332 | 1424.653 | 844.293 |
| GELU + residual 17×65 | 8.052 | 6.041 | 35.391 | 35.560 |
| GELU + residual 1024×4096 | 3078.632 | 2294.051 | 2751.039 | 2060.271 |
| RoPE 17×66 | 1.864 | 1.887 | 35.530 | 34.951 |
| RoPE 64×128 | 43.200 | 43.907 | 39.988 | 35.024 |

固定 program 映射：五个 RMSNorm 的配对新/旧耗时比为 0.43–0.67，
三个 LayerNorm 为 0.38–0.40，softmax 为 0.63，SwiGLU 约 0.70，
GELU 约 0.75。两种 RoPE 的 program 路径没有私有数组，发射计数为零，
没有触发本轮变换；仍如实保留测得的约 1.2% / 1.6% 变慢。

local 的窄行不是新的普遍胜利：17×65 RMSNorm 几乎没变化，GELU 的
配对比中位数反而为 1.005，且两轮一快一慢（0.982–1.028）。它们仍有
varying CFG 和 CPU task/唤醒成本。只改访存不能弥补整套映射选择的失配。
这也是保持默认 program、保留 local 候选而不按算子名字兜底的原因。

## Torch 实际生成了什么

本轮重新编译并归档了实际 [Inductor C++](native-64-l1/inductor.cpp)、
动态库和反汇编；不是手写的类似 kernel。它的结构为：

```text
一行输入 ── 连续 4×FP32 读 ── 向量平方/累加 ── 水平归约
    │                                            │
    └──── 输出循环重读 ── × gamma ── × reciprocal(scale) ── store
```

对当前实现有三点直接启发：

1. **地址事实要保留到机器代码。** 共同基址/slot 不应被拆成完整指针向量，
   再通过动态 lane 提取还原。新路径复用当前 cohort 的 seed，基址则直接
   来自不可变的 allocation；GEP 的保存偏移保证跨 block 的地址快照不变。
2. **向量 IR 不等于便宜的机器指令。** 第一版直接生成 masked vector
   load/store，并从 masked handles 重构地址。初步 capture 中小 RMSNorm
   反而从约 27.9 变成 32.6 µs。实际 ARM64 暴露了额外的掩码、`rbit/clz`
   和向量地址暂存。这一版没有采用，原始 LLVM/object 和记录保存在
   `capture/`；这些单次 capture 数字不是 balanced native 性能结论。
3. **下一步是阶段与算术，不只是 buffer layout。** Torch 把输入读与
   归约放在一个循环，输出阶段重读；本路径仍保存两个 Tile snapshot。
   Torch 还先算 `1/sqrt` 再乘，而当前 Tile 测试表达式明确写成逐元素除以
   `sqrt`，实际 ARM64 保留 `fdiv.4s`。不能把这种舍入策略差异隐藏成免费的
   编译器等价变换，也不能无 alias/effect 依据地删除 snapshot。

修正后的实际 ARM64 已有成对的 `ldp/stp q` 私有访存，LLVM 有 10 个
静态 eligible reads 和 2 个 eligible writes（64×256 program capture）。
这些是静态发射计数，可能受 region versioning 影响，**不是动态访存次数，
也不包含每个 masked store 为保留 inactive bits 所需的额外读**。

## 合法性与通用性边界

变换基于封闭地址树和 value class，不查 RMSNorm、LayerNorm 或其他算子名字。
只接受 `private alloca -> typed scalar GEP -> load/store`，元素为 4/8 字节；
aggregate access、地址逃逸、address PHI、shared memory 保守回退。

对 `address[lane] = base + (q*W + lane)*sizeof(T)`，warp-uniform 的 q 可跨
Schedule block 使用；cohort-uniform 的 q 只在 GEP 与访问同 block 时采用。
不同动态 cohort/epoch 的相等性不能互相替代。变换使用保存的 handle offset，
不重新求值可能已经变化的 index。

封闭的 packet-private allocation 在每个 slot 都有 W 个 lane 的实际空间。
因此可完整读向量，再将 inactive 值选为零；写时用旧值保留 inactive bits。
空 cohort 使用 slot zero。该规则不适用于外部 buffer 或 shared memory，
不许可越界读、不扩大可观察写，也不改 snapshot lifetime 或 effect 顺序。

## 成本模型与后续计划

当前新增的是机器访存候选，不是已经求解任意时空资源映射。模型仍需分别估计：
私有读写/保留旧值的流量、active-mask 密度、目标 ISA 的 predication 能力、
地址投影开销，以及 LLVM 优化后的控制流/寄存器压力。静态计数只提供可检查
的 realization 信息，不能直接当作 cycles 或替代独立校准。

优先继续做有 effect/alias 依据的 snapshot/归约阶段融合，核对同样数学策略
下的 reciprocal/除法成本，并把 CPU task 粒度与 SIMD packet 映射分离。
随后以独立微基准校准，在 held-out 算子/尺寸上检查 solver 的选择后悔值。
没有重测的 attention、GEMM、CNN/filter、sort/Top-K，以及 Metal/MPS/BLAS
都不在本轮性能结论内。

## 复现与验证

`measure.py` 固定开关，不按时间选择 kernel；native replay 复用上一检查点的
`replay_native.py/.cpp`，只调用实际生成入口，不另写 RMSNorm 实现。
`archive.py` 保存隔离构建来源、源码/二进制身份、原始日志、LLVM、机器码和
实际 Inductor 源码；大 tensor 仅归档指纹，全量数值检查发生在原始测量期间。
第一版被拒绝的实现保留实际生成物，但没有完整 C++ 源码快照。

隔离的完整构建通过，38 个选定 Tile/SIMD CTest 全部通过（237.34 秒）。
新增测试覆盖 W2/4/8/16、32/64 位数据、0..W 个活跃 lane、排除 lane zero
的分支、最后一个 slot、varying index 和跨 block cohort index 的回退；
整个工作区及前后 64 字节 guard 逐字节比较。两个修改后的 C++ 源文件的
clangd 语法检查通过。详见 [验证记录](validation.json)。

[独立审计](audit.json) 重算全部 120 次 E2E / 72 次 native 访问，核对
完整 case 集、输入/生成物身份、计时边界和数值检查范围，并拒绝八种被篡改的
证据。审计不声称仅凭 tensor 指纹就重新验证未归档的数值。

Doxygen XML 已重新生成；配置/源码警告仍在，不宣称 warning-free。
全新输出目录的严格 Sphinx 构建通过，50 页的 4179 个本地链接/资源、
199 个兼容锚点通过检查。结果表和新增架构段落的桌面/手机呈现已检查，
宽表最后一列、代码图右侧均可水平滚动到达，页面没有整体横向溢出。
截图检查记录见 [docs-qa.json](docs-qa.json)；生成 HTML/XML 不提交。

本机后台应用没有被关闭。有限的两轮多线程访问不足以证明几个百分点的稳定
收益；原始样本和慢项均保留。本轮只覆盖 M1 Max、W8、FP32 性能；32/64 位
存储与 W2/4/8/16 的正确性测试不能替代其他 CPU/ISA 的性能验证。

报告沿用仓库既有 Sphinx 技术文档结构，不增设平行站点。精确比较采用表格：
小/大矩阵跨越多个数量级，将它们挤进统一速度排名会掩盖计时边界和控制变量。
