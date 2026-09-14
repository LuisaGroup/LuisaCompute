# 从 Torch SIMD 代码到独立的执行分布与临时布局

## 结论：表示有改善，联合 planner 还不能成为默认

2026 年 9 月 8 日，Apple M1 Max、FP32。本轮把 TorchInductor 的实际
C++/ARM64 生成结果用于检查实现，而不是只比较 Python 调用耗时。

落地了两项独立能力：一整个 SIMD packet 协作执行一个逻辑程序，以及
编译器临时数组的 element-major/lane-interleaved 布局。第二项在固定的
整程序映射下改善多种大尺寸算子；第一项能改善 CPU 并行度不足的宽行，
但对窄行有严重反例。**没有达到 TorchInductor 的纯 kernel 性能，也没有
新的 Metal/MPS/BLAS 胜利。**

默认保持整程序映射，启用经过封闭地址树检查的临时数组布局。
`PlannerOptions::local_lanes=0` 才开启实验性联合搜索；设为目标 packet
宽度可以固定行内协作。这样保留了可测试的候选，而不把未校准的估计当作
普适选择器。所有合法性判断基于维度、use-def、操作和访问关系，不查算子名字。

## 范围与计时：两个边界不能混成一个排名

- Runtime 实验：13 个形状/算子、2 种执行映射 × 2 种临时布局，共 104 次访问。
  W8、请求 8 个 CPU workers；按四种组合的正序/逆序各一轮，每次预热 50 ms、
  5 个目标 15 ms 的批次。表中是两轮 p50 的中位数，单位 µs。
  包含 Runtime dispatch、同步和任务调度；排除 JIT、输入上传和预分配。
- Native-entry 实验：实际 ORC object 和实际 Inductor 生成入口，1 个线程。
  三个入口的六种排列全部执行，每次预热 100 ms、7 个目标 30 ms 的样本。
  计时区是 C++ 循环，不经过 Python、Runtime 或线程池，不做分配；仍包含
  native call 和 Luisa launch-record 的少量重置。**不是硬件 cycle counter。**
- Runtime 每次在计时前后进行全量 FP64 oracle 和 34 个输出哨兵检查，另有
  Python FP64 全量复核。Native 每次检查全量输出及 68 个哨兵。
  所有浮点容差仍为 `atol=rtol=5e-5`，没有调宽容差。Tile 路径没有启用
  全局 fast-math；Torch 使用本机 Inductor 的默认编译配置。

## 固定执行映射：临时数组布局收益不依赖算子名字

`program` 表示每个 lane 执行完整程序，`local` 表示一个 packet 执行一个程序。
AoS 是 `[lane][slot]`，SoA 是 `[slot][lane]`。下表列出全部 13 个 case，
不删除慢项；四种完整组合和逐轮比值见 [audit.json](audit.json)。

| 算子/尺寸 | program + AoS | program + SoA（默认） | local + SoA（候选） |
|---|---:|---:|---:|
| RMSNorm 64×256 | 51.943 | 49.498 | 47.139 |
| RMSNorm 17×65 | 2.879 | 2.969 | 34.917 |
| RMSNorm 1024×4096 | 2653.451 | 1203.375 | 1120.759 |
| RMSNorm 17×16384 | 1104.171 | 612.752 | 175.882 |
| LayerNorm 17×16384 | 1833.721 | 1139.470 | 271.925 |
| LayerNorm 1024×4096 | 4026.243 | 2027.935 | 1975.577 |
| Masked softmax 64×4096 | 1122.010 | 850.885 | 297.866 |
| SwiGLU 17×65 | 5.078 | 5.072 | 35.063 |
| SwiGLU 1024×4096 | 3311.135 | 1927.430 | 1439.445 |
| GELU + residual 17×65 | 8.208 | 8.159 | 35.303 |
| GELU + residual 1024×4096 | 4304.535 | 3282.297 | 2940.586 |
| RoPE 64×128 | 45.316 | 45.585 | 39.487 |
| RoPE 17×66 | 2.005 | 1.974 | 36.892 |

固定 `program` 时，大尺寸 RMSNorm/LayerNorm 的 SoA 耗时为 AoS 的约
0.47–0.62 倍；SwiGLU 1024×4096 为 0.58 倍，GELU 和 softmax 约为 0.76 倍。
这些是**同轮配对比值的中位数**，不是展示中位数相除。RMSNorm 17×65
反而慢约 3.1%，RoPE 64×128 差约 0.6%；不能把有限的两轮读成微小收益的证明。

在 SoA 不变时，local 对 17×16384 RMSNorm/LayerNorm 的配对耗时约为
program 的 0.287/0.239 倍。但前者只有一个 CPU block task，后者有五个：
收益混合了地址访问和 CPU 并行度，**不能宣称全部来自向量化**。
17×65 的 RMSNorm/SwiGLU/GELU 和 17×66 RoPE 则慢约 4.3–18.7 倍。
这些尾部候选会产生额外 varying 控制流，并唤醒多个 workers 处理很少的工作。

## 纯 native 入口：目前仍约为 Inductor 的三倍耗时

下面两条 XIR 路径都启用 SoA，避免把临时布局差异当成执行映射收益。
Torch 为本机实际生成的 `torch.compile(..., fullgraph=True)` RMSNorm，
版本、构建配置、生成源码和动态库身份保存在各自的 `results.json`。

| RMSNorm 尺寸 | program + SoA | local + SoA | TorchInductor |
|---|---:|---:|---:|
| 64×256 | 27.858 | 29.914 | 8.972 |
| 1024×4096 | 7204.372 | 7536.289 | 2190.762 |

默认路径分别仍是 Inductor 的约 3.11/3.29 倍耗时；local 在这两个单线程
native 实验中还更慢。小矩阵 Runtime 表中 local 略快，不能用来反驳此结果：
它同时改变了线程池任务数，计时边界和实际并行度不同。

## Torch 生成了什么，以及我们具体学到了什么

实际 C++ 入口可直接查看 [64×256](native-final64/inductor.cpp) 和
[1024×4096](native-final1024/inductor.cpp)，不是手写仿制品。它按行迭代，
内层使用 4 个 FP32 元素的 `Vectorized<float>::loadu`，先做向量累加再进行
水平归约，然后进入向量化输出循环。输出阶段重读输入和 gamma；这里不能
机械照搬到任意 Tile 程序，因为 `.load()` 是有快照语义的 SSA 值。

对应的实现经验是：

1. **保留映射分解。** 对 `e = W*q + lane`，私有槽就是 `q`。降低时保存
   `(q, lane)`，不要先线性化再制造一串 varying-i64 除法来求逆。
2. **执行分布与存储布局独立。** 在 XIR bridge 中决定元素归哪个 lane；
   在 SIMD backend 中选择私有数组的地址排列。不能只把外部读改连续而
   忽略临时数组的写回/读回。
3. **用内置 collective 表示合并。** 闭合 unordered reduce 使用
   `WARP_READ_LANE` butterfly；所有参与 lane 会合后再执行，初始累加值只进入一次。
4. **下一步需要阶段和资源生命周期。** Torch 的两段向量循环比我们的
   snapshot → reduce → gamma snapshot → output 产生更少的中间流量。
   消除 snapshot 或改成重读必须利用 effect/alias 关系，不能从参数 `const`
   或 `parallel` 的无冲突承诺推出不存在相关覆盖。

SoA 使用封闭的 `alloca -> typed scalar GEP -> load/store` 地址树检查；
地址逃逸、PHI、aggregate access 和 shared memory 不参与此变换。它没有
改动用户 Buffer layout、Tile SSA lifetime 或任何前端语法。

## 模型缺口与下一步

当前是有限候选上的精确枚举，不是时空资源映射的通用最优解。该实验暴露了
成本函数尚未覆盖的关键量：CPU task 粒度/worker activation、不同尾部 CFG 的
执行成本、私有布局对应的实际 masked load/store、重复加载和阶段融合。
SoA 私有访问在目前 prior 中仍保守按 gather 估计，不能说已经“完成成本校准”。

下一阶段优先：让 CPU task 粒度与 SIMD packet/逻辑 block 分离；把
共同 slot 的连续访问事实传到 private-memory emitter；在不移动不安全
effects 的前提下组合 load/reduce 和输出阶段。随后用独立微基准估计成本，
以 held-out 算子/尺寸的 top-choice regret 验证 solver，才考虑默认联合搜索。
未重测的 attention、GEMM、CNN/filter、sort/Top-K，以及 Metal/MPS/BLAS
都不在这次性能结论范围内。

## 复现、验证和证据边界

`measure.py` 固定两个因素、两种顺序，不根据测得时间选 schedule。
`replay_native.py/.cpp` 只调用归档实际入口，检查 ABI/形状/符号/guard；
不是另一套 RMSNorm 实现。`archive.py` 保存 20 个隔离构建源码文件的身份、
生成 LLVM/ARM64 object、实际 Inductor 源码/动态库、原始日志和所有 tensor 指纹。
大 tensor 本体不重复入 Git，全量数值验证发生在原始测量期间。

`audit.py` 独立重算两种计时边界、检查 104 次覆盖和相同输入、生成物身份，
并拒绝篡改边界、样本、oracle 数量、输入身份或删除访问的六种负例。
它不伪称仅凭指纹就重新验证了未归档的浮点输出。

实验仅覆盖一台 M1 Max、W8 和 FP32。两轮 E2E 是描述性对照，不是置信区间；
个别多线程访问有明显轮间波动，所有原始样本均保留。Native 使用相同输入，
但两套编译器的舍入/归约树和输出分配 ABI 不同；分配均在计时区外。
不把本轮结果外推为全算子、所有尺寸或多硬件性能结论。

完整隔离构建和 38 个选定 CTest 测试通过；W1/2/4/8/16 的显式运行时
回归各通过 220 个断言，覆盖尾部、归约初始值和 load 快照语义。
Doxygen XML 已重新生成（已有配置/源码警告仍在），全新输出目录的严格
Sphinx 构建、50 个 HTML 页的 4177 个本地链接/资源与 199 个兼容锚点检查通过。
三个修改段落各检查桌面/手机呈现，包括宽表和代码块的水平滚动可达性。
结果见 [validation.json](validation.json) 和 [docs-qa.json](docs-qa.json)。

文档沿用现有 Sphinx 技术文档结构。数值表用于精确查阅形状和两因素组合，
避免把跨数量级的小/大算子画成误导性的统一速度排名。验证门禁的最终结果
另存于本目录；架构入口为 `docs/source/internals/tile/xir.md`。
