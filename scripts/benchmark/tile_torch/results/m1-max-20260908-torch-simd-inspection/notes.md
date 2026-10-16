# Torch CPU SIMD 实现检查：先修候选结构，再校准成本

日期：2026-09-08；Apple M1 Max / macOS 26.6.2。这里的 SIMD 指 CPU ARM64/NEON，
不是 Metal 的 SIMD-group。本次是**实际调用栈、生成代码与机器码检查，不是性能榜单**。

结论：Torch 的优势结构不是某条神奇的 intrinsic，而是**行间线程并行、行内连续向量循环、
有界的累加状态和分阶段存储**。我们的直接 XIR/SIMD 候选尚不包含这套分布。更严重的是，
动态 Tile 提取的静态选择链在最终机器码中仍然存在；单改 planner 权重不能消除它。
我们的后端已有向量数学实现，不应把下一步误写成“补齐向量 exp”。

## 1. 检查了哪些真实实现

Torch 为 `2.14.0`，源码 commit `08187d9e0fba026dc8217405802ab5381dc88d90`。
两个 uv 环境的 `libtorch_cpu.dylib` SHA-256 均为
`ca979f3619acb92240c2ef46d7581b1a388761e0fccfa7dc491a7eaa153d584a`。
[环境记录](aten/inspection-64x256.json)保存编译配置：OpenMP、请求 8 线程、interop 1、
Accelerate BLAS、CPU capability `DEFAULT`。**DEFAULT 不等于没有 NEON**：实际执行的
Softmax/LayerNorm FP32 函数有 `.4s` 指令，Softmax 调用 `Sleef_expf4_u10`。

对 1024×4096 和 64×256 分别执行五个 eager 算子；GEMM 两次均为 1024³，不随行算子尺寸变化。
用 ATen profiler 识别算子调用，再用 macOS `sample` 确认运行中的 native 函数。
对 Softmax/RMSNorm/SwiGLU 另外执行 `torch.compile(backend="inductor", fullgraph=True)`，
保存其生成 C++ 并反汇编 RMSNorm worker。六个 compiled/eager 完整输出比较均通过
`atol=rtol=5e-5`；eager 本身仅在此脚本中检查 finite，不冒充独立数值 oracle。

| 算子 | 本机实际 eager 路径 | 本次 Inductor 观察 | 对我们的启发 |
|---|---|---|---|
| Softmax | `_vec_softmax_lastdim<float>`；NEON + SLEEF exp | 行间 OpenMP，feature 每次 4 项；max、exp+sum、normalize 三段，线程私有 exp scratch | 同一 Tile 可分阶段生成，昂贵的共享表达式有明确物化选择 |
| LayerNorm | `LayerNormKernelImplInternal<float>`；向量行统计和输出 | 本次未 compile 此算子 | Welford/分层合并值得借鉴，不能只看 sum 吞吐 |
| RMSNorm | `_fused_rms_norm` 内实际仍有 pow/mean/sum/rsqrt/mul 等 ATen 调用 | 两段有界向量循环；先 sumsquares，再 normalize×gamma | eager 名称不是融合证据；后续性能比较应增加 compiled 基线 |
| SwiGLU | `aten::silu` + `aten::mul` | 扁平连续向量循环，一次读 x/u，一次写输出 | 纯 pointwise 可分布整个元素域，无须保留每行一个大私有 Tile |
| GEMM | `cpublas::gemm → SGEMM (libBLAS.dylib)` | 本次未 compile 此算子 | 与 Torch GEMM 比较实质包含 Accelerate provider；不能把它说成普通 NEON C++ 循环 |

证据分别在 [Softmax native sample](aten/softmax.sample.txt)、
[LayerNorm sample](aten/layernorm.sample.txt)、[RMSNorm trace](aten/rmsnorm.trace.json)、
[GEMM sample](aten/gemm.sample.txt)，以及 [Inductor Softmax](inductor/1024x4096/softmax.cpp)、
[RMSNorm](inductor/1024x4096/rmsnorm.cpp)、[SwiGLU](inductor/1024x4096/swiglu.cpp)。
sample 部分栈存在未知符号/不完整 unwind，只使用可确认的内核与 provider 帧；
**不能据 opaque Accelerate 地址断言使用了 AMX，也不把采样计数解释成精确耗时比例**。

对应版本的 [ATen Softmax 源码](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)
显示每个线程处理行区间，行内做向量 max、exp、sum 和缩放；
[moments_utils.h](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/moments_utils.h)
使用 Welford 与 cascade 合并改善行统计的数值稳定性。不能因数学上等价就把我们的
两遍均值/方差改写成 Welford 并宣称保持所有严格 FP 语义；需单独定义合法变换/算法选择。

## 2. 与直接 XIR/SIMD 的结构差距

Luisa 使用独立构建 `/tmp/luisa-reduction-checkpoint.gQUERp/build`，生产 bridge/backend
来自 `f5daf25e6`；测试 helper 的 QK probe/元数据改动随后提交为 `db52dbc59`，不影响
本次行算子的捕获。未混入主工作区的未完成 matrix 实验。
[构建产物指纹](xir/build-artifacts.sha256)保留实际 benchmark、XIR bridge 和 SIMD backend。
这是局部 provenance，不是全部外部依赖的可重现构建证明。

64×256 RMSNorm/SwiGLU 均通过两次完整 FP64 对照和 34 个 guard 检查；各次检查 16384 个元素。
请求 W8、8 CPU workers，planner 选 32 programs/block、root order `[0]`，记录的
`contiguous reads=0`。W8 是逻辑 packet，AArch64 可拆成两个 4-wide NEON 向量，
不能与 Torch 的物理 4-wide 指令按“宽度大一倍”直接比较。

### 2.1 行内动态提取变成了二次工作量

[lower.cpp](../../../../../src/tile/bridge/xir/lower.cpp) 当前用 `vector<XIR Value*>`
表示 Tile 的每个元素；`TILE_EXTRACT` 为每个可能的 flat index 生成比较和 SELECT。
reduce 又保留一条运行时循环。这个组合的问题已不只是源码猜测：

- [RMSNorm 完整汇编](xir/rmsnorm-64x256.s.gz) 的 `LBB0_2308` 内包含从 0 到 255 的
  选择比较，末尾累加并循环 256 次；即 **256 项选择 × 256 次迭代**。
- [实际 ORC object 的反汇编](xir/rmsnorm-64x256.object.asm.gz) 对应循环位于
  `0xcbec..0xf3c8`，并未被机器优化还原成直接索引。这里的选择工作是 O(D²)，
  不是 reduction 算法本来需要的 O(D)。
- 实际函数序言分配 `80 + 32768 + 256 = 33104` 字节栈；大量 spill/reload 与按 lane
  的 gather/guard 留在输出中。捕获 object 的整个 `__text` 为 214252 字节。

相同 64×256 的 [Inductor RMSNorm worker](inductor/64x256/rmsnorm-worker.asm) 则用
`ldr q` 连续加载、`fmul.4s/fadd.4s` 累加，最后水平归约一次；该 worker 栈框架为
208 字节，归约结果另写在输出临时数组，不能把框架大小当作总工作集。这里不计算
跨 runtime 的代码尺寸或栈大小“加速比”。

### 2.2 Pointwise 也不该整行展开

[SwiGLU LLVM](xir/swiglu-64x256.ll.gz) 已有 **256 个静态 v8 exp 调用点**，使用
`__luisa_cpu_native_exp_f32_v8_u10`，不是逐 lane 调用 scalar `expf`。
[最终汇编](xir/swiglu-64x256.s.gz) 也无 `expf` 调用；向量数学体被展开进巨大函数，
捕获 object 的 `__text` 为 345696 字节。问题是为整行建立 256 份静态表达式，再把
每份沿不同 row 打包。TorchInductor 则在一个有界循环中沿连续元素复用同一向量计算体。

因此当前 [planner.cpp](../../../../../src/tile/bridge/xir/planner.cpp) 虽会给动态
Tile 提取计入成本，但候选只有 root 轴序和 block 宽度；这里只有一个 root row 轴，
求解器选不出 feature 内向量循环。**合法候选缺失，不是再调几个系数就能解决。**

### 2.3 保留没有完成的大尺寸检查

1024×4096 RMSNorm 的汇编导出在最后一次 `ps` 检查时运行了 415 秒，随后手动 SIGTERM，
shell exit 143。保存的 [native sample](xir/rmsnorm-1024x4096-compile.sample.txt)
位于 `emit_assembly_copy → MachineSinking`；[状态记录](xir/large-inspection-status.json)
明确没有完成源码输出或数值检查。后续小尺寸命令使用 60 秒 subprocess timeout。

这是额外的诊断汇编生成，**不是正常 JIT 编译时间、更不是 kernel 时间**。小尺寸 JSON
也保存了 diagnostic compile_ms 和两个 host-wall samples，但这些次数不足、非交错、
输入不同，不能用于本轮 Torch/Luisa 性能比较。不能把大尺寸失败从报告里移除后宣称
完成了相同尺寸的机器码对照。

## 3. 应进入通用 planner 的候选

下面是**待实现的候选表示**，不是新增 DSL 语法，也不是已经校准的 cost model。
设独立输出坐标为 o、贡献坐标为 r、逻辑 packet 宽度 W，每个输出使用 p 个 lane，p 整除 W：

```text
lane l → (o0 + floor(l/p), r0 + l%p)
time t → r0 = t*p

p=1：同一 packet 的 lane 对应不同输出，贡献维逐项推进
p=W：同一 packet 的 lane 对应不同贡献，最终合并为一个输出
1<p<W：同时打包若干输出和各自的贡献；尾部使用精确 mask
```

线程池再分配输出区间，局部展开因子 u、输出 grain、scratch/recompute 策略独立选择。
每个 phase 可以有不同的 p；CPU 水平归约和 Metal subgroup collective 共享逻辑
候选描述，但拥有不同的容量、向量 atom、通信成本和 emitter。

1. **先消除动态 SELECT 链。** 物理值表示允许有界向量、可索引的 compiler-owned
   临时 storage，或在效果/别名条件成立时延迟生成元素表达式。不能因 view 是 const
   就把 snapshot load 移过可能重叠的 store，也不能让用户为编译器临时量显式写 Memory。
2. **加入真正的局部分布。** 纯 pointwise 分布元素域；closed reduction 使用贡献分块、
   多累加器和水平合并。先支持已识别的 sum/max/min；任意自定义 body 保留合法 fallback。
   `unordered_tree` 授权相关重排；fold L/R 不能偷换为无序树，仍可沿独立输出向量化。
3. **按 phase 规划 live state。** RMSNorm 可重读输入避免保留整行 SSA；Softmax 可保存
   exp 避免重复超越运算。选择取决于复用次数、缓存流量、寄存器压力和效果边界，不取决于
   算子名字。布局/alias 证据不充分时保留可索引 snapshot。
4. **再拟合成本。** 服务时间至少分开连续访问/gather、向量数学、水平合并、mask、
   峰值 live state/spill、phase 转换与线程粒度。无重叠时比较
   `T_dispatch + Σ T_phase + Σ T_transition`；有重叠才解资源受限 makespan。
   编译膨胀另设 IR/代码预算；JIT 摊销按预期调用次数进入另一目标，不混成 GPU kernel 时间。
5. **基于机制验证泛化。** 使用 pointwise、sum/max、dot、RMSNorm、LayerNorm、Softmax、
   非连续布局及 ragged tails；至少覆盖 width 1/3/127/256/1537/4096/16384 和不同 row 数。
   先做固定候选交错 A/B，再校准 held-out shape/算子 regret，不能只在这两个尺寸选默认值。

这与 [Metal QK/PV 分布实验](../m1-max-20260908-composed-reduction/notes.md) 的结论一致：
execution calculus 要同时表示独立输出、贡献维和时间分块。不同 backend 的物理答案
不相同，但“先有合法且可生成的候选，再求解它”是共同的结构。

## 4. 借鉴，但不盲目复制 Torch

这份 Inductor RMSNorm 输出循环中，行级 sumsquares 被反复读取，`fsqrt/fdiv` 也留在
每次 4-element 输出迭代内；两个尺寸的 C++/汇编均可确认。地址别名信息不足可能是
阻碍外提的因素，但本次没有收集优化 remark，**尚未确认根因**。
我们的 Tile SSA/效果分析应能在合法时把行级归一化因子保留一次并广播；是否更快仍需测量。
不得凭此给允许重叠的 kernel 参数无条件添加 noalias。

同样，SLEEF 的符号/指令存在不代表我们的数学 provider 较慢；应先修结构，再在匹配
数值契约下比较数学吞吐。后续 benchmark 需分开 ATen eager、Inductor generated native
entry、含 Python/分配的 compiled wrapper，以及 Accelerate provider；CPU 原生入口的
wall time仍包含线程调度，并不自动成为无开销的“纯 kernel hardware time”。

## 5. 复现与验证边界

```sh
uv run --no-project --python 3.13 --with torch==2.14.0 \
  python scripts/benchmark/tile_torch/results/m1-max-20260908-torch-simd-inspection/inspect_torch.py \
  --compile --rows 64 --width 256 --output /tmp/new-torch-simd-inspection
```

输出目录必须不存在；macOS `sample` 用于 native 栈，Inductor cache 单独放入输出目录。
生成源码在本目录 `inductor/` 原样保留，归属对应 Torch 版本；其环境绝对路径不是仓库依赖。
原始大型 XIR LLVM/汇编/ORC object 仅 gzip 压缩，未做格式化或删行。
通过 `llvm-nm --numeric-sort --demangle` 确认函数边界后使用 `llvm-objdump` 导出；
反汇编工具为 LLVM 22.1.8，Luisa 独立构建使用 LLVM 21。两者不要混成同一编译器版本。

[audit.py](audit.py) 核对环境、10 个 eager case、6 个 compiled 完整对照、4 次 XIR
全输出/guard receipt、实际选择循环及 v8 exp 调用点，直接解析 Mach-O section size；
结果与证据文件哈希保存在 [audit.json](audit.json)。这些是一次诊断的证据完整性检查，
不是性能统计检验，也不覆盖特殊浮点输入、所有布局或整个 LLM 算子族。

文档保持在原有 performance/internals 结构：fresh Sphinx `-W` 构建通过，50 个 HTML、
4158 个本地链接/资源和 199 个兼容锚点通过；1280/390 像素两种宽度的渲染检查无页面
横向溢出。日志与 [QA receipt](validation/docs-qa.json) 位于 `validation/`，截图留在
receipt 指向的临时目录。系统 Node 缺少 `libuvwasi` 的失败日志也保留；实际 QA 使用
已安装的 bundled Node 完成，没有修改系统依赖。
**本轮没有修改生产 planner/lowering，没有新增 MPS/Torch 性能达标结论。**
