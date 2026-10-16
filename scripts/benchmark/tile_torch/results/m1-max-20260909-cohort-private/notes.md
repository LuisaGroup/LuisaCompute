# SIMD private 索引：从逐 lane 访存到连续向量

2026 年 9 月 9 日，Apple M1 Max，FP32。**本轮 6 个固定映射 RMSNorm 的纯 native 耗时均低于本机 TorchInductor；5 个不规则尺寸相对上一轮实现进一步改善约 4.7–5.5 倍。** 这填上了上一份报告中具体的 ragged RMSNorm 缺口，不等于所有 LLM 算子、自动 planner、默认路径或 Metal/MPS/BLAS 的目标已经完成。

## 1. 比较的是同一映射的访存实现，不是重新挑参数

本轮只切换 `LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS` 对应的功能，显式 `LUISA_SIMD_DISABLE_COHORT_PRIVATE_ACCESS=1` 优先。两侧都启用上一轮的 predicated memory effects 和 full-packet specialization，关闭 load/reduction fusion、fast math 与 relaxed precision。W8、local=8、block=32、legacy task grain 固定，不按测量结果筛选映射。

- **纯 native entry**：6 个 RMSNorm 尺寸，每个 baseline/candidate/Inductor 全部 6 种顺序，每次 7 个样本、30 ms 目标采样、100 ms warmup，共 108 次。直接链接实际 ORC 对象，并调用实际 `torch.compile` 生成的入口；只有一个 CPU 线程。Runtime 调度、Python、JIT、分配不在 timer 内。
- **Runtime E2E**：6 种 row 算子 × 9 个尺寸 × local=1/8 × 开关 × 2 个反向顺序，另加 2 个 whole-program attention 尺寸 × 开关 × 2 个顺序，共 440 次。8 个请求的 CPU workers；每次 7 个样本、20 ms 目标采样、75 ms warmup。该边界包含 dispatch 和同步，不称作纯 kernel。
- **代码捕获**：另外 12 次单 worker 运行只用来取得真实 LLVM、对象、输入和 ABI；不把这些捕获时的 Runtime 时间混入性能统计。

Native timer 保留原生函数调用、必需的 block 遍历、Luisa launch record 重置和编译器生成的 libc 调用。它不是剥离所有入口工作的算术循环，也不是硬件 cycle counter。两侧均校验完整输出与 guards。137×1023、513×2051 是本轮在查看时间前加入的新尺寸，但本轮没有拟合 cost model，因此不把它们称为训练/测试分离后的模型泛化验证。

## 2. 不规则 RMSNorm 的已测缺口已经闭合

单位 µs；时间是每轮 p50 的中位数。比值是逐轮计算后的中位数，不能直接用显示时间相除替代。6 轮范围见 [完整表格](tables.md)，不是置信区间。

| RMSNorm | baseline | candidate | Inductor | candidate/baseline | candidate/Inductor |
|---|---:|---:|---:|---:|---:|
| 17×65 | 1.364 | 0.249 | 0.729 | 0.182 | 0.340 |
| 257×1538 | 544.018 | 116.720 | 247.348 | 0.213 | 0.469 |
| 1024×4097 | 5784.021 | 1148.873 | 2593.370 | 0.198 | 0.442 |
| 129×768，对齐控制 | 26.311 | 26.238 | 63.246 | 0.998 | 0.416 |
| 137×1023，新尺寸 | 193.905 | 40.881 | 86.733 | 0.211 | 0.471 |
| 513×2051，新尺寸 | 1456.840 | 283.972 | 655.054 | 0.197 | 0.436 |

36 个 candidate/Inductor 配对轮次均小于 1。对齐控制的微小开关差异不能算作新收益，既有优势也不是本轮创造的。所有结论限于这里的 FP32、单线程、输入分布和固定映射；尚未测 BF16/FP16、全模型或其他机器。

Torch 为本机实际 `2.14.0`，git revision、构建配置、源码和动态库哈希保存在各 `native-*/results.json`。数值合同没有被偷偷改成同一种算法：Inductor 使用 reciprocal-then-multiply，4097 宽度还使用 cascade sum；Tile 程序仍保留除法。两者通过同一个完整 FP64 容差检查，不声称相同 reduction tree 或逐 bit 等价。

### 跨算子的收益与反例

以下仅为 **Runtime E2E**，固定 local=8、1024 行，宽度 4097（RoPE 为其所需偶数宽度 4098）。这支持规则跨算子生效，不证明这些算子的 native 时间已经超过 Torch；本轮只有 RMSNorm 测了 native Inductor 对照。

| 算子 | off µs | on µs | on/off 配对比 |
|---|---:|---:|---:|
| RMSNorm | 889.739 | 307.457 | 0.346 |
| LayerNorm | 1739.144 | 413.998 | 0.238 |
| Masked softmax | 3661.417 | 1913.912 | 0.531 |
| SwiGLU | 1566.149 | 934.094 | 0.597 |
| GELU + residual | 2513.825 | 1566.668 | 0.624 |
| RoPE | 1179.285 | 450.866 | 0.383 |

必须一起保留的反例：

- 17×65 RMSNorm 的 local E2E 为 30.472→32.095 µs，配对比 1.053；whole-program on 为 1.335 µs。纯 native 的大幅进步并不能抵消这种细粒度多线程 dispatch，不能继续把 native mapping 和 CPU task grain 混在一起选择。
- 对齐的 LayerNorm 64×256 local 为 45.780→47.899 µs，配对比 1.046；RoPE 新尺寸 137×1024 local 为 63.087→66.295 µs，配对比 1.048。没有对这些非 native-capture 点保存对象身份，不凭尺寸对齐就把差异宣称为纯噪声或代码退化；它们作为待复核反例保留。
- 两个 attention 点的配对比为 1.022、1.030，两个顺序的方向均不一致，没有稳定收益。没有新 attention 映射，也没有新 GEMM、Metal、MPS 或 BLAS 性能结论。

完整 [tables.md](tables.md) 保留全部 110 个固定映射对照和逐轮范围。尤其 softmax 与部分 whole-program 大尺寸有明显轮间波动；两个顺序只支持描述性比较，不能把所有差异解释为稳定的因果收益。

## 3. 为什么这个规则可以泛化

问题不是 memory 应该决定 execution hierarchy，而是 lowering 丢失了执行上下文中的事实：**某个值跨循环退出需要 varying 存储，不代表它在当前循环体的活跃 cohort 内不相等。**

```text
lane-equal start + constant step
          │
          ▼
body epoch q: active lanes use i = start + q*step
          │
          ▼
GEP use-site equality（不修改全局 ValueClass）
          │
closed private allocation + same-block consumer
          ▼
address(i,lane) = base + sizeof(T)*(W*i + lane)
          │
          ▼
连续 slot vector；空/非前缀 mask 保持原有读写语义

循环退出 / 跨块 pointer / varying start → 不获得这条新许可
```

实现复用 `_lane_index_step` 和现有 `cohort_uniform_operand_index`，没有新 DSL primitive、算子名分支或额外的 reduction 重排权限。Schedule verifier 允许 GEP index 使用该事实，但拒绝 base ordinal、越界 ordinal 和不支持的 opcode。Memory realization 仅消费直接单索引 GEP、封闭的 private scalar array 以及同一 Schedule block 内的 load/store。

地址必须使用已保存的 GEP offset 和不可变 allocation base，不能在循环退出后拿当前 induction 重新计算。shared、逃逸 pointer、PHI/跨块 transport 和 opaque 使用不在这条规则的范围内。现有 warp-uniform 值拥有更强的跨 epoch 性质，不受新规则限制。

## 4. 实际机器码支持访存解释，不冒充采样归因

1024×4097 的两侧 execution mapping、21 个 Schedule blocks、direct CFG 状态相同；静态 contiguous private reads 从 7 增至 13，writes 从 3 增至 4。实际候选汇编的完整区间输出循环现在是 `ldp q` → `fdiv.4s` / `fmul.4s` → `stp q`，以 32 字节步长推进。旧对象在相应区间仍需逐 lane 地址生成和 `ldr/ld1.s` 组装。

这与固定开关实验的性能变化一致。仍然不能用整个对象的静态指令站点数当作动态执行次数：clone、尾部 fallback 和入口内联都影响站点数。实际 LLVM、ORC 对象及反汇编归档在 `captures/`，Inductor 的生成 C++、动态库和入口反汇编在 `native-*/`；[静态检查记录](native-inspection.json) 明确标为非硬件 profile。

## 5. 对 planner / cost model 的要求

本轮是通用事实传递与 realization 改进，**没有重新校准 solver 的成本预测**。1024×4097 两侧估计的 relative work 仍同为 `62030848`，纯 native 时间却约为原来的五分之一。仅在 Tile 语义层数访存次数，无法预测该差距。

后续应联合考虑执行映射 `s`、合法 realization `r` 和 CPU task grain `g`：

```text
logical execution + resources
       → candidate s
       → per-use facts / epochs / masks
       → legal realization r + resource demand
       → target cost policy θ
       → solver selects (s,r,g)

T_native(s,r;θ) ≈ Σ_dynamic_regions count(region,s) × cost(region,r;θ)
T_E2E(s,r,g;θ)  ≈ parallelized native work + activation(g;θ) + dispatch/sync
```

公式是下一步的建模边界，不是已经验证的时间预测器。应把连续/广播/gather 访存、完整/残余 cohort、循环形态、临时资源和寄存器压力作为 realization 特征；native 与 task overhead 必须独立校准。静态访问计数只能充当特征，不能直接当动态总成本。还需用未参与校准的算子/尺寸检查排序准确度和最差退化。

新开关继续默认关闭。需要把本轮与 full-packet、predicated effects、task grain 的相互作用纳入自动候选评估，并扩大 native 对照到 LayerNorm、softmax、pointwise、attention 和 GEMM，才能把局部领先转成用户不手调也能得到的性能。

## 6. 验证与复现边界

源码实现始于 9 月 8 日，最终性能测量在 9 月 9 日。完整构建与 7 个 C++ translation units 的 clangd/tidy 检查通过。全套 CTest 初次为 205/209；64 个 SIMD/Tile XIR 测试全通过。两个 Metal coroutine 超时案例在不改源码的重跑中通过；另外两个 tutorial 因隔离构建不含 fallback backend 而 abort。失败日志保留，不称全套全绿。

开启新规则、predicated effects 与 full-packet 后，W2/W8/W16 的 Tile XIR Runtime 测试各通过 12 项、903053 assertions。新增 codegen oracle 覆盖 W2/4/8/16、32/64-bit private storage、所有 active counts、空/奇/偶/全 mask、不同循环长度与退出索引、不同 start、跨块 GEP；逐元素检查整个 private allocation，并检查输出和 workspace guards。

源码与二进制在测量前冻结，完整 overlay、继承来源和哈希见 [provenance.json](provenance.json)。隔离源码树有前序已记录的 overlays，不冒充干净的父提交 checkout。固定映射的 on/off 输出要求逐 bit 一致；这个要求不跨 whole/local 映射，也不跨 Tile/Inductor 的不同浮点计算树。

[audit.json](audit.json) 独立重读 452 份 Runtime/capture 完整输出并对照 FP64 oracle；116 个固定映射组的输出 bits 完全相同。108 次 native replay 在每次测量中验证完整输出和 guards。Audit 重新计算全部中位数、配对比、范围，并拒绝缺失、重复、failed oracle、错误 median、混合输入、错误 worker 数、错误开关等 7 类破坏证据。Oracle 和 replay ABI header 另核对为冻结父提交中的原样文件。

复现（输出路径必须不存在）：

```sh
uv run --offline --no-project --python 3.13 --with numpy python measure.py \
  --suite broad --binary /absolute/build/bin/benchmark_tile_xir --output /new/raw/broad
uv run --offline --no-project --python 3.13 --with numpy python measure.py \
  --suite capture --binary /absolute/build/bin/benchmark_tile_xir --output /new/raw/capture
uv run --offline --no-project --python 3.13 --with numpy --with torch==2.14.0 \
  python replay_native.py --baseline /new/raw/capture/rmsnorm-17x65-r0-l8-p0 \
  --candidate /new/raw/capture/rmsnorm-17x65-r0-l8-p1 --output /new/raw/native-17x65
```

其余 5 个 native 尺寸重复 replay，然后运行 `audit.py --raw /new/raw` 和 `inspect_native.py --raw /new/raw`。Audit 需要对应测量的源码/binary freeze 身份。大型 `.f32` 原始数组留在临时目录；仓库保存完整测量、oracle 结果、日志、bit fingerprints、源代码和机器码证据。Native 数组不保留，独立 audit 复算统计与核验计时阶段的完整输出/guard 记录，不声称重新读取了 native 数组。

报告继续使用原有 Sphinx Tile 架构和 performance 结构。这里的 exact-lookup 表格用于核对多个固定因素，避免用跨尺寸绝对时间图掩盖小 kernel；所有显示单元格和横向滚动范围在桌面及窄屏渲染中检查。
