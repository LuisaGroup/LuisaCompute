# SIMD 满 packet 特化：映射选择取决于实际代码生成

## 技术结论

本轮找到了一个真正影响 execution mapping 盈利性的后端缺口：**完整 packet 的入口仍然携带动态 active-lane 参数，LLVM 不一定将它常量化。** 增加一份共享的 constant-width 函数后，固定 packet-local 映射、关闭 load/reduction fusion 的 RMSNorm，在 64×256 和 1024×4096 上分别达到 5.334 µs 和 1117.151 µs。相对本轮重新编译的单线程 TorchInductor，配对时间比为 **0.603 / 0.511**，对应的 12 个配对轮次全部获胜。

这不是算子专用替换：实现只修改 SIMD Schedule→LLVM 后端，基于入口范围和 CFG 形态，不查看 kernel/operator 名称；不修改 Tile DSL、归约树、浮点表达式或内存别名约定。

但这还不是“默认路径打败 Torch”：whole-program 映射仍然输；窄行的 packet-local E2E 仍然严重退化；融合也仍然不是最佳选择。**满 packet 特化、local mapping 搜索和 load/reduction fusion 都保持 opt-in。** 下一步应将这个可执行候选纳入联合 cost policy，而不是由本次两个尺寸直接推出全局默认。

## 先明确比较对象与计时边界

- 机器：Apple M1 Max，arm64 macOS；LLVM 21.1.8；Torch 2.14.0。实际平台、Torch 配置和源码/二进制指纹保存在原始结果中。
- 精度：FP32；Luisa fast-math / relaxed-precision 关闭。每次完整输出检查 FP64 oracle，容差 `5e-5 + 5e-5 * abs(expected)`。Torch 的 reciprocal-then-multiply 与 Tile 的 division 仍有舍入路径差异，不能宣称逐 bit 相同。
- `whole-program`：一个 SIMD lane 执行一个逻辑行 program；`packet-local`：一整个 W8 packet 协作执行一个 program 的局部元素。两者是 execution distribution，不是 memory tier。
- `F`：上个检查点的 first-consumer load/reduction fusion；`P`：本轮 full-packet specialization。每组固定 mapping 和 F，仅切换 P。private-array interleaving、contiguous private access 均开启，保持同一套 planner / block-width 决策。
- **Native-entry**：链接 Runtime 实际导出的 ORC `.o`，直接在 C++ 计时循环调用入口；计入 native call、launch record reset 和编译器产生的 libc `memcpy`；排除 Runtime、Python、线程池、分配和 JIT。一个 CPU 线程，不是硬件 cycle counter。
- **Runtime E2E**：8 个请求的 CPU workers，批量 dispatch 后同步的 host-wall µs。实际可并行任务数依赖 block grain；请求 8 workers 不意味着每个 case 都使用 8 个。不是纯 kernel 时间。

native 的 8 个 factorial cells 各运行 baseline / candidate / Inductor 的全部 6 种顺序；每次 warmup 100 ms、7 个样本、单样本目标 30 ms。表中时间是各轮 p50 的中位数；配对比值是同轮比值的中位数，不一定等于表中两个中位数之比。不同表行独立重测 Torch，不能拼成一个无条件的全路径排行榜。

## 满 packet 特化释放了 local mapping 的收益

下表包含全部 native cells，包括未获胜的 whole-program 和融合候选。单位 µs；比值越小越好。来源是各 `native-*/results.json`，独立重算见 [audit.json](audit.json)。

| RMSNorm | Mapping | F | P=0 | P=1 | Inductor | 配对 P1/P0 | 配对 P1/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 64×256 | whole-program | 0 | 11.404 | 11.406 | 9.123 | 1.000 | 1.250 |
| 64×256 | whole-program | 1 | 13.900 | 12.404 | 9.121 | 0.892 | 1.360 |
| 64×256 | packet-local | 0 | 14.593 | 5.334 | 8.840 | 0.366 | 0.603 |
| 64×256 | packet-local | 1 | 18.044 | 8.306 | 9.126 | 0.460 | 0.910 |
| 1024×4096 | whole-program | 0 | 3251.837 | 3251.474 | 2185.154 | 1.000 | 1.489 |
| 1024×4096 | whole-program | 1 | 3807.891 | 3384.539 | 2145.719 | 0.889 | 1.577 |
| 1024×4096 | packet-local | 0 | 3293.971 | 1117.151 | 2184.957 | 0.339 | 0.511 |
| 1024×4096 | packet-local | 1 | 4039.846 | 2091.677 | 2185.819 | 0.518 | 0.957 |

F=0 的 local 两组不仅比旧 local 快，也真正低于本轮 Inductor；P1/P0 的全部轮次范围分别是 0.361–0.372 和 0.336–0.340。相反，whole-program F=0 的 native 时间基本不变。**因此不能将这个收益称为一个与映射无关的“统一 reduction 加速倍数”。**

F=1 也受益于 P，但仍慢于 F=0 的最佳 local 实现。大矩阵 local F1 的 P1/Torch 中位数为 0.957，却有 1/6 轮反转。这继续反驳“去掉 snapshot / 少访问私有内存就应该更快”的模型。所有 144 次 native visit 均检查完整输出与 68 个 guard 元素，未删除慢轮次。

## 不是只对 RMSNorm 有效，但任务粒度问题仍在

E2E 共 15 shapes × 2 mappings × 2 F × 2 P × 2 顺序 = 240 次，全部输出正确。下表展示 F=0 的逐行查阅值：这是固定 local mapping 的 P 对照，不是自动 planner 选择后的总体成绩，也没有在这一轮重测其他算子的 Torch E2E。

| 算子与尺寸 | Local P0 µs | Local P1 µs | 配对 local P1/P0 | Whole P1/P0 |
|---|---:|---:|---:|---:|
| RMSNorm 64×256 | 45.650 | 38.505 | 0.852 | 0.910 |
| RMSNorm 17×65 | 35.975 | 37.690 | 1.047 | 1.000 |
| RMSNorm 1024×4096 | 706.820 | 307.779 | 0.436 | 1.040 |
| RMSNorm 17×16384 | 94.056 | 68.132 | 0.724 | 1.012 |
| RMSNorm 1×4096 | 3.377 | 1.123 | 0.332 | 0.999 |
| LayerNorm 1×4096 | 5.043 | 1.842 | 0.365 | 1.023 |
| LayerNorm 17×16384 | 124.388 | 82.136 | 0.660 | 0.876 |
| LayerNorm 1024×4096 | 1093.969 | 449.250 | 0.411 | 0.799 |
| Softmax 64×4096 | 191.515 | 168.982 | 0.887 | 0.811 |
| SwiGLU 17×65 | 36.902 | 35.664 | 0.966 | 1.011 |
| SwiGLU 1024×4096 | 1270.318 | 1283.392 | 1.010 | 0.999 |
| GELU+residual 17×65 | 37.689 | 36.839 | 0.978 | 0.935 |
| GELU+residual 1024×4096 | 2984.755 | 2137.794 | 0.717 | 0.805 |
| RoPE 64×128 | 37.718 | 36.878 | 0.978 | 0.872 |
| RoPE 17×66 | 35.448 | 36.270 | 1.023 | 0.992 |

宽行 norm 和大 GELU 的改善具有跨算子证据，但 SwiGLU 未表现出一致收益。只看 P1/P0 还会漏掉更大的问题：RMSNorm 17×65 的 whole-program P1 仅 1.314 µs，而 local P1 为 37.690 µs。一次选择更多 SIMD 协作，也可能把原本的单任务执行变成需要线程池调度的多任务执行；不能把 native 的获益直接搬到 E2E。

softmax、SwiGLU、GELU 和 RoPE 不触发 F 的规则；这几种算子的 F0/F1 是重复控制，不应把噪声说成 fusion 收益。仅两个 E2E 顺序不构成统计置信区间；例如小 RMSNorm 的 whole native 基本不变，E2E 却有约 9% 表观变化，说明该粒度下必须慎用单个 E2E 百分比。所有 F1 值、同轮范围、回退计数见 audit 与 `matrix.json.gz`。

## 实现边界与机器码证据

```text
Tile execution mapping + memory realization
                    │
              exact packet range
                    ├── full ── shared clone(active_lanes = W)
                    │             └── LLVM folding / ordinary inlining
                    │                   └── native vectors / contiguous copy
                    └── tail ── original dynamic-active-lanes body
```

后端使用 LLVM 的 [CloneFunction API](https://llvm.org/doxygen/Cloning_8h.html)，只替换 active-lane 参数。特化前 IR 中保留的三个 pointer 参数仍带有原本的 ABI attributes；wrapper 仍然按自己的可变 launch-config 契约执行，不能继承 packet body 的 readonly 属性。所有 full call sites，包括 partial block 中完整的 packet，都复用同一个 clone；tail 不冒充 full。普通 LLVM inliner 可以继续做决定，没有给大型 CFG 的每个 call site 强制内联。

候选仅接纳 direct CFG、静态非空 1D packet range、已启用精确 tail narrowing、W2/W4/W8/W16。原函数最多 4096 条优化前 LLVM 指令，这是克隆构造量上限，**不是**从这两个 RMSNorm 尺寸拟合出的盈利阈值。Cooperative/coroutine、state-machine、standalone packet、非 1D 形状和超预算函数均保持旧路径。

64×256 的 actual ORC object 中，local 无融合版本保留了 `_llm_rows.full_packet` 和连续 snapshot copy 的 `memcpy`；1024×4096 则被普通 inliner 合入 block 入口。对应机器码、全部 text section 的静态指令/分支统计见 `capture/*/native.asm.gz` 与 [assembly-summary.json](assembly-summary.json)。代码拓扑可能改变，所以不把“整个新入口”与“旧的单个 outlined 函数”伪装成同范围比较。事实上这两个 local case 的全部静态指令数由 510→874、515→726 增加，却明显更快：静态分支数、stack store 数都不是动态 profile，不能据此声称已证明 misprediction 或 spilling 是全部原因。

旧 native replay 原本拒绝任何未解析符号，因此在第一个 local P1 的 `_memcpy` 处主动停止，尚未开始该 cell 的计时。本轮复制出独立 runner，只允许这个 Darwin libc 符号，并记录 `otool -L` 的系统库依赖。没有补写一个 RMSNorm，也没有把 copy 移出计时。原先已完成的 36 个 preflight visits 被保存并明确排除；**整个 8-cell 实验重新运行**，不是按结果挑选重跑。见 [preflight.json](preflight.json)。

P=0 的 8 对 LLVM / ORC objects 与上个检查点逐 byte 相同。测试新增空范围、所有 lane residues、非对齐起点、末 block、stack/workspace、guard 和不适用路径；最终完整构建、38 项 CTest 和 clangd 结果见验证日志。

## 对 cost model / solver 的直接启发

当前 planner 的 arithmetic/memory work prior 在 P0/P1 之间完全一样，但 local native 时间可以差近三倍。**这个模型需要知道“某种地址与 mask 在目标上如何实现”，而不只是 Tile IR 有几次 load。** F1 低估 work 却更慢的反例也仍然存在。

一个更合适的下一步分解是：对计划 `s`，为每个实际 CPU task `j` 估算

```text
task_time(j, s) = Σ full_packet_cost(mapping, materialization, target)
               + Σ tail_packet_cost(active_count, residual_masks, target)

kernel_time(s) ≈ max_j task_time(j, s) + native_entry_overhead
E2E_time(s)    ≈ kernel_time(s) + dispatch_and_synchronization(s)
```

这只是下一阶段的 optimization formulation，不是本提交已实现的 calibrated model。必须保留三点：

1. **full execution packet 不等于所有 memory access 无 mask。** 局部 ragged tile 的边界条件还在，不能把 W active lanes 误读成任意地址都合法。
2. 将 `mapping × full/tail state × snapshot/fusion × native memory realization` 作为联合候选特征；target policy 描述 SIMD 宽度、masked access 成本、复制/连续读写吞吐。求解器不应绑定 RMSNorm 名称。
3. CPU task grain 是独立变量。用 max-task critical path 加 dispatch 成本，不要仅按请求的 worker 数平均分摊 work；小任务应考虑 caller-thread 路径。候选按 held-out shapes/operators 验证后再讨论自动默认。

## 复现、校验与后续问题

使用现有 benchmark 及真实导出对象，不需要重写 DSL：

```sh
uv run --offline --no-project --python 3.13 --with numpy --with torch==2.14.0 \
  measure.py --binary /absolute/build/bin/benchmark_tile_xir --output /new/run/capture --capture
uv run --offline --no-project --python 3.13 --with numpy --with torch==2.14.0 \
  run_native.py --raw /new/run
uv run --offline --no-project --python 3.13 --with numpy --with torch==2.14.0 \
  measure.py --binary /absolute/build/bin/benchmark_tile_xir --output /new/run/matrix
```

运行时开关：`LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION=1`；显式 `LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION=1` 优先关闭。映射仍通过已有 planner options / benchmark 控制选择。不要将开关当成新的 DSL entity。

[provenance.json](provenance.json) 记录与隔离构建逐文件核对的 20 项源码身份及父提交；`source-overlay.patch.gz` 只包含本轮自有修改，未混入用户正在修改的 TIRx/Metal matrix 分支内容。全部 visits、测量协议、Torch 源码与 `.so`、actual ORC objects、系统库依赖、输出/input bit fingerprints 均归档。大 `.f32` 留在原始临时目录，不塞入 Git；`audit.py --raw /absolute/raw` 可以再次执行 256 次完整 FP64 output oracle。

报告按现有 Sphinx 页面与中文实验记录交付，不新增另一套文档站；表格用于逐 cell 的精确查阅，不把不同规模混在一个绝对时间柱图。当前结论可带限定分享：它证明了一个通用代码生成候选在两个 RMSNorm native cohorts 的优势和其他几类算子的 E2E 收益，没有新 Metal/MPS/BLAS 测量、跨 CPU 证明、bf16/fp16 结论，也没有证明所有常见 LLM kernels 已达到目标。

后续优先验证：把 full/tail realization 状态接入后端 policy；联合选择 local distribution 与独立 CPU task grain；对更多 norm/elementwise 的 actual native ABI 做 TorchInductor 对照；用动态 profile 区分剩余 snapshot traffic、寄存器压力和数学函数实现成本。fusion 继续作为候选保留，不预设它应获胜。
