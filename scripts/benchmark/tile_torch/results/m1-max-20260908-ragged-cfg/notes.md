# SIMD ragged control flow：真实 kernel 收益与剩余差距

2026-09-08，Apple M1 Max，`codex/tile-programming-design`。本轮结论：**通用 lowering 改进有效，但还没有解决不规则宽度相对 TorchInductor 的差距。** 三个 ragged RMSNorm 的单线程 native 时间约改善 3.3–7.1 倍，仍比 Inductor 慢 1.87–2.23 倍。对齐控制组代码不变，已有优势保留。本轮没有新的 Metal/MPS/BLAS 结果，也没有把实验开关改成默认策略。

## 1. 先把计时边界说清楚

- **Runtime E2E**：344 次访问，6 种 row 算子 × 7 个尺寸 × 2 种映射 × 开关 × 2 个反向顺序，另加 2 个 attention 尺寸 × 开关 × 2 个顺序。FP32、W8、8 个请求的 CPU workers、block=32、legacy task grain、full-packet 开、fusion 关、fast math 关。每次 7 个样本，20 ms 目标采样、75 ms warmup。计时包括 Luisa dispatch/同步，不是纯 kernel。
- **Native capture**：另外 8 次单 worker 捕获，只用于拿到实际 LLVM、ORC 对象、输入和 ABI。它们的 Runtime 时间不加入性能对比。
- **单线程 native entry**：4 个 RMSNorm 尺寸，每个 baseline/candidate/Inductor 的全部 6 种顺序，每次 7 个样本、30 ms 目标采样、100 ms warmup，共 72 次。实际 ORC 对象链接成动态库，实际 `torch.compile` 生成的 C++/`.so` 入口直接调用。没有重写 RMSNorm，也没有把 Python wrapper 计入。

Native timer 保留原生函数调用、Luisa launch record 重置及 LLVM 生成的 libc 调用。旧状态机只导出 `packet_batch`，辅助 C++ 按 `SIMDShader` 的真实方式遍历 blocks，传入整个 block 的 packet 数，由真实 wrapper 处理最后一个不完整 block；新路径导出 `packet_batch.blocks`，直接调用。**这不是剥离所有入口开销后的算术循环耗时，也不是硬件 cycle counter**，但两侧确实计算并校验了同一完整输出，Runtime 调度、Python、JIT 和分配均在 timer 外。

固定映射下只切换本轮 lowering 开关；不根据本轮时间选择更好的映射。主比较的源码和动态库在测量前后保持同一份身份，见 [provenance.json](provenance.json)。构建来自已记录 predecessor overlay 的隔离源码树，不冒充一个干净的父提交 checkout。

## 2. 纯 native 结果

以下为各轮 p50 的中位数，单位 µs。配对比值单独逐轮计算，不是直接相除这张表的显示值；范围是观察到的 6 轮范围，不是置信区间。

| RMSNorm | baseline | candidate | Inductor | candidate/baseline | candidate/Inductor |
|---|---:|---:|---:|---:|---:|
| 17×65 | 9.508 | 1.334 | 0.714 | 0.140 | 1.869 |
| 257×1538 | 1818.533 | 526.531 | 236.831 | 0.290 | 2.224 |
| 1024×4097 | 18361.792 | 5645.630 | 2531.710 | 0.308 | 2.230 |
| 129×768，对齐控制 | 25.616 | 25.579 | 59.315 | 0.998 | 0.431 |

三个 ragged 组的 18 轮都明显改善，但 18 轮也都慢于 Inductor。129×768 的 LLVM 和 ORC 对象在开关两侧逐字节相同，不能把约 0.2% 的计时差说成优化；相对 Inductor 的已有优势也不是本轮新创造的。

Torch 版本为实际本机 `2.14.0`，git revision、编译配置、源码/动态库哈希在各 `native-*/results.json` 中。数值语义差异依旧保留：Inductor 使用 reciprocal-then-multiply，我们的 Tile 程序保留除法；宽度 4097 的 Inductor 还使用 cascade sum。两者都通过同一完整 FP64 容差检查，但不声称逐 bit 等价或相同 reduction tree。

## 3. 泛化到其他算子了吗？

下面都是 **Runtime E2E**，固定 packet-local=8、1024 行；RoPE 按其定义要求偶数宽度，因此用 4098，其余用 4097。开关两侧尺寸相同。

| 算子 | off µs | on µs | on/off 配对比 |
|---|---:|---:|---:|
| RMSNorm | 2623.024 | 859.845 | 0.329 |
| LayerNorm | 4652.261 | 1610.669 | 0.346 |
| Masked softmax | 6290.556 | 3025.538 | 0.481 |
| SwiGLU | 3223.042 | 1407.687 | 0.437 |
| GELU + residual | 4078.573 | 2268.784 | 0.556 |
| RoPE | 3055.139 | 1003.560 | 0.328 |

这说明规则不依赖 RMSNorm 名称，也没有往 DSL 加专用实体；但**不等于所有映射都更快，更不等于这些算子都已经超过 Torch**。这里只给 RMSNorm 新测了 native Inductor 对比。

完整 [tables.md](tables.md) 保留全部 86 个固定映射对照、每轮范围和状态机/direct CFG 信息。必须保留的负面结果：

- RMSNorm 17×65 的 local E2E 仍为 28.119 µs，whole-program 为 1.276 µs。相同 native 改善并不能抵消小任务的多线程调度开销。
- 对齐 64×256 的 GELU local 55.312→59.552 µs，配对比 1.082；17×16384 的一些点也没有改善。只有两个顺序，噪声较大的点不能用一个 p50 宣称稳定退化或收益。
- Attention 只测 whole-program，两个点的 paired ratio 为 1.017、0.965；本轮没有实现 packet-local attention，也没有据此宣称 attention 新加速。

## 4. 通用 compiler 改了什么

```text
partial local interval
  ├─ 单臂 tail if ── exact mask → buffer / private memory / PHI effects
  └─ 后续固定循环 ── start/stride/bound 的 cohort equality → header 分支
                                 │
                    所有 region 均可直接发射
                                 │
                        direct CFG + full-packet clone
```

根因是两个约束叠加：

1. 旧 if-conversion 只接受很小的双臂 diamond；真实 Tile tail 常是单臂 triangle，空臂还可能携带 PHI edge assignment。现在以原有 masked memory emitter 发射空/非空臂，明确保留每条边的 mask 和赋值。
2. 保守的控制流 uniformity 会让 tail 后面的固定计数循环也带上 varying 分类。现有 canonical-loop analysis 证明相同 start/bound 与常量 stride 后，现在能给 header 提供 **use-site cohort-equal predicate**。不是把整个 induction/state 全局标为 uniform；读取条件也使用 active seed lane，而不是硬取 lane 0。

规则可处理 bounded private GEP/load/store、非 volatile buffer write 和可安全推测的浮点运算。shared、atomic、volatile、collective/participant-mask、opaque effect、整数除法、float→int 转换仍拒绝。32 条指令的 cap 仅限制构造规模，不是经过校准的 profitability 阈值。真正 lane-varying 的循环边界继续用状态机。

新功能默认关闭：`LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS=1` 开启，显式 `LUISA_SIMD_DISABLE_PREDICATED_MEMORY_EFFECTS=1` 优先。direct CFG 也能消费已有的 cohort header facts，这部分不依赖新开关；因此不声称整个 backend 的默认代码在所有 kernel 上逐字节不变。

## 5. 看 Torch 生成代码，对下一步有什么启发？

本次检查的是**实际生成物**，不是把 eager ATen 源码当作 Inductor 结果：

- [17×65 Inductor](native-17x65/inductor.cpp) 明确区分 4 元素连续向量区间和 1 元素尾部，reduction 后横向合并；输出也保持 full/tail 分区。
- [1024×4097 Inductor](native-1024x4097/inductor.cpp) 使用 `CascadeSumHelper`，full 与 masked-tail 累积状态分开。不能把所有宽度都简化成完全相同的累加树。
- Luisa 的 actual object assembly 在完整区间循环中仍有向量地址生成、逐 lane 的 `ldr/ld1.s` 组装和 reload；并非只有最后一个元素才需要这些工作。对应 LLVM 存在 private masked gather/scatter。实际反汇编已归档，不用 pre-optimization IR 单独冒充机器码证据。

目前的 header equality 证明只消除了分支调度；private common-slot 识别仍依赖 index 的 uniform/cohort 分类。**“这个 epoch 的循环条件相同”没有自动变成“本次访存的 slot 相同”。** 下一步应为 loop-body 的具体 memory use 建立带 epoch/dominance 限定的索引相等事实，配合 full/tail 区间分割，解锁连续 private 访存；不能为了好看而全局把 varying 值改成 scalar。

[native-inspection.json](native-inspection.json) 记录对象身份和静态观察。静态指令站点因 clone/fallback 重复而可能增多，不代表动态工作增多；这些现象定位了候选问题，但还不是对剩余延迟的采样归因。下一轮需要独立开关、native 时间与跨算子/尺寸验证。

## 6. 对 planner / cost model 的要求

本轮是通用 legality + realization 修复，**不是已经改好或校准了 solver 的成本预测**。固定映射的 1024×4097，两侧 report 都估计 `62030848` 单位 relative work，实际 native 却从 18.36 ms 变成 5.65 ms。仅数 Tile 语义操作不足以预测这类结构差距。

建议后续把求解分成两个相互反馈的层次，而不把 operator 名称作为分支：

```text
mapping candidate s
  → legality / facts: epochs, address equalities, full/tail intervals
  → realization r: scheduled/direct CFG, masked/contiguous access, clone
  → target policy θ: native realization cost + CPU task overhead
  → finite solver selects (s, r, task grain)

T_native(s,r;θ) ≈ max_task Σ_region C(region, active interval, r;θ)
T_E2E(s,r,g;θ)  ≈ T_native + activation(g;θ) + dispatch/synchronization(g;θ)
```

这是下一步 formulation，不是本轮声称实现的 calibrated time model。需要量纲明确的 native/memory/math 成本、合法性约束、临时存储/寄存器压力和负载均衡项；语义可行性与是否值得优化必须分开。校准和 hold-out 验证不能使用同一批点冒充泛化。

## 7. 验证、复现与限制

- 相关完整构建通过；66/66 Tile/SIMD CTest 通过。启用新功能与 full-packet 后，W2/W8/W16 的 Tile XIR Runtime 测试各通过 12 项、903053 assertions。
- 新 codegen oracle 覆盖 true/false 两种 triangle、空 mask 下 null buffer、无效 tail 地址、private/output guards、空臂 PHI、W2/4/8/16、varying-bound fallback；原有双臂、非前缀 mask、32/64-bit private memory 和危险效果拒绝测试也在开启状态执行。
- 352 次 Runtime/capture 输出经 [audit.py](audit.py) 独立重读并完整比对 FP64 oracle。90 个固定映射组在开关/轮次间 output bits 一致。72 次 native 在计时时逐次验证完整输出与 guards；native 数组未留存，独立 audit 复算的是统计值及记录的完整检查，不冒充重新读取这些数组。
- audit 还拒绝 7 类故意损坏的 evidence：缺失、重复、failed oracle、错误 median、混合输入、错误 worker 数、错误开关。
- 两次 preflight 被保留且不混入主结果：第一次拒绝不支持的 packet-only ABI；第二次辅助函数把 W8 block 检查误用于 Torch ABI，导致它拒绝执行。修正 replay 的 ABI 检查后，四组全部从头重跑，未修改 frozen kernel binary。
- 构建有 LLVM dylib deployment-target 警告，syntax check 有已有 deprecated hints；未发现新错误。Doxygen/Sphinx/链接和桌面/窄屏渲染收据另见 `validation.json`，不把 Doxygen 现存 warning 数说成零。

复现入口（当前目录；输出路径需不存在）：

```sh
uv run --offline --no-project --python 3.13 --with numpy python measure.py \
  --suite broad --binary /absolute/build/bin/benchmark_tile_xir --output /new/raw/broad
uv run --offline --no-project --python 3.13 --with numpy python measure.py \
  --suite capture --binary /absolute/build/bin/benchmark_tile_xir --output /new/raw/capture
uv run --offline --no-project --python 3.13 --with numpy --with torch==2.14.0 \
  python replay_native.py --baseline /new/raw/capture/rmsnorm-17x65-r0-l8-p0 \
  --candidate /new/raw/capture/rmsnorm-17x65-r0-l8-p1 --output /new/raw/native-17x65
```

对另外三个 native 尺寸重复 replay，然后执行 `audit.py --raw /new/raw`（源码/binary freeze 身份须与该次测量一致）和 `inspect_native.py --raw /new/raw`。大型、可由测试输入生成器重建的 `.f32` 留在原始临时目录，Git 中保存 bit fingerprints、完整 measurement/oracle/logs 与源代码/对象。报告继续落在原有 Sphinx Tile 架构与 performance 结构，不创建第二套站点。
