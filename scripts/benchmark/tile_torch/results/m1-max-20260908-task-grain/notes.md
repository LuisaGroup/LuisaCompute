# CPU task grain 必须独立于 SIMD 执行映射

## 技术结论

本轮把 CPU 调度粒度真正接进了 XIR planner → SIMD Runtime：CPU 任务粒度不再只由固定 heuristic 决定，也不必通过改变 native block 大小间接控制。`blocks_per_task` 可以独立固定或参与有限搜索，`ExecutionCostPolicy` 可以替换系数与完整目标函数，而不改变 bridge 的合法性约束。

**这个候选空间确有跨算子价值，但当前简单成本先验未通过普适选优验证。** 在固定 native 代码的 129×768 cohort 中，caller 执行把 RMSNorm 从 52.909 µs 降到 25.662 µs，LayerNorm 从 62.951 µs 降到 42.272 µs，RoPE 从 56.531 µs 降到 27.680 µs；同尺寸的 softmax、SwiGLU、GELU 却应该保留并行。这不是一个按矩阵尺寸设阈值就能解决的问题。

试验性联合模型将小 RMSNorm 64×256 的端到端时间从 34.100 µs 降至 5.544 µs，也改善了小 attention；但它仍会错选 ragged local 映射，并在部分大问题上分块过粗。因此 **task-grain search 与 local mapping search 仍保持 opt-in，试验系数没有进入生产默认值**。

## 比较对象、计时范围与完整数据

- Apple M1 Max，FP32、W8；请求 8 个 CPU workers，并不等于每个 case 都实际调用 8 个 worker。
- 全部候选开启上一轮 full-packet specialization，关闭 load/reduction fusion、fast-math 和 relaxed precision。本轮未改变 primitive 的数值权限。
- 测量为 C++ Runtime 一批 command dispatch 后同步的 host-wall 时间；排除 JIT、分配、上传，但包含线程池和 Runtime。**不是纯 kernel 时间，也不是 hardware cycles。**
- 固定代码实验使用 local=8、block=32，比较 G=0（旧 heuristic）、G=1、G=3 与 G=UINT32_MAX（caller 整段执行）。18 个算子/尺寸 cohort 的 LLVM、实际 ORC object 和完整输出分别逐字节一致；这证明变化属于调度，不能声称 native kernel 本身变快。
- 每次 warmup 75 ms，7 个样本，单样本目标 20 ms。pilot / fixed-grain 使用正反两个顺序；扩展矩阵使用四个预先指定顺序。保留所有慢轮次。表中时间是各轮 p50 的中位数，配对比值是同轮比值的中位数，两者不必完全相除一致。
- 三组对照合计 592 visits：pilot 40、扩展验证 408、固定代码 144。全部重新读取完整输出并对照 FP64 oracle；输入 bit fingerprints、日志和原始时间样本全部保留。[audit.json](audit.json)独立复算，并拒绝六类故意篡改的数据。

[完整逐算子、逐尺寸结果表](tables.md)包含所有比较，而不仅是获胜行。`pilot.json.gz`、`validation.json.gz`、`fixed-grain.json.gz`保留全部原始 records，`captures/`保留固定代码的真实 LLVM 与 ORC object。

## 同一形状的不同原语组合，需要不同 CPU 服务模型

下表固定 local=8、block=32，只有调度粒度不同。单位 µs；最后一列是整段 caller 执行与旧调度的配对比值。

| 129×768 | 旧 task grain | Caller | 配对时间比 |
|---|---:|---:|---:|
| RMSNorm | 52.909 | 25.662 | 0.486 |
| LayerNorm | 62.951 | 42.272 | 0.672 |
| RoPE | 56.531 | 27.680 | 0.490 |
| Masked softmax | 85.638 | 227.854 | 2.661 |
| SwiGLU | 73.753 | 132.546 | 1.797 |
| GELU + residual | 86.084 | 239.004 | 2.776 |

简单 norm/elementwise 的较短服务时间可以低于线程池唤醒成本；包含较多数学函数的 program 则可能值得并行。这里后一句是由跨算子结果支持的解释，不是已经测得的单个 `exp`/`tanh` 指令成本。下一轮需要原语级 math realization 与 actual native profiling，不能让模型把所有 `ELEMENTWISE` 都当成一次同价加法。

同样不能全局使用 caller。1024×4096 的六个 fixed-code cohort 中，caller 比旧调度慢约 3.2–7.1 倍；RMSNorm G=3 则为 258.943 µs，对照 G=0 的 281.013 µs。只有两个顺序，不应把小差异说成确定的统计提升。

## 唤醒成本能解释小任务，但不足以自动选优

试验配置在测量前固定：`local_lanes=0`、`search_task_grain=true`，worker activation=1,000,000、task dispatch=128，其余维持既有相对工作权重。这些值是 **uncalibrated prior**，不是纳秒、硬件规格、回归拟合结果或逐 kernel lookup table。对照 `joint_legacy` 也搜索 local mapping，但不搜索 task grain、不收取这两项费用。

| Case | Joint legacy µs | Joint tasks µs | 配对时间比 |
|---|---:|---:|---:|
| RMSNorm 64×256（pilot） | 34.100 | 5.544 | 0.163 |
| GELU 64×256（pilot） | 58.386 | 39.814 | 0.682 |
| SwiGLU 64×256（pilot） | 48.173 | 22.233 | 0.462 |
| Attention 1,4,2,16,32,16,16 | 35.189 | 14.872 | 0.424 |
| LayerNorm 4096×1024 | 359.189 | 389.362 | 1.122 |

Attention 维度顺序是 B,Hq,Hkv,Q,K,D,Dv；它没有实现 local redistribution，收益来自 whole-program 的 CPU 调度调整，不能说成 packet-local attention 加速。较大的 attention cohort 约 100 µs，未显示一致收益。

两条反例决定下一步工作：

1. **Ragged lowering 未进入成本模型。** RMSNorm 17×65 的 whole-program 是 direct CFG + full-packet specialization，1.299 µs；联合模型选择的 local 是 state machine，未触发 full-packet specialization，9.637 µs。257×1538 的 RMSNorm / LayerNorm / softmax 也会错选 local。`parallel` 的语义合法性没有问题，问题是实际实现成本与 Tile-level work prior 脱节。
2. **最少 callback 不等于最短并行时间。** 大 LayerNorm 的模型选择较粗的 G=4；静态同速 worker 的负载量看似理想，实际却比 G=0 慢。必须把 work stealing 的可用并行余量、任务尾延迟和 CPU 异质性放进目标函数或不确定性项；不能把每个 worker 静态分到一样多的 packet 当成性能证明。

## 已实现的代数与 policy 边界

```text
root programs × local distribution
                 ↓
          native packets / blocks        ← 决定机器码与逻辑坐标
                 ↓
       consecutive block ranges (G)      ← 独立的 CPU 调度决策
                 ↓
          caller 或 persistent pool
```

合法候选由 bridge 生成：root axis permutations × block widths × admitted local widths × optional task grains。task grain 搜索的是 powers of two 加旧 heuristic 和整个 launch；不是所有正整数。预算超限返回错误，不把搜索前缀伪装成 exact optimum。显式粒度优先于搜索，默认 G=0 保留 Runtime heuristic。

`ExecutionWork` 报告每 packet 的加权算术/访存先验、packet/block/task 数、active workers 和 static home-assignment critical counts。令 Q 为 packet 数，L 为 block 数，G 为每 task 的 blocks，C=ceil(L/G)，h=min(C,H)。h>1 时每个 home worker 领取 round-robin chunks；前 C−1 个 chunk 满载，最后一个可能短。对 capacity R 与总量 N，最长 home assignment 的精确计数为：

```text
F = C - 1
last = N - F * R
critical = max(ceil(F/h) * R, floor(F/h) * R + last)
```

对 packet 取 N=Q,R=G×B/W，对 block 取 N=L,R=G；task critical=ceil(C/h)。h=1 时 Runtime 直接调用一次整个 range，所以 critical packets=Q、blocks=L、tasks=1。公式只精确描述计数，不证明 heterogeneous wall time。

默认目标保留算术、访存、block work、静态失衡，并加入独立的 task callback 与 worker activation 项。所有系数、输出分项与总分检查 finite/nonnegative。

`ExecutionCostPolicy::coefficients()` 可以在抽取 work 前重写系数；`evaluate()` 返回完整目标，solver 不再二次除以 worker 数。后端可继承 `AnalyticExecutionCostPolicy` 重写任一 hook。借用期仅限同步 `plan()`；没有跨 DSO 所有权、TVM 类型、RTTI、算子名称规则或新的用户 DSL entity。policy 无权修改 hard constraints、数值语义与未实现 redistribution 的边界。

## 验证、可复现性与尚未覆盖的范围

最终源码完整构建、38 项 CTest 与 6 个修改后 C++ translation units 的语法检查均通过。新增 tests 枚举 W1/2/4/8/16、1/3/8 workers、非整除 dispatch 与极端粒度，逐任务模拟核对闭式公式；另检查 policy 覆写、非法/NaN/Inf 成本、预算超限，以及 Runtime 中固定代码、完整输出与边界哨兵。

测量后的代码审查还纠正了一个只影响单 CPU worker 成本记账的细节：Runtime 即使收到较细的 grain，也会在 caller 上执行一次整个 range。原 592 次对照全部请求 8 workers，此修正不改变其中的分配公式；仍分别保存测量时的完整源文件 archive 与最终 source patch，未把两者混作同一 binary。最终二进制另运行 pilot smoke，见 `provenance.json`。

复现：用完整构建后的 `benchmark_tile_xir` 运行 `measure.py --suite pilot|validation|fixed-grain --binary /absolute/binary --output /new/directory`，再用 `audit.py --raw /absolute/raw`复算三个目录。所有 diagnostic env 只存在 benchmark，生产配置通过 `PlannerOptions`；worker activation/task dispatch 的默认系数仍为零。

本轮没有新的 Torch/MPS/BLAS 对照，没有 Metal 改动，没有纯 native kernel 加速声明；也没有证明全部常见 LLM kernels 已达到目标。后续应把 direct-CFG/state-machine/full-packet 可实现性、原语级数学成本及并行任务尾部风险纳入后端服务模型，再用新形状和真实 native ABI 对照验证自动选择。
