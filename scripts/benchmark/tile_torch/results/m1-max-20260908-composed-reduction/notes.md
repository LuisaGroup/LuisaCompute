# Composed reduction 与 contraction 分布：受控实验记录

2026-09-08，Apple M1 Max / macOS 26.6.2，Torch 2.14.0。结论：**给已有 softmax
阶段加入 collective，只带来小幅收益；QK 的贡献维分布是更重要的候选，但收益与执行
宽度强耦合。尚未成为通用 planner 改进，更没有达到 Torch/MPS 性能目标。**

## 1. 实验边界与计时口径

生产编译器固定为 `f5daf25e6` 的独立源码快照，沿用
[归约策略 checkpoint](../m1-max-20260908-reduction-policy/notes.md) 的构建与依赖。
未混入工作区的 matrix initializer/view 实验。只修改 benchmark 控制及 fixture 的可选
QK 分解；没有修改生产 cost model、lowering 或 fast-math。二进制与五个 TVM 动态库的
SHA256 记录在每个 `results.json`，每次实验均完成整树构建并检查 artifact 未变化。
初始 pilot、输入-view 控制、QK probe 是三个不同 benchmark/protocol 版本，不能混为
同一二进制实验；每组内部的二进制和四份 protocol/fixture 指纹一致。

三种 decode shape 均为 FP32、bottom-right causal mask、GQA，固定 block 为 `1×32`：

| 名称 | B,Hq,Hkv,Q,K,D,Dv | 目的 |
|---|---|---|
| S1 | 1,8,2,1,2048,64,64 | 标准 decode |
| S2 | 1,8,2,1,2053,80,96 | K 尾块、不同 QK/PV 宽度 |
| S3 | 1,16,4,1,4096,128,128 | 更大 context/head/channel |

后两组每个配置 4 rounds、每 round 7 samples、30 ms sample window、100 ms warmup；
每组内部 native/Torch 顺序交替。**不同配置没有交错执行**：下面的开关/分解差值只是
描述性对比，不是 paired A/B、置信区间或经过 held-out 验证的决策规则。

- 主表是 **no-counter command-buffer GPU interval / invocation**，包含 buffer 内
  执行及空隙；不含 host 编码，但不等于纯 kernel 时间。
- 单次 compute-pass 时间另存；native 每次调用对应一个 kernel、一个 compute pass，
  probe throughput 的 pass 数也等于 repetitions。这是带插桩的区间，不是硬件
  per-dispatch counter。Torch 的 SDPA 可能有多个 pass，不能直接称为同一 kernel 的对照。
- E2E batch 和单 dispatch latency 同时保留。Torch 使用 functional SDPA，计时包含其
  输出及内部中间分配，mask 构造在计时外；不是 compiled Torch/MPSGraph 基线。
- native 使用 `fast_math=false`、source `unordered_tree`；Torch 使用默认数学策略。

## 2. 初始 pilot：collective 开关并没有隔离 collective

`compiler.cpp` 中当前 view 请求为
`forward_readonly_tile_loads || subgroup_reductions`。因此旧开关会同时尝试将 Q/K/V
snapshot 改为经过验证的只读 view。自动宽度也由 1024 变为 64。

| 配置 | 实际线程数 | S1 GPU µs |
|---|---:|---:|
| off / auto | 1024 | 375.473 |
| on / auto | 64 | 423.201 |
| off / 64 | 64 | 494.627 |
| on / 32 | 32 | 529.060 |
| on / 64 | 64 | 425.968 |
| on / 128 | 128 | 426.995 |
| on / 256 | 256 | 425.207 |

这 7 个目录（`off-0` 等）保留全部 28 行记录；每配置 2 rounds、5 samples、20 ms window。
不能从其中选择一个较大的比值宣称是 warp intrinsic 的加速。

## 3. 固定 views 与执行宽度后，closed collective 收益有限

新增 `--forward-input-views`，对两侧都显式请求同样的只读 view 和同样的线程数。
`audit.py` 检查每个 shape 的完整 Metal source：除了两个 barrier 分隔的 sum/max
归约阶段外，其余代码逐字相同；没有换掉 QK/PV、input staging、资源声明或数学策略。

| 线程数 / shape | collective off GPU µs | collective on GPU µs | on / Torch，同 round 比值的中位数 |
|---|---:|---:|---:|
| 64 / S1 | 444.315 | 421.277 | 11.110× |
| 64 / S2 | 1297.879 | 1265.897 | 9.296× |
| 64 / S3 | 1328.437 | 1281.082 | 16.819× |
| 1024 / S1 | 538.915 | 492.850 | 13.503× |
| 1024 / S2 | 1004.094 | 958.355 | 6.947× |
| 1024 / S3 | 1418.442 | 1328.624 | 17.338× |

六个描述性差值约 2.5–8.5%。相对于仍有约 7–17× 的 Torch 差距，不是结构性解决方案。
此组 96 行全部通过完整 FP64 输出检查；native 还在计时前后检查全部输出及两端 guard。

## 4. 用已有 DSL 做 QK 分解 probe：不是新 primitive

可选 `--attention-qk reduce` 仅把 fixture 的 QK 改写为：

```cpp
auto dot = reduce(query * key, d, add);
// PV 仍为 mma(probability, value, acc * alpha)
```

默认仍使用 `mma`。C++ 参数只是测试 helper 的 host 选择，不进入用户 DSL 或生产 planner。
实际 source 验证 multiplication 没有构造巨大的中间 product tile，而是作为 reduce 的
贡献直接读取 Q/K；QK 多一个 `simd_sum`，PV 仍为输出元素并行、K 串行的 contraction。

| shape | QK reduce / 64 GPU µs | QK reduce / 1024 GPU µs | 1024 / Torch 配对比值 |
|---|---:|---:|---:|
| S1 | 718.674 | 363.985 | 9.245× |
| S2 | 1535.067* | 519.987 | 3.798× |
| S3 | 1924.518 | 730.416 | 9.640× |

`*` S2 / 64 的四个 native round 全通过，但第三号 round 的 Torch 计数器检查报
`Metal compute-pass duration exceeds enclosing command-buffer time`，整行判为失败，
因此不发布该 case 的完整 native/Torch 比值。没有放宽计时校验或覆盖失败记录。
该组 48 行保留 47 个有效结果与 1 个失败；24 个 native 结果均通过独立 FP64 与 guard。

1024 线程相对同宽度的原 QK 写法，描述性下降约 26% / 46% / 45%；但 64 线程全部变慢。
因此不能据此把所有 `mma` 机械地改成 subgroup reduction。

| QK reduce / 1024 | 无插桩 GPU 单次 µs | 插桩 compute-pass 单次 µs | E2E 单 dispatch µs |
|---|---:|---:|---:|
| S1 | 371.771 | 371.896 | 629.167 |
| S2 | 526.896 | 527.104 | 783.792 |
| S3 | 750.542 | 734.937 | 1014.750 |

这三种口径不可互相替代；上方吞吐表也不是这张单次延迟表的同义词。

## 5. 对 execution calculus / planner 的具体启发

下面是**源码观察与待验证的成本解释**，不是硬件 bank/cache counters 的测量：

```text
                         输出维 spatial             贡献维 spatial
QK: K[key, channel]       lane → key，channel 串行   lane → channel，key 分批
PV: V[key, value_dim]     lane → value_dim，key 串行 lane → key，value_dim 分批
                         │                         │
                         └── 每个 phase 分别选 ─────┘
                                  │
                     跨 phase 的重分布、共享状态与同步
                                  │
                     全 group 的线程/寄存器/shared 预算
```

1. **候选空间缺失比系数不准更先要解决。** decode 的 Q=1 无法进入当前要求 M/N/K 为
   8 的倍数的 matrix atom matcher；剩下的 `mma` 是输出并行、贡献串行。workload
   collector 只以输出 domain × executions 计入 independent work，漏掉串行 K 工作。
   matrices 为空时，planner 返回参考宽度，不进行完整的混合阶段成本搜索。
2. **执行分布须包含贡献维因子。** 每输出 lane 数 `p`、每线程输出数、输出 domain
   的分批数应成为候选，而不是只有整 group 的 T。当前 QK probe 固定 `p=32`：T=64
   时 32 个输出要做 16 批 collective，T=1024 时一批完成。若只数总 FLOPs 或内存
   连续性，无法解释这种反转。`p=1/2/4/8/16/32`、向量读取和局部寄存器累积值得比较；
   subwarp emitter 尚未实现，不能把这份清单当作现成能力。
3. **QK 和 PV 不应强制共享一个局部 layout。** QK 的 channel 连续、PV 的 value_dim
   连续；相同的代数 contraction 不同 access composition 会有不同的合适分布。统一的
   是候选描述和成本接口，不是统一的物理 lane 方向。
4. **成本应组合 phase 服务时间与转换成本。** 候选可用
   `(output_partition, reduction_partition, vector_width, resource_choice)` 描述，比较
   `sum(phase_service) + sum(redistribution/synchronization)`，并满足峰值 live state、
   同步作用域及资源容量约束；若引入重叠执行，改用依赖图的资源受限 makespan。
   不能默认把不同 pipeline phase 视作无代价并发。
5. **数值契约与映射合法性分开。** FP32 reduction 重排依赖 `unordered_tree`；不能
   仅因 load contiguous 或调用了 `mma` 就绕过明确的严格策略。现有只读 view、效果、
   归约 matcher 应成为候选的前置条件，失败时保留合法参考实现。

下一步应先对 typed contraction 实现通用贡献维候选，做同二进制交错 A/B；再把
contraction、closed reduction、标量阶段与转换流量联合计入 backend cost policy。
也需要 standalone dot/GEMV、小矩阵、不同内存排列及 CNN/filter 等非 attention 反例，
避免以这三个 decode shape 的结果校准出算子名规则。split-K / 在线状态 merge 仍是
另一层候选，本次没有实现或测量。

## 6. 验证、复算与复现

[audit.py](audit.py) 不导入 benchmark 的汇总函数，独立从 raw ns、repetitions 和每轮
samples 复算六种时间口径、四种 published 汇总及 native/Torch 配对比值；同时检查
完整 shape/round/path 粒度、顺序、原始日志、输入/source 指纹、native guard receipt。
删除行、伪造比例、错误 GPU 除数、错误 view 请求、部分输出、篡改 source hash 六类
对抗输入均须被拒绝。结果为 [audit.json](audit.json)：**172 行，171 valid，1 retained
failure；可带 caveat 分享，不支持全局性能或因果加速结论。** 临时 tensor 已由 driver
删除；保存的 receipt/hash 不是事后重新执行 FP64 检查的替代品。

PoC 和控制元数据修改后，独立整树构建、全部 **35/35 Tile CTests**、**111/111 Python
benchmark tests** 通过；clangd 对实际构建快照中的 benchmark TU 无诊断。新增 CPU
检查覆盖 QK 表达式分解的 prefill/decode/GQA 及 ragged contribution/output；原有用例
没有删除。完整日志保存在本目录 `validation/`。

```sh
uv run --offline --no-project --python 3.13 \
  python scripts/benchmark/tile_torch/results/m1-max-20260908-composed-reduction/audit.py
```

复现实验用 `compare_llm.py`，指定上述三个 `--case`、`--attention-block 1 32`、
`--rounds 4 --samples 7 --sample-ms 30 --warmup-ms 100`，分别选择
`--group-threads 64|1024 --forward-input-views`、有/无 `--subgroup-reductions`；QK probe
额外加 `--attention-qk reduce`。native 为 `benchmark_tile_tirx`，使用独立 checkpoint
build，显式记录 Metal timing helper 与全部五个 TVM compiler/runtime dylib。
每次指定全新的输出目录，不覆盖已有记录。完整 native invocation 保存在每行 `command`，
所有要求的开关保存在 metadata；原始 generated Metal 与 native log 一并保留。
