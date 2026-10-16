# 2026-09-10：禁用 full-packet specialization 的反证实验

## 结论

**不采用禁用 specialization 作为小算子修复。**这六组实际生成物没有消除
helper 调用，反而让完整 packet 进入仍带动态 lane mask 的通用 helper。
减少整个对象的指令字节数，并不意味着减少热路径工作。

本实验保持 full-packet specialization **关闭**，在同一 cohort 内比较
pointwise fusion off/on 与冻结 Inductor。它不是 specialization 开关的配对
A/B：不能用此前另一 cohort 的绝对时间相除，声称测得关闭 specialization
的因果减速比。此前完整实验见
[pointwise-v2](../m1-max-20260910-native-pointwise-v2/notes.md)。

## 实际测量

Apple M1 Max，FP32，单 CPU worker，W8/local=8/block=32，fast math 关闭。
除新增 `LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION=1` 外，复用此前固定
环境；每个 case 的 off/on 唯一差异仍是 pointwise fusion。全部 12 个 Tile
capture 的 specialization/clone 两项计数都明确为零，其余 mapping/math/
fusion/alias 检查保持不变。使用独立 adapter，不修改此前冻结的 runner。

共同 C++ timer 调用实际 ORC/Inductor native entry，排除 Runtime、Python、
JIT 和调用方分配；必要 traversal/reset 和生成代码内的 libc/分配保留。
每 case 三种 variant 的全部六种访问顺序，每 visit 七样本、100 ms warmup、
30 ms target。时间列为六个 visit 中位数的中位数（µs）；比值列为六个同轮
比值的中位数，方括号为范围，不是置信区间，也不是显示时间相除。

| Case | Off µs | On µs | Inductor µs | On/off [range] | On/Inductor [range] |
|---|---:|---:|---:|---:|---:|
| RoPE 17×66 | 1.234 | 1.265 | 0.113 | 1.028 [1.011–1.030] | 11.249 [11.199–11.533] |
| RoPE 1024×4098 | 3526.180 | 3959.862 | 970.197 | 1.123 [1.118–1.146] | 4.089 [3.970–4.251] |
| LayerNorm 17×65 | 2.453 | 2.216 | 0.564 | 0.902 [0.871–0.912] | 3.924 [3.888–3.967] |
| LayerNorm 1024×4097 | 5838.031 | 5336.964 | 4907.180 | 0.914 [0.912–0.918] | 1.088 [1.086–1.090] |
| SwiGLU 17×65 | 2.779 | 2.502 | 1.293 | 0.897 [0.893–0.901] | 1.934 [1.912–1.965] |
| SwiGLU 1024×4097 | 9013.422 | 8175.583 | 4925.292 | 0.905 [0.880–0.917] | 1.660 [1.641–1.675] |

RoPE 两个 case 的 On/off 都输 6/6 轮；LayerNorm、SwiGLU 都胜 6/6 轮。
但全部六个 case 的 On/Inductor 都输 6/6 轮。保留所有结果，不用较好的
LayerNorm fusion 比值掩盖整个 realization 仍慢的事实。

## 正确性与证据边界

108 visits 全输出、输入不变和 guards 检查通过，六个 off/on 输出均逐位一致。
独立审计在原目录重读 18 个输出、复算 FP64 oracle 与统计，检查 373 个身份
条目，并拒绝 11 类证据篡改。数值容差沿用 `atol=rtol=5e-5`；数据为有限的
确定性 fixture，不是 NaN/任意输入分布的完整精度证明。

计时前后一分钟 load average 为 3.223 / 3.392；记录背景负载并不证明机器
安静。此任务未在正式 replay 期间运行构建、测试或另一 benchmark。
原始目录为 `/tmp/luisa-pointwise-no-specialization.whVpcS`。持久归档排除
张量 payload；原始 JSON 保留其路径和哈希，不能把审计记录当作归档内仍有
那些数值。Guards 在执行时检查，释放后的存储没有事后重读。

## 生成代码揭示的问题

十二个对象均保留通用 `_llm_rows` helper；外层 wrapper 为 540 B 指令、
112 B 栈帧，含 12 个静态调用点（并非每次调用都执行全部 12 个点）。
完整 packet 仍传 `active_lanes=8`，但跨函数边界无法将 helper 内部 mask
常量化。小 RoPE on 的 helper 起点 `0x00–0x28` 生成动态 mask，
`0x190–0x208` 已有逐 lane 分支和 scalar loads；对应的特化 helper 使用
向量读取。该观察来自实际对象反汇编，不是仅看高层 IR。

原 specialization 路径的小 RoPE / LayerNorm off 被完全 inline，on 保留
compact full-packet helper。因此此前回退不是“多一个函数”这样独立的变量：
mask 常量化、cold tail 的 inline 决策、alias guards 和栈帧都会相互影响。

后续已经完成 [outlined-tail 单变量实验](../m1-max-20260910-outlined-tail/notes.md)：
保留 full-packet specialization，仅将真正 narrow tail 的调用标成 NoInline，
完整 packet 调用不变。它减小了 wrapper frame，但没有带来普遍收益，小 RoPE
还稳定回退，因此候选已撤回。之后可以单独检验依据 uniformity 提升
batch-invariant alias guards；不能靠假设用户 buffers 不 alias 来删除保护。

这是一条通用 codegen/cost-model 启发，尚未证明新的默认策略或 Torch/MPS
性能目标已经达到。
