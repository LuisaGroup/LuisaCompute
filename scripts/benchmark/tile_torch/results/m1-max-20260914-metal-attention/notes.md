# Metal attention：数值恢复与计时诊断，不是新的性能达标结果

2026-09-14，Apple M1 Max。当前应把三件事分开：GPU 能否完成工作、输出是否正确、计时是否足以比较性能。**没有 GPU Hang 不等于测量稳定；数值通过也不等于已经超过 Torch/MPS。**

本轮保留 MPS health、两个 Metal4 tiny capture、TIRx/Torch-MPS 小对照及固定批次 Metal4 pilot。原始 JSON 中的 `OK` / `cohort_valid` 表示执行与协议检查，不替代本报告对性能验收的判断；不修改原记录去掩盖差异。

| 路径 / 探针 | 已有证据 | 可以得出的结论 |
|---|---|---|
| Torch/MPS health | 27 个阶段没有捕获到 GPU Hang，但观察到延迟波动 | 本轮能完成这些小工作；不是全设备健康或稳定性能保证 |
| XIR → Metal4，tiny GELU 与 attention | 两次完整导出 oracle、precise / feedback control 检查通过 | 数值与计时采集协议可工作；没有可接受的跨框架速度结论 |
| TIRx → Metal 与 Torch/MPS，小 attention | 四次 visits 数值通过；同 round GPU 比值由 3.3912 变到 0.2045，E2E 剧烈漂移 | **拒绝性能验收**，不得选较快 round 宣称胜出 |
| XIR → Metal4，固定批次 pilot | 2 cases × 2 rounds 全部数值/协议通过，repetitions=8，samples=3，artifacts 未变 | attention 的 instrumented dispatch 为约 1.9–2.0 ms；不是所有慢都能归因于 host 等待，仍不构成跨路径排名 |
| XIR → Metal4，BQ4/BQ1 两个 ABBA cycles | 两个 shape、16 次 visits 完整数值/协议通过，输入与产物身份一致 | small 的 precise/control throughput 都改善；prefill 两种 GPU 口径方向不一致，不能宣布普遍改善 |
| native MPP | 当前没有整个 attention 的原生 MPP lowering 成绩 | 不把 GEMM/子步骤的 MPP 结果外推为完整 attention |

## 这次 TIRx / MPS 比较为什么不能用于性能验收

尺寸 `[B,Hq,Hkv,Q,K,D,Dv] = [1,4,2,16,33,32,32]`。下表保留原始每 visit 中位数，单位 µs。GPU 列是无 encoder instrumentation 的 command-buffer interval / invocation，不是隔离的零开销 kernel 时间。

| Round / 执行顺序 | 路径 | Host throughput µs | Host latency µs | GPU control throughput µs |
|---|---|---:|---:|---:|
| 0 / 1 | TIRx Metal | 8032.084 | 8013.541 | 67.2501 |
| 0 / 2 | Torch/MPS | 61.202 | 365.958 | 19.8305 |
| 1 / 1 | Torch/MPS | 25023.583 | 3859.792 | 101.1250 |
| 1 / 2 | TIRx Metal | 21.756 | 338.041 | 20.6803 |

GPU Tile/Torch 的同 round 比值是 3.3912 和 0.2045；Host throughput 比值则是 131.2390 和 0.0008694。符号翻转与量级漂移足以拒绝这次性能比较；不把两个 round 的中位数合成“接近 MPS”或“超过 MPS”的成绩。只保留它作为数值成功、采集环境/协议仍需稳定的诊断证据。没有据此确定是硬件、系统活动、同步、驱动还是 adaptive batch 造成波动，也没有修改生产 planner。

## 历史成绩单独保留

历史 TIRx→MSL→Metal 的 cooperative prefill 比 Torch SDPA GPU control 时间慢 **2.894×**；三组 decode 是 **3.798×–9.640×**。这些是不同 revision、尺寸与独立 cohort 的历史证据，不是本次刷新成绩。详细值和计时边界见 [attention mapping review §3](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md#3-量化-prior历史证据不是当前-revision-的成绩)。

XIR → Metal4 目前的 pipeline/MMA 路径仍只接受 local distribution 1；它支持独立 program 的执行，并不意味着已经把单个 MMA 映射成成熟的 GPU 协作矩阵实现。原生 MPP 的完整 attention、稳定的多尺寸对比与性能目标仍未闭合。

## 计时工具的修补

`metal4_timing.py` 现在接受七维 attention 与可选 `--attention-block BQ BK`，row 默认 case 和 `1×1` block 不变。验证四个真实 tensor 的形状与完整导出内容，而不是把七个问题维度相乘当成输出元素数；复用已有 bottom-right causal GQA / FP64 oracle。

实际 block、QK/PV 模式、唯一 `local_lanes` 字段、显式 lane 请求和每条 timestamp 的 dispatch size 都必须与实际 metadata 一致。自动 lanes 请求 0 允许后端选择合法正值，没有硬编码 32 上限。GPU Hang/错误日志或超时后不再启动剩余 visits，并将整个 cohort 判为无效，已完成的证据仍保留。13 个新工具测试与 compare_llm 合计 37 个纯 Python tests 通过；两个实际 Metal4 输出也经更新后的完整验证通过。

三种时间始终分别呈现：instrumented dispatch interval、无 encoder instrumentation 的 feedback-only command-buffer interval / dispatch、同步 host-wall / dispatch。批次使用自己的实际 dispatch 分母；有 counter 开销的区间不能改名为零开销“纯 kernel 时间”。

## 固定批次 pilot：四次 visits 数值与协议通过

Case 为 `gelu_residual:2,17` 与 `attention:1,4,2,16,33,32,32`，attention block `4×16`。两轮、每批八次、三个样本，requested/actual local lanes 均为 1。四次 visits 的完整输出、协议与 artifact identity 均通过。下表单位统一为 µs，各列是该 visit 三个样本的中位数；throughput 每样本包含八次 dispatch，latency 每样本一次。

| Case | Round | Throughput precise | Throughput control | Throughput host | Latency precise | Latency control | Latency host |
|---|---:|---:|---:|---:|---:|---:|---:|
| GELU+residual 2×17 | 0 | 19.438 | 98.844 | 150.625 | 143.208 | 74.250 | 1013.500 |
| attention | 0 | 1980.438 | 2056.062 | 2184.370 | 2278.667 | 2353.375 | 3336.084 |
| GELU+residual 2×17 | 1 | 40.292 | 27.641 | 191.453 | 24.583 | 23.500 | 4947.000 |
| attention | 1 | 1883.021 | 2873.891 | 3080.307 | 2524.917 | 2360.625 | 6759.500 |

Attention 本身 instrumented dispatch 已是毫秒级，说明不能把全部差距都解释成 Runtime 等待；但 precise/control 是独立采集阶段且有不同 instrumentation，不能简单相减得到精确 host 或 driver 成本。GELU 的 host 与 latency 仍有明显波动。只有两个 rounds，这也不是“大矩阵覆盖”；不把这批与之前 TIRx/MPS cohort 拼接，亦不宣称当前 GPU 已稳定到可以验收全部性能。

完整独立输出误差：GELU 34 个元素最大绝对误差 `2.2304e-7`；attention 2,048 个元素 `1.9352e-7`。另 tiny attention 是 `[1,2,2,5,5,5,3]`、block `2×3`，30 个输出元素最大误差 `8.4636e-8`，不与较大的 pilot 混为同一性能 case。

## BQ4 / BQ1：执行结构和资源必须一起看

两个 shape 分别是上述 small `[1,4,2,16,33,32,32]` 和预声明的较大 prefill 点 `[1,4,2,64,128,64,64]`。每个 shape 按 `4,1,1,4,4,1,1,4` 执行两个 ABBA cycles；BK16、local lanes 1、FP32、QK/PV MMA、repetitions8、每 visit 三样本固定。下表比值为 **BQ1 / BQ4，低于 1 表示 BQ1 耗时更短**；使用相邻的 `(1/0,2/3,5/4,6/7)` 四对，不用所有原始样本假装四倍独立样本量。范围只是四对的 min–max，不是置信区间。

| Shape | 口径 | 四对中位数 | min–max |
|---|---|---:|---:|
| small | Throughput precise | 0.4254 | 0.4151–0.4344 |
| small | Throughput control | 0.6467 | 0.4345–0.8598 |
| small | Throughput host | 0.5693 | 0.5027–1.3550 |
| small | Latency precise | 0.4586 | 0.3178–0.6196 |
| small | Latency control | 0.5473 | 0.5076–0.6180 |
| small | Latency host | **1.6368** | 0.3887–2.9945 |
| prefill | Throughput precise | **1.1385** | 0.7152–1.3939 |
| prefill | Throughput control | 0.3872 | 0.3178–0.4218 |
| prefill | Throughput host | 0.3974 | 0.3775–0.4070 |
| prefill | Latency precise | 0.5764 | 0.2023–1.4585 |
| prefill | Latency control | 0.6902 | 0.2676–1.2820 |
| prefill | Latency host | 0.6802 | 0.4585–1.0810 |

Small 的两类 GPU throughput 在四对中都改善，但 host latency 中位比反而退步；prefill 的 control/host throughput 改善，而 precise throughput 中位数退步且范围跨 1。保留全部负结果，不能只选最有利的计时口径。这是有用的映射诊断，不足以验收通用 BQ1 默认策略，更没有同时重测 TIRx/MPS。

| Shape | BQ | 实际 dispatch / block（x） | Threadgroups | 每 worker snapshot B | Snapshot allocations |
|---|---:|---:|---:|---:|---:|
| small | 4 | 16 / 32 | 1 | 6688 | 10 |
| small | 1 | 64 / 64 | 1 | 4224 | 4 |
| prefill | 4 | 64 / 64 | 1 | 12832 | 10 |
| prefill | 1 | 256 / 256 | 1 | 8320 | 4 |

BQ 还改变 program 数、自动 group 宽度与每 worker 资源，**不是固定硬件布局下孤立改变 BQ**。上表的 y/z 均为 1；两种 BQ 在两个 shape 上都只派发一个 threadgroup，BQ1 增加的是同组内独立 programs，并没有增加跨组并行。local lanes 始终为 1，不代表 group width 固定。不能仅凭此试验把全部差距归因于 reduce，也不能将小 BQ 的收益当作已经校准的 planner 决策。

静态实现审查还发现当前 MetalTileCostPolicy 主要计入总 packet work 与 block dispatch，未建立 residency 模型，可能偏向只合并 group 的选择；这是模型结构上的判断，不是采样证明某类硬件热点占比。下一步需要结合普通线程内 reduction carry loop、MMA 与临时存储的实际 lowering 判断，不能把已有 `WARP_READ_LANE` 支持误报成整段 attention 已获得协作规约映射。

独立只读 [审计（含实际 geometry）](audit-2.json) 重读 4 次 fixed + 16 次 query visits：80 个 tensor receipts、完整 FP64 oracle、全部 timing records/分母、actual lanes、dispatch identity、输入与 capture artifact receipts 均通过。每个 query shape 的八次输出在本次输入上逐 bit 相同；small 最大绝对误差 `1.9352e-7`，prefill 为 `2.1782e-7`。这不证明所有输入上的 bitwise 保序等价。所有四个配对比值和全部六类样本均在审计 JSON 中保留；初次未列 geometry 的 [审计](audit.json) 原样保留。

## 证据与归档边界

原始目录为 `/tmp/luisa-metal-attention-recovery.5MPxs3`。归档保留本轮原始文件（排除 Python virtualenv 与 `__pycache__`），并只从 fixed 与 16 个 query cohorts 明确记录的 17 个临时 tensor roots 收取其 `tensor_receipts` 引用的 80 个文件，不扫描其他 `/tmp` 数据。每个成员记录 byte size / SHA-256，压缩完成后完整读回核对，原 raw 不改。

`native-provenance.json` 在 native 探针前记录选定源码与二进制指纹。包内只收录该 manifest 中 selected-source 路径下、逐项 hash 相符的源码；二进制/dylib 只保留指纹，不复制整个动态依赖树。本轮 Python helper 也收录并与 fixed cohort 的实际 artifact receipts 核对。测试 helper 仅作为最终验证源码快照，不追认成 capture 的生产依赖。

这不是全 HEAD、自包含重建包或完整 loader-closure 证明。受保护 TIRx WIP 不补写成 native 前冻结源码；该路径的二进制指纹与实际生成 `.metal` 保留为有限证据。任何补录只能明确标为 post-run inspection snapshot。

归档已完成：`evidence.tar.xz` 为 672,040 B，503 个成员，包括156个冻结 selected-source 文件与80个外部 tensor；完整读回哈希验证通过，原始 raw 未改变。详见 [成员清单](package-inventory.json)。另在全新临时目录解包，使用包内 helper/tensor 完成 [portable audit](portable-audit.json)，20 visits / 80 tensor receipts 再次通过，不读取原机器的 tensor 临时目录，也不执行 GPU 或编译器。

`audit.py` 仅做离线数值/计时/receipt 检查。原始 [audit.json](audit.json) 保留；[audit-2.json](audit-2.json) 补充实际 launch geometry，portable audit 使用这一版脚本。37项测试的独立执行记录保存在归档 `raw/python-protocol-tests/` 与主线程复核 `raw/python-unit-final/`；它们是两次测试执行，不是74个不同测试。本轮新增的是计时/验证设施及诊断报告，没有修改生产 Metal planner 或 reduction lowering。
