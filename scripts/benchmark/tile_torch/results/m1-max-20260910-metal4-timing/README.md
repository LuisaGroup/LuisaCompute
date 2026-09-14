# Metal4 原生计时：实现与验证记录

## 结论

2026 年 9 月 10 日，Apple M1 Max / macOS 26.6.2。独立的 `Metal4TimingExt`
已在真实 `TileIR -> XIR -> LLVM/AIR -> Metal4 Runtime` 路径验证：可以分别保留
逐 dispatch 插桩区间、无 counter 的 command-buffer GPU 区间和主机提交/反馈阶段。
本机 counter 频率为 **24,000,000 Hz（24 MHz）**，不是 26 MHz。

首个 indirect dispatch 边界已修正：在 `99fad8c49` 基线上，完整配置重建后的
8 项选定 CTest 全部通过，耗时 306.58 s；专用计时测试实际执行 **6 个用例、
8,515 个断言**。修正前的 5 用例 / 8,279 断言日志独立保留。
随后 fast-forward 到 `811022021` 并完成隔离完整构建；首轮 20 项集成回归为
**19 PASS + 1 STL 测试入口 SegFault**。修正入口并完整构建后，三项 STL 补验
全部通过；现在所选 20 项均有通过记录，但**不是一次 20/20 全绿运行**。
18 项固定批量 pilot 已完成；扩展矩阵在用户中断时保留为 **35 OK + 2 Error +
107 NotRun / 144 项**，没有重新运行或用成功子集冒充完整结果。
**这是计时设施与正确性进展，不是性能目标完成：没有
Torch/MPS 对照，没有成本拟合，也没有零开销的“纯 kernel 时间”结论。**
本轮不更改 planner、默认等待策略或默认 feedback queue；排障时的阻塞等待、显式队列等
实验已撤回，保留的 [诊断 patch](validation/diagnostic-runtime-overlay.patch) 不是提交实现。

## 三种时间必须分开解释

公共接口是 [Metal4TimingExt](../../../../../include/luisa/backends/ext/metal4_timing_ext.h)，
实现位于 Metal4 backend，不借用 legacy Metal 的 `MTLCommandBuffer` 计时 helper。

- **Precise dispatch interval**：在每个实际 direct dispatch 前后写入 MTL4 precise
  counter，保留原始 tick、频率、shader checksum、dispatch/block 尺寸和所属 CB。
  `elapsed_ns = (end_ticks - begin_ticks) × 1e9 / frequency_hz`。
  counter 可能拆分 encoder、改变调度，因此它是带扰动的 dispatch 区间。
- **Feedback-only control**：不插入 counter 指令，保留 `MTL4::CommitFeedback` 的
  command-buffer GPU 起止时间及实际 dispatch 数。吞吐统计为所有非空 CB 的区间之和
  除以 dispatch 总数；它包含 CB 内工作及间隙，**不是单 kernel 计时**。
- **主机时间**：未开启采样的 host-wall batch/单次 E2E 另外测量；采样记录还保留
  commit 前后、feedback 入口、callbacks 完成、CPU completion 发布前的主机时间戳。
  这些主机戳来自同一 `steady_clock`；不能直接与 GPU/feedback 秒值相减。
  feedback 可能先于 `commit()` 返回，不强加两者的错误顺序。

`begin_sample()` 先排空边界前工作，`end_sample()` 先解除采样再排空；两次边界自身的
同步提交不计入样本。样本内部的同步空 CB 必须保留为 `dispatch_count=0`，其 GPU 区间
可能无效，但不能因此删掉记录或当作一次 kernel。不同 stream 独立；同一 stream 的
begin/dispatch/end 由调用方串行化。重复 begin、无 active sample 的 end、overflow、
缺失请求的 timestamp 均显式拒绝/报错，不静默生成成功数据。
overflow 保留部分原始记录，不截断实际 dispatch，也不能把该前缀当作完整样本。
后续审查曾发现 indirect range 恰好是样本首个 command 时，错误标记可能先于
command-buffer 采样状态建立而丢失。现在先建立采样状态，再标记 unsupported range；
新增第六个回归已验证 feedback-only 与 precise 两种模式均显式报错，且不丢失真实
indirect 工作。**它仍不支持 indirect per-dispatch 计时**，不能把 opaque range
计成一个已测 direct dispatch。

## 已验证版本的正确性覆盖有明确边界

[修正前 CTest XML](validation/test-final.xml) 保留完整输出，
[CTest 汇总](validation/test-final.log) 给出 8/8 与 145.56 s。
其中计时测试耗时 58.54 s；这不是性能测试结果。各组断言为：TileIR 340、
backend target-info 642、Metal4 timing 8,279、Metal4 rows 98,682、
SIMD Runtime 5,085,845、SIMD LLM 2,370,853；另有两个独立子进程验证非法 attention
输入的 fatal 拒绝，不能用 `expect(throws(...))` 替代。

修正前的专用测试使用
FP32 17×65 SwiGLU、32-lane local distribution、64 threads/group，实际执行五类检查：

1. Feedback-only 的单次/三次 dispatch 精确计数、前后采样边界、样本内空同步 CB。
2. 三个 precise dispatch 的非零且递增 ticks、24 MHz 换算、CB 归属和 host 阶段顺序。
3. 两条 stream 各自的 sample ID、两次/三次 dispatch 与独立输出。
4. 容量为 1、实际执行 3 次的 overflow：必须报错，保留计数和有效前缀。
5. 重复 begin、无 active end、非法容量，以及解除采样后正常 dispatch。

修正前计时验证及原 benchmark 所用改动保存在 [benchmark 源码覆盖包](source-overlay-benchmark.tar.gz)，
以 `99fad8c49` 为基线；该包只包含本轮相关覆盖文件，不是完整仓库快照。
[工作树中的测试](../../../../../src/tests/unit/runtime/test_metal4_timing.cpp) 已增加
首个 indirect command 的第六个用例；该用例的验证见下节，旧日志和覆盖包身份不变。

以上五个用例总计 7 次完整 readback，每次检查全部 1,105 个输出与两端各 17 个 guards：
7,735 次 FP64 oracle 元素检查、238 次 guard 检查。输出先填 NaN，容差为
`5e-5 + 5e-5 × abs(reference)`。本测试**没有**检查输入不变性，不能借用别的测试
替它声称该覆盖。缺 counter 能力的设备明确标记 precise correctness 未验证；
本机能力成立，实际 precise 用例也执行并通过，不只是 API 创建成功。

独立 Metal4 rows 测试仍覆盖 41 个 FP32 正例与 5 个预期拒绝，包含完整输出、输入不变性
和 guards；CPU 回归保持原 oracle。它们不扩展本计时测试的算子/精度范围，也不说明 W64
在 M1 Max 上执行过。完整构建日志和编译产物属于本轮版本，不能沿用
[前一 checkpoint](../m1-max-20260910-xir-target-info/README.md) 的二进制哈希。

## 首个 indirect command 修复通过，后续集成保留真实失败

修正后的 [CTest XML](validation/test-indirect.xml) 与
[汇总](validation/test-indirect.log) 记录 `99fad8c49` 基线上的 **8/8 PASS、306.58 s**。
专用计时测试为 **6 用例 / 8,515 断言、57.49 s**；这些时长仅是回归运行时间，
不是 kernel 性能结果。

第六个用例先在采样外准备 indirect range，然后让该 range 成为样本内的首个 command。
feedback-only 与 precise 各执行一次，分别完整检查 **65 个 uint32 输出和 34 个 guards**。
同时检查明确的 unsupported 错误、无虚构的 direct-dispatch 记录、保留的非 direct 工作
标记与有效 CB feedback。独立 SIMT 测试 kernel 的输出正确，证明“不支持计时”没有
抑制实际执行；这不是 Tile/SIMT 的 intra-kernel DSL 混合，也不扩展 FP64 oracle 范围。

随后仓库 fast-forward 到 `811022021`（项目自有 C++ 的无异常改动），保持原 benchmark
不重跑。TIRx 集成增加两个兼容修正：`codegen()` 用 `std::optional<Module>` 表达失败；
`map_execution()` 在诊断失败时返回已有输入 module，不构造默认 `IRModule`。
[新增 target 回归](../../../../../src/tests/unit/tile/bridge/test_tirx_targets.cpp) 分别移除
Metal strict fold 所需的 target/runtime precise-math contract，检查编译返回可恢复诊断、
没有有效 module，并在作用域结束时恢复原 registry；该测试不运行 GPU。

隔离完整构建后的 [首轮集成 XML](validation/test-integrated.xml) 与
[汇总](validation/test-integrated.log) 为 **19/20 PASS、379.26 s**：所选 TileIR、
XIR target-info、Metal4、SIMD 和九项 TIRx 测试均通过，计时测试再次执行
6 用例 / 8,515 断言（83.68 s）；此阶段 XIR Runtime 为 18 用例 / 5,085,897 断言，
TIRx targets 为 4 用例 / 3,273 断言，不沿用旧基线的断言数。
唯一失败为 `test_stl_containers` 的入口参数检查
与 Boost.UT 重载交互导致的 SegFault。原失败日志保留；入口已改为独立 argc guard，
[完整构建](validation/build-stl-fixed.log) 后的
[三项 STL 补验](validation/test-stl-fixed.log) 为 **3/3 PASS、0.70 s**，
[XML](validation/test-stl-fixed.xml) 确认正常 STL 实际执行 28 用例 / 118 断言，
另外两个子进程验证 mutable/const 的非法 `map::at` 拒绝。
这些分阶段结果覆盖了原来选定的 20 项，但不能合并成一次 20/20 PASS 的测试日志。
[集成验证 receipt](validation/integrated-receipt.json) 保存这一阶段正确性产物的 SHA-256，
与原 benchmark 的冻结 receipt 分开；兼容性修复提交为 `4cb312287`。

最终独立复核也发现计时测试入口有相同的 Boost.UT 短路隐患，已使用返回原生 bool
的字符串成员比较修正 argc 与可空环境变量检查。[完整构建](validation/build-timing-guards.log)
后，明确 unset `LUISA_TEST_REQUIRE_METAL4_TIMESTAMPS` 的
[默认环境回归](validation/test-timing-guards.xml) 为 **6 用例 / 8,514 断言、44.56 s**；
精确 counter 仍实际执行，少一个断言仅因不再强制要求 counter 能力。
无参数调用输出 usage 并返回 2，不崩溃。[最终入口 receipt](validation/timing-guards-receipt.json)
记录新的测试二进制及源码哈希；此改动不改变 backend 或 benchmark 二进制。

本节是不同源码阶段的正确性记录，不能改写原 pilot/扩展矩阵的源码覆盖包、产物 SHA
或未完成状态；`99fad8c49` 上的性能数据也不自动成为 `811022021` 的性能数据。

## 超时在旧路径也复现，不能归咎新计时设施

最初 default 和 blocking-wait 运行均在 180 s 超时，保留
[default](validation/timing-default.xml)、[blocking](validation/timing-blocking.xml)
和相应 [采样栈](validation/test-running.sample)。卡点是初始化 `reset -> synchronize`，
还未进入采样。单 stream 的 [提交 trace](validation/timing-trace.log) 中 commit 已返回，
但没有 feedback 入口；延迟第二条 stream 的创建没有把问题解释为双 stream 独有。

[未改 Metal4 基线](validation/head-backend-control.sample) 同样停在等待 completion，
[旧 Metal Buffer IO](validation/legacy-metal-buffer-io.sample) 也停在系统
`waitUntilCompleted`。保持机器 awake 后反馈恢复，随后完整测试通过。这支持系统/awake
状态参与的解释，但现有记录不构成具体电源状态、QoS、驱动或自旋机制的单变量因果证明。
不能宣称更改等待/queue 已修复根因，也不能拿被超时截断的样本拟合延迟。

特别排除 [最初 awake 过滤探测](validation/timing-awake-default.log)：它只完成
7 个 setup 断言，**5 个用例全部 SKIPPED**，不算计时验证。
[随后完整运行](validation/timing-awake-full-detail.log) 才是 5 用例 / 8,279 断言；
修正前最后一次重建后再以 `test-final.xml` 复验。仅凭 CTest 退出成功或被截断的
XML 摘要不够。

## Smoke 证实 GPU/主机时间分离，不给出排名

[RMSNorm smoke](smoke/results.json) 是 FP32 128×1024、local=32、实际 256 threads/group，
固定 8 次 dispatch、1 个样本、1 轮。吞吐阶段的三种观测分别为：

- precise：8 个实际 dispatch 区间的中位数 **16.583 µs**；
- feedback-only：非空 CB 区间之和 / 8，为 **25.063 µs/dispatch**；
- 未插桩 host-wall：**1.215747 s/dispatch**，即 1,215,747.34375 µs/dispatch。

这三项来自同次进程的不同采样阶段，不是同一个事件的可相减分解。
两次全量 FP64/guard 检查通过；[原始记录](smoke/0000-r0-rmsnorm-128x1024-lanes32.stdout.json)
保留全部 dispatch、空 CB 和 host 戳。运行期间存在并发 Doxygen 工作，且只有一个样本；
这个 smoke 只证明计时通路和巨大 host/GPU 分离，**不用于 cost、policy 或性能胜负判断**。

之前同 plan 的 SwiGLU 约 59× 差异还混入了不同 batch 分母：forced32 仅 1 次，auto
为 119 次，均是自适应 host-wall 校准后的数据。这个差异既不是 planner 的 59× 优化，
也不能只靠 batch 摊销认定全部解释完毕；旧记录不改写成 GPU 时间。

## 18 项固定批量 pilot 已完成，仍只作诊断

[串行 runner](../../metal4_timing.py) 为 RMSNorm、masked softmax、SwiGLU 各使用
FP32 128×1024，分别请求 local lanes `1 / 32 / 0`，两轮共 18 项。第二轮只反转每个
算子的 mapping 顺序（`0 / 32 / 1`），算子顺序不反转。`0` 表示自动选择，不是物理零
lanes；比较必须读取实际 realization。
[cohort/results.json](cohort/results.json) 已记录 **18/18 OK**、`finished_utc` 和
`artifacts_unchanged=true`，八项受检编译产物/runner 的开跑前后 SHA-256 一致。
这三个算子在本 cohort 的 auto 都实际选择 local=32；与 forced32 的 kernel mapping
相同，因此两者测得的时间差不能解释为自动 planner 生成了更好的映射。

每项 `samples=1`，host/device throughput 都固定 8 次，latency 为单次；目标窗口
20 ms、请求 warmup 10 ms、每进程 timeout 90 s。固定 repetitions 不再让一次长 host
等待把不同 variant 校准成 1 次和 119 次。每样本 precise 统计为 8 个区间的中位数；
control 为非空 CB 区间之和除以实际 dispatch 数，不把空同步 CB 加入该分母。
空 CB 原始记录仍保留。两轮观测不能给出稳定分布、置信区间或跨设备泛化结论。

```sh
# 先完成所选构建树的完整构建；结果目录必须尚不存在。
caffeinate -diu python3 scripts/benchmark/tile_torch/metal4_timing.py \
  --binary /absolute/build/bin/benchmark_tile_xir \
  --output /absolute/new-results-directory \
  --case rmsnorm:128,1024 --case masked_softmax:128,1024 --case swiglu:128,1024 \
  --local-lanes 1 32 0 --rounds 2 --samples 1 --repetitions 8 --timeout 90
```

Runner 检查计数、几何、checksum、tick 换算、空 CB 规则、每项两遍完整 oracle/guards、
跨 mapping/轮次输入哈希和采样前后产物哈希。JSON 的外部 `caffeinate` 状态只记录为
要求、未由 runner 自动验证；也不证明独占 GPU 或稳定频率。原始 tensor 导出保留在
JSON 指定的临时目录，仓库仅保留其 size/SHA-256 receipt，**不是自包含 tensor 归档**。
因样本极少且当前目的为归因，不绘制性能排行榜。

## 144 项扩展矩阵被中断，保留错误与未运行项

[rows/results.json](rows/results.json) 预注册六个算子：RMSNorm、LayerNorm、
masked softmax、SwiGLU、GELU residual、RoPE；每个算子使用四种尺寸、三个 local
lanes 请求（1 / 32 / auto）、两轮反向 mapping 顺序，共 144 项。尺寸为
17×65、129×768、257×1538、1024×4097；RoPE 对应偶数宽度 17×66 和
1024×4098。每项请求 3 个样本、8 次 throughput dispatch、120 s 进程超时。

用户中断后封存的清单是 **35 OK、2 Error、107 NotRun**。成功项包括第一轮的
RMSNorm 12 项、LayerNorm 11 项、masked softmax 11 项，以及 SwiGLU 1 项；GELU
residual、RoPE 和整个第二轮都未完成。清单没有 `finished_utc`、`artifacts_after`
或 `artifacts_unchanged`。恢复后另行核对当前产物哈希不能补造当时的正常结束记录，
也不能把 `NotRun` 当作性能或正确性失败。

两个 Error 都是 1024×4097、forced local=1，在 lowering 阶段超出 local snapshot
storage budget，分别保留 [LayerNorm 原始错误](rows/0021-r0-layernorm-1024x4097-lanes1.stdout.json)
和 [masked softmax 原始错误](rows/0033-r0-masked_softmax-1024x4097-lanes1.stdout.json)。
它们没有成功生成输出，不是数值比较失败；相同尺寸的 local=32 与 auto 均为 OK。
这是当前实现的候选存储预算问题，不能表述为 Metal 硬件最多允许多少 lanes。
后续应把存储/liveness 可行性前移到候选筛选：未固定 mapping 时寻找替代方案，
手动固定不可实现候选时保留明确诊断，不能只放宽预算、静默改写固定参数或删除失败行。
此次不重跑矩阵，也不对不平衡的成功子集做全算子排名或成本拟合。

## 独立审计确认已有记录，但不补齐未完成实验

[独立审计结果](validation/recovered_independent_audit.json) 与
[审计脚本](validation/recovered_independent_audit.py) 不调用 runner 的计时 validator，
而是从原始 ticks 和 CB 区间重算指标，并读取保留的全部输出及输入张量做 NumPy
FP64 oracle。两份清单中的 **53 个 OK 项**均通过：共检查 36,186,684 个输出元素、
2,214 个 dispatch 记录和 984 个 CB；张量 SHA-256、跨 mapping/轮次输入一致性及
计数/时间换算均无审计问题。pilot 与扩展矩阵的最大绝对误差分别约为
`5.09e-7` 和 `7.59e-7`。guards 未导出，因此独立审计不重复声明 guards 检查；
该覆盖仍以 C++ 运行时验证为准。

审计时，八项当前编译产物/runner 的哈希均匹配各清单的 `artifacts_before`。
它是恢复时的单独检查，不改变扩展矩阵缺失结束 receipt 的事实。所有这些记录仍为
**diagnostic**：没有 Torch/MPS 外部对照，mapping 相同的样本之间仍有明显波动，
precise counter 也有插桩扰动。数据完整性通过不等于成本模型可拟合，更不等于
planner 性能提升已获证明。

## 下一步：先可信归因，再泛化规划

1. 封存 `811022021` 集成及 STL 入口修正后的源码/产物身份与文档验证；独立保留
   首轮失败和补验结果，不覆盖旧 benchmark receipt 或把分阶段通过改成单次全绿。
2. 固定源码/产物和批量，记录 awake/电源及并发条件；利用同一 host clock 的提交、
   feedback、callback 和 completion 阶段排查长等待。必要时做单变量空同步、微小
   dispatch 和不同 batch 对照，不能从 GPU interval 短直接断言某个 host 根因。
3. 在环境稳定后扩大重复、尺寸和算子，分别保存 precise/control/E2E，并检查插桩
   扰动；用 held-out kernels 检验资源可行性与 cost policy，而不是给算子名加特例。
4. 将候选 storage/liveness、寄存器压力、occupancy、通信/同步等可实现约束纳入
   execution mapping 搜索，再做可追溯校准。当前 uncalibrated GPU prior 未因此升级。
5. 最后才进行匹配精度、math policy、分配与计时边界的 Torch/MPS 对照；本记录不声称
   打败任何库，也不把 instrumentation 的实现冒充 kernel 优化。
