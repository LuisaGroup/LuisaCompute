# XIR 候选静态资源准入：2026-09-10 checkpoint

本轮把 snapshot 预算从“选定方案之后的 lowering 拒绝”前移到公共 solver 的
候选筛选。**这是通用可行性修复，不是一次 kernel 加速测量。** 没有更改成本权重、
默认局部分布或 fusion 开关，没有重新测量 Torch/MPS/BLAS。

实现与两轮主回归基线为 `next` 的 `01e482839` 加本 checkpoint 的改动。收尾时又
安全快进到 `6291071ee`，其八个 DX/Vulkan TLAS/测试文件不涉及本轮 Tile 实现，
同步到隔离源后再次完成全构建；本机没有执行 DX/Vulkan 测试。隔离源码在
`/tmp/luisa-next-integration.roZbN8/source`，构建在同级 `build`；使用 LLVM 22.1.8、
RelWithDebInfo、SYSTEM_STL、SIMD/Metal/Metal4/TIRx/DSL，关闭 GUI/CUDA/fallback。
TIRx 与 XIR 的同进程测试比较仍关闭，避免既有 LLVM21/22 共存问题；独立 TIRx
targets 仍参与全构建。原工作区 TIRx/matrix/iOS/skill/子模块 WIP 没有纳入本轮。

## 实现范围

```text
固定 root 约束校验
       ↓
后端几何准入 → 共享静态资源分析 → 后端预算
                                  ├─ 超限：记录候选、需求、预算、原因
                                  └─ 合法：work extraction → schedule/cost
                                                            ↓
                                                   选择 → lowering → native 检查
```

- `analyze_resources()` 与 lowering 共用 value/carry 分配和静态 body 展开规则，
  不先生成一份试验 XIR。统计为 64 位的 `snapshot_bytes_per_worker` 和
  `snapshot_allocations`；分析报告需求，不消耗传入预算。
- `ExecutionTargetInfo::resource_limits(candidate)` 可按后端及具体候选返回预算。
  SIMD 使用实际物理 packet W 的 `16 MiB / W`，Metal4 使用既有 64 KiB compiler
  snapshot 上限。公共 solver 没有重新引入 CPU/GPU lane 常数。
- 超预算候选不进入 `schedule/evaluate`。自动搜索继续；固定约束不会被偷偷放宽。
  `PlanningResult.rejected` 保留诊断；全部候选失败也保留，不回退成空记录。
- root 分块/排列验证由 planner、resource analysis、lowering 共用；非法约束返回
  错误，不在 mixed-radix helper 中终止进程。
- lowering 独立检查预算，实际发射分配与分析值核对；backend 继续检查最终
  workspace/PSO/ABI。编译器静态字节数不是 native 对齐后工作区的替代品。

精确定义和公式见 [XIR 内部文档](../../../../../docs/source/internals/tile/xir.md#static-snapshot-admission-precedes-cost-ranking)。
这里统计静态发射的 allocation sites，不是动态循环迭代数、寄存器数、峰值 liveness、
stack frame 或 occupancy。Pointwise fusion 的 alias fallback 仍生成快照，因此
disjoint 快路径的流量下降不要求总 snapshot 数变为零。

## 验证记录

完整配置构建通过。首轮 8/8 选定 CTest 通过；共享 root 校验和 lazy work extraction
补完后重新全构建，同样 8/8 通过（240.86 s，`selected-ctest-final.log`）：

| 测试 | 断言 / 用例 |
|---|---:|
| TileIR | 340 / 10 |
| XIR target-info / resources | 2,015 / 12 |
| Metal4 timing | 8,514 / 6 |
| XIR → Metal4 数值/边界 | 238,024 / 4 |
| XIR → SIMD Runtime | 5,085,897 / 18 |
| XIR → SIMD LLM | 2,370,853 / 6 |
| attention 非法 heads/block 子进程 | 2 / 2 进程 |

随后仅补旧 XIR 测试 CLI 参数解析，全构建通过。收尾上游整合再次全构建后，
按**完整精确名称**运行其 17 个相关测试，合计 13,598 条断言通过；每次只执行一个
指定用例，其余 19 个由名称选择跳过。三组深层预算负例未计通过。第一次使用 glob
过滤时，vendored matcher 不支持 `*`，产生 0 断言/全 skipped；该日志明确不算通过。
未修改测试框架或删除原负例。七个修改的 C++ 编译单元通过 clangd 检查
（0 errors，保留原始 tidy warnings，不称 warning-free）。

12 个修改的源码/测试文件通过 clang-format dry run，源码 diff 无空白错误。
no-throw 检查覆盖 2,575 个项目 C/C++ 文件（排除 200 个第三方文件）。隔离源的
Doxygen 与严格 Sphinx 构建通过；文档检查通过 72 个 HTML、5,426 个本地链接/资源
及 199 个兼容锚点。新增资源准入小节在 1280/390 px 下渲染检查无横向溢出。
验证日志保留于 `validation/`；原始失败/全 skipped 日志也保留，不计为通过。

新资源测试覆盖预算恰好/差一字节/零预算、candidate-dependent limits、W32/W64、
auto/fixed、拒绝原因和 cost 隔离；uint8/FP16/BF16/FP32/FP64 的实际 XIR alloca
类型是独立计数 oracle。W64 和 FP64 是通用 IR 测试，不是本机 Metal 硬件能力声明。
循环 carry、动态 extract、map、load/expression fusion 和 pointwise alias fallback
均保留表示级核对。一个 65 次 reduction、4 partitions 的特殊用例，body 内有
68-byte snapshot，静态生成 4 seeds + 4 bulk + 1 tail，期望为 **612 bytes**，
不是一次 68 bytes，也不是运行次数乘 68。

Metal4 新用例为 LayerNorm 与 masked softmax 的 `17×4097`：fixed local=1
超预算必须可恢复拒绝，随后同进程 auto=0 选择 W32，并检查完整 FP64 oracle、
输入不变性和输出 guards。静态预算与 row 数不相乘，因此小 row 数就能回归之前
`1024×4097` 的预算问题；它不是重新测完原大尺寸性能矩阵。

保留限制：旧 `test_tile_xir` 的 recipe-depth/SSA-expansion 负例仍遇到深层 fatal
检查，与预期的可恢复错误不一致。一次传入名称过滤的运行实际未过滤（旧入口是
`main()`，没有解析 argv），触发 recipe depth 的 abort，日志保留。入口已补标准
参数解析；不删除这些负例，也不声称完整 `test_tile_xir` 通过。

## 性能证据和接续

[前一个固定 batch 矩阵](../m1-max-20260910-metal4-timing/README.md) 仍保留
35 OK、2 Error、107 NotRun；本次不能改写其状态或二进制身份。

[RoPE 机器码审查](../../../../../src/tile/ROOT_MAPPING_COST_NOTES.zh.md)
基于 9 月 9 日已归档代码：大尺寸保留四份 snapshot 与两遍输出循环，小尺寸有
整行展开造成的 spill/reload。对照 Inductor 的共享输入、双输出单遍结构，下一步
先验证既有 guarded pointwise fusion，再把 mapping、materialization/fusion、
chunk/unroll 作为联合候选。不能只用静态 snapshot 字节拟合寄存器或加速比。

所有常见 LLM kernels 达到或超过 Torch/MPS/BLAS 的目标仍未完成。
