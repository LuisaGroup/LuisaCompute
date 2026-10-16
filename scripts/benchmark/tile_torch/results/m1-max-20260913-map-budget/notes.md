# Structured map 展开预算：修复编译膨胀，不等于算法加速

本检查点接续 [ranking 首轮筛查](../m1-max-20260913-ranking/notes.md)。原 SIMD ranking 回归在 LLVM MachineScheduler 中超过 300 秒；加入共享结构预算后，同一 26 个配置、4 类输入的完整测试先后以 22.69 / 25.87 秒通过。这里是整项测试耗时（含 JIT），不是 kernel 加速比。

## 改动与边界

`LowerOptions` / `PlannerOptions::max_unrolled_region_work` 默认 4096，用饱和加乘估算 small map 内嵌 execution region / MMA 被重复展开的工作量。结构工作超限时使用运行期 map loop。普通 pointwise map 保持原来的元素阈值；0 禁用额外预算，`max_unrolled_tile_elements=0` 保留完全展开的诊断形式。

Coordinate classification、动态 extract 的 snapshot 需求、emission、resource accounting 和 planner 的读写成本共享这一判断。不能只改 loop emitter，否则被动态索引的源 Tile 仍可能留在不能直接索引的 SSA 表示中。后端可以通过公开 planner 配置调整该预算；直接 benchmark 可用 `LUISA_TILE_BENCH_XIR_REGION_WORK` 做诊断对照。

这是编译表示的代码膨胀保护，**不是**精确机器指令数、周期模型、全 LLVM pipeline 的代码大小上界，也不改变直接 MMA/fold 的展开策略。Top-K/sort 仍为 O(N²) 参考库实现；不能把首轮修复前的性能表标成修复后的成绩。

同时补上 deferred recipe 的 SSA 深度预检：纯 map 或 map/expression 链超过 64 层时，analysis/lower/planner 返回错误，保留 emitter 的末端检查。63/64/65/70 边界测试通过。新资源测试最初漏算 MMA 初始化 `full(shape(65), base)` 的 snapshot，修正手写期待为 **524 B / 3 allocations**；实际 Alloca、资源分析和 planner 的独立对账一直保留。

## 验证结果，包含未解决的问题

| 检查 | 结果 |
|---|---|
| 配置的全量 build | 各轮成功；既有 TVM deployment-target 链接警告保留 |
| SIMD ranking | 473 assertions，104 dispatches 通过；22.69 / 25.87 s |
| Metal4 ranking | 首轮 300.03 s Timeout；加进度日志的逐 case 重跑 473 assertions 全部通过 |
| TIRx/Metal ranking | 237.20 s 通过（之前初测 4.20 s）；它不使用新增的 XIR 结构预算 |
| SIMD Runtime / LLM，候选 v1 | 62.41 / 20.86 s 通过 |
| v2 target-info / SIMD ranking / LLM（含新 PV probe） | 3/3 CTest 通过；LLM 21.54 s |
| v2 deferred-depth 独立测试 | 48 assertions 通过 |
| v2 target-info | 2264 assertions / 13 tests 通过 |
| v2 host tests，逐项 fresh process | **18/20 通过，2 项 Error**，不是全绿 |
| Python measurement/oracle/metadata units | 88/88 通过 |

两项 host Error 是既有的 SSA expansion 预算错误传播缺口：`tile_xir_large_tiles_have_bounded_code_and_eager_load_snapshots` 和 `tile_xir_expansion_budget_is_fail_closed` 要求可恢复 rejection，现有 `_charge` 却调用 fatal diagnostic，进程退出 134。此次没有把测试改成 death test 或放宽预算来掩盖它；完整 host suite 仍不通过，需要后续修复显式错误传播。

Metal4 失败采样停在 `synchronize()` 新提交的空 compute command buffer 的 `Submission.completed` 等待，不是 LLVM 编译。逐 case 日志中连 1×1×1 排序的 upload/dispatch/download/sync 都会从毫秒抖到几秒，之后全部通过。完成位还包括反馈回调与 host 清理；现有证据不能区分 GPU 队列迟滞和完成反馈迟滞，不能宣判 GPU 永久挂起，也不能归因于 map 预算。无计数器 GPU control、dispatch 探针与 E2E 必须继续分开，当前桌面状态不适合精确 cost 校准。

## Attention 衔接与证据

另增加了默认关闭的 PV reduction 分解探针及 4 个 FP64 oracle 测试，QK/PV 开关独立；默认 kernel 仍为 MMA。它用来检查 contribution-axis mapping，不能把 benchmark 源表达式改写称作 production planner 优化。新 attention 实测记录单独归档，不与本检查点的 ranking 首轮表混合。

`evidence.tar.gz` 保存完整检查日志、首次失败源码、sample 及最终测试源码快照。源基于 `147700bd431eb054694ba649817b7f50a335b2b1` 的隔离导出，加本次有明确归属的修改；**没有**合入工作区受保护的 TIRx matrix-initializer / guarded-input 改动。不是完整可执行工具链镜像。`validation.json` 保存逐项 host 退出码；不把被过滤的其他测试算作该次独立调用通过。
