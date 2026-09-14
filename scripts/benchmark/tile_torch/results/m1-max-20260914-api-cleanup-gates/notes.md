# Tile API 清理验证（2026-09-14，M1 Max）

API 清理已完成；**完整构建通过，选定回归 13/15 通过，整体 CTest 为失败（exit 8）**。
本记录不是性能报告，也不表示整个工作区或所有后端均通过验证。

## 接口收敛

| 范围 | 当前唯一接口 | 删除内容 |
| --- | --- | --- |
| XIR planner | `plan(Function, ExecutionTargetInfo, PlannerOptions)` | 原始 `ExecutionTarget` 重载、`plan_with_target_info`、兼容模板 |
| TIRx reduction cost policy | `reduction_cost(candidate, model) -> ReductionCost` | `reduction_score` 虚函数和基类兼容转发 |
| FP8 IR 类型 | `FLOAT8_E4M3FN` | `FLOAT8_E4M3` 别名 |

提交：`413cefa3e`（XIR/FP8）和 `0d0c77833`（TIRx）。调用方、测试和正式文档均随接口迁移，
不保留兼容入口。`ExecutionTarget` 仍是描述线程池目标的数据类型，不再直接充当 planner 参数。
TIRx 的 analytic 公式、wave 计算和 Service policy 保持不变；solver 只使用完整的 `kernel_score`，
不会再次乘 wave。编译期测试覆盖新接口签名和旧入口不可调用；动态库符号检查也未发现被删除入口。

## 验证范围和来源

为排除其他并行开发的影响，测试使用已有 full12 冻结源码，叠加上述两次清理的 **12 个 C++ 文件**，
不复制当前工作区的其他修改。已有 27 文件快照与清理文件的去重并集为 35 文件；构建和 CTest
前后均校验该集合未变。记录不是整个源码、SDK、目标文件或二进制的完整可复现封包。

- Source：`/Users/mike/.cache/luisa-tile/metal-program-team.xIQgCS/source`
- Build：`/Users/mike/.cache/luisa-tile/metal-program-team.xIQgCS/build`
- 完整构建：`cmake --build <build> -j 6`；不是仅构建被测 target。
- SIMD、Metal4、TIRx bridge 和 tests 均启用，LLVM 22。
- 三次完整构建均 exit 0：247.990 s、41.738 s、17.750 s；最后一次只补齐格式修正后的对象。
- 12 个改动 C++ 文件格式检查通过；9 个 TU 的 clangd/clang-tidy 检查为 **0 errors、222 warnings**。
  未做历史 warning 对照，不宣称零 warning 或全部都是历史 warning。
- 仓库 no-throw 扫描通过：2595 个项目 C/C++ 文件，200 个排除文件。

## CTest 结果

串行运行下面全部 15 项，单项超时 240 s；没有重试、删项或放宽时限。
总时间 428.20 s。完整命令、环境和日志保存在归档的 `ctest/`。

| 测试 | 结果 | 秒 |
| --- | --- | ---: |
| `test_tile_dsl` | Pass | 1.22 |
| `test_tile_types` | Pass | 1.12 |
| `test_tile_types_simd` | Pass | 1.14 |
| `test_tile_types_metal` | **Timeout** | 240.03 |
| `test_tile_xir` | **Abort** | 1.00 |
| `test_tile_xir_target_info` | Pass | 1.12 |
| `test_tile_xir_program_team` | Pass | 0.43 |
| `test_tile_xir_metal`（Metal4） | Pass | 107.19 |
| `test_tile_xir_runtime`（SIMD） | Pass | 52.95 |
| `test_tile_values_cpp20` | Pass | 0.75 |
| `test_tile_values_cpp23` | Pass | 0.44 |
| `test_tile_tirx_values` | Pass | 2.41 |
| `test_tile_tirx_execution`（CPU） | Pass | 7.49 |
| `test_tile_tirx_planner` | Pass | 0.67 |
| `test_tile_tirx_execution_metal` | Pass | 10.23 |

### 尚未解决的失败

1. **XIR 静态展开预算负向测试**：`tile_xir_large_tiles_have_bounded_code_and_eager_load_snapshots`
   直接调用 `lower(... max_expanded_values=256, max_unrolled_tile_elements=0)`，耗尽预算后
   `Lowerer::_fail` 调用 `LUISA_ERROR`，而测试预期返回带 error 的结果。崩溃发生在下一行迁移后的
   `plan` 之前；该 direct-lower 测试和失败路径在清理前相同。历史独立运行记录也显示该 suite
   在旧 baseline 和候选均 exit -6。不是本次 target-info 迁移引入的默认行为变化；此处未修复。
2. **旧 Metal backend 的类型测试超时**：使用 `metal` + TIRx，而不是 `metal4`。
   1 秒线程采样显示主线程在 `test_storage<half>` 的首次提交后通过
   `MetalStream::synchronize` 等待 `-[_MTLCommandBuffer waitUntilCompleted]`，CPU 占用为零。
   这只定位了等待位置，尚未确定根因，也未通过本轮 baseline A/B 排除回归。
   不能用 Metal4 或其他 TIRx 测试通过替代这一失败。

## 原始记录

归档 SHA256：`0878ffae810b75011cc459f5704947ca60616f61cc538ecbe1e2f83fb2c55c93`。

`results.tar.gz` 包含本轮 `luisa-tile-api-cleanup.VUOgJS/` 下的命令、日志、逐文件 SHA256、
源代码补丁、格式补丁、语法诊断和 Metal 线程采样；另附历史 XIR 失败的
`isolated-host-results.json` 与 `baseline-isolated-14/` 的原始命令/日志。
历史记录来自 `/tmp/luisa-metal-short-reduce.JdDz9f/postbuild3-host/`，不是本轮重测。
历史 summary 引用的其他运行不全部包含在此小型归档中。

原始目录仍为 `/tmp/luisa-tile-api-cleanup.VUOgJS`。归档不含新性能数据，也不改变之前报告的
MPS/Torch 性能结论。本次提交没有纳入工作区中的 pipeline 命名、MMA planner 或 Metal stream 等其他修改。
