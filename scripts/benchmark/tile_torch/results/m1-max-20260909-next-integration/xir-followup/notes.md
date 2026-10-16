# next XIR 后续集成

本检查点合入两个共享 XIR 修复，不更新性能排名、融合默认值或成本模型参数。
以后仍在阶段性检查点同步上游；同一轮性能对照中固定源码、依赖和被测二进制。

## 版本与范围

- 上游：`03a0f5158b53768abefea555f480f87ee5bc5a1e`。
- 合并提交：`bc7b1df1f4e8785d50736169a2a7bde3023b231e`，两个父提交分别为
  `77fb22c42b4708368ab1af69b61ab2e98dcf3863` 和上述上游提交。
- `afdc139ec` 修复跨循环迭代的退出分派重构；`03a0f5158` 以有序 copy-in/copy-out
  降低 callable 的多分量 swizzle 引用，保证引用写回先于调用结果的使用。
- 唯一合并冲突是 Metal 测试入口的注释。两边的参数防护代码已经一致，采用上游注释后，
  该文件与上游逐字节相同；没有恢复不安全的 `argv[2]` 访问。
- 没有额外修改 Tile DSL、Tile bridge 或 SIMD/Metal planner/lowering；共享 XIR 的改变
  仍可能影响编译产物，不能据此假定性能不变。

## 验证边界

隔离源码直接从合并提交及递归固定的 Git 子模块导出：19 个仓库、19,146 个文件，
逐文件保存 SHA-256，无源码 overlay，不带入工作区尚未提交的 TIRx matrix 实验。
配置为 Apple M1 Max / macOS 26.6.2、RelWithDebInfo、Apple C++ 编译器、LLVM 21.1.8，
启用 SIMD、Metal、Tile TIRx 和测试。外部 TVM 使用配置中指定的已有本地库，没有重建；
这不是完全 hermetic 的构建，也不包含未启用的 Metal4、CUDA 等后端验收。

`checkpoint.py` 保留首次调用，`recheck.py` 记录最终复核；`source-snapshot.json.gz`、
`pinned-repositories.json` 和 `configure-command.json` 标识具体输入。
构建与正确性测试的耗时只用于检查运行过程，不是纯 kernel 或端到端性能成绩。
此前冻结的 expression/pointwise native 对象、被排除的计时轮次及已接受的历史成绩均不改写。

## 已执行结果

完整配置构建通过；运行测试前再次完成完整构建检查。以下三组 CTest 名称互不重复，
JUnit 均记录零失败、零跳过；它们是选定范围，不是整个仓库、所有后端的全集。

| 范围 | 结果 |
| --- | --- |
| XIR/SIMD：`unit_xir\|unit_simd` | 80/80 |
| Tile：`^test_tile_`，包含 XIR/SIMD、native Metal、TIRx CPU/Metal | 35/35 |
| 完整 Metal fixture：`test_metal_local_codegen` | 1/1 |

新加入的 `test_ast_callable_swizzle` 实际执行 48 条断言。此前的表达式归约融合和默认关闭策略
随 Tile Runtime 回归一并检查；没有重新捕获 native benchmark 对象，也没有做性能回归结论。
最终逐文件复核隔离源码未改变，原工作区 11 个未提交文件及全部依赖 checkout 均保持不变。

首次 supervisor 把可执行文件名 `test_metal_codegen_regressions` 当成 CTest 名称，
`--no-tests=error` 拒绝了零测试选择；该调用不计为成功。最终使用实际注册的
`test_metal_local_codegen`，先复核完整构建再执行。失败调用、空 JUnit 和正确重跑均保留，
`verification-final.json` 是有明确范围的最终结果，不覆盖 `verification.json`。

冲突文件的语法检查为零错误、一个原有 unused-include 警告。合并行的格式检查通过，
整文件仍有七处上游格式诊断；文件与 `next` 完全相同，没有借合并重排无关代码。
上游新增的 loop-epoch 调查页接入既有 performance/validation 导航；本记录仍由 Tile
正确性文档索引，未新增顶层文档体系。文档生成与本地链接检查单独记录在 `documentation.json`。
