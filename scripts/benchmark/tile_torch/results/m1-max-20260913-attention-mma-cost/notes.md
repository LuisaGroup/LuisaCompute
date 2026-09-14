# Attention MMA 工作模型一致性检查点（2026-09-13）

本目录记录 compiler/cost-model 回归，**不是新的 benchmark cohort**。未执行 GPU 性能测试，也没有新的 Torch/MPS/BLAS 对照；此前 native-entry 结果保持原口径，不用新的 cost score 重写历史排名。

## 改动

- XIR cost extraction、资源分析共用同一组 representation options；MMA 读取计数使用实际准入的 output block 和 runtime K 方案，包括预算回退。
- 广播侧每组每 K 一次投影，另一侧每输出每 K 一次；乘加数量不变。尾组、外层 scope 重复次数保留。
- cap 新增的小 Tile 快照收取定义处写入及动态读取；有快照但索引为常量时仍使用 SSA，不额外收取数组读取。动态性按 MMA 操作数的坐标依赖区分。
- `ExecutionWork::mma_per_packet` 提供不带硬件系数的乘加、输入/seed 投影、runtime K 循环调用和迭代计数。Thread-pool 调度不丢失字段，backend policy 可以读取，不能与已有 weighted work 双重收费。

这些是逻辑工作 prior，不是精确 native 指令、内存事务或时间。未增加 cache/寄存器/clone 的虚构收益系数，未改变 R1/cap0 默认或新增自动候选搜索；本轮未修改 emitter。快照容量和原严格 K 顺序保持不变。

## 独立期望与验证

新测试的 432 个组合：K∈{0,1,9}、N∈{0,1,5}、R∈{1,2,4}、cap∈{0,8}、转置/交换各两种、外层重复次数∈{1,3}。另有四个大输出配置、一个预算回退、四个常量输入配置，共 **441 个 planner 配置**；这是 host 测试数量，不是 GPU kernel 执行数量。

设 W=8，默认数组读取/写入 prior 单位为 `g=2×W=16`。A/B 跨独立 programs 广播，C 各 program 独占一行；K=9、N=5 时外部工作为 `E=9+45+2×5×16=214`。一次 MMA 的参考期望为：

| 配置 | arithmetic_per_packet | memory_per_packet |
|---|---:|---:|
| cap0，R1 或 R4 | 90 | 214 |
| cap8，R1 | 90 | 2518 |
| cap8，R4 | 90 | 2086 |

后两项分别是 `E + 54g + 90g` 和 `E + 54g + 63g`。重复 MMA 只增加每次读取，不重复收费于外层定义的 54 次 snapshot stores。小 CONSTANT 在 cap8 下仍生成 SELECT/CMP 链，测试独立遍历 pre-cleanup XIR 检查其数量，再核对动态 prior，未以优化后的 native 性能替代该事实。

- 全量 selected-tree build 完成后，四个 CTest 全通过：schedule codegen **34.89 s**，target-info **0.86 s**，SIMD runtime **57.26 s**，SIMD LLM **32.74 s**，合计 **125.76 s**。测试墙钟时间不是 kernel benchmark。
- 七项精确名称的 host 回归分别执行了非零断言：root cost 5、map cost 21、expression/reduction cost 220、load/reduction cost 183、task schedule 9900、task policy 24、planner 46。每进程其他 19 项为有意未选择，不能算全部 host suite 通过。
- 两个更早存在的 host SSA-budget fatal 问题未解决，本轮没有运行完整 host suite 来宣称绿色。
- clangd/clang-tidy：planner 0 errors / 12 warnings；target-info 0 errors / 42 warnings。Warnings 保留，未声称 warning-free。格式和 scoped diff 检查通过。

## 保留的失败与限制

初轮新增零输出夹具错误地同时绑定零尺寸物理 buffer，144 个配置被 `invalid XIR buffer footprint` 拒绝。保留失败源码与日志；修正仅将物理绑定最小长度设为 1，逻辑 Tile 仍为零输出，之后完整重建与回归通过。

另三次 glob 形式的额外 host 命令返回成功但 **0 asserts / 全部 skipped**。本地 UT 的简化 matcher 未实现 parser 生成的 `.*`；这些运行明确不计为验证，随后以七个精确名称重新执行并核对断言数。未借机修改无关测试框架。

## 来源与复现范围

- 工作分支 next，改动基线 `f1ced773a`；运行的是 `/tmp/luisa-next-integration.roZbN8/{source,build}` 隔离导出构建，不是主树 protected TIRx WIP 或尚未合并的 upstream next。
- `changes.tar.gz` 含三个最终 C++ 文件和基于该提交的 scoped patch；它们与构建源逐字节一致。完整前序源包可参考相邻 `m1-max-20260913-attention-mma-strided/sources.tar.gz`，本包不伪装为全仓可独立构建快照。
- `validation.tar.xz` 含初轮失败、后续 build/test/syntax 输出、空跑的 glob 输出及实际 exact-name 输出。首轮大构建输出仅在会话中保留，归档包含后续两轮完整构建日志。无性能测量或 GPU 证据隐含于其中。
- 两个归档校验和见 `SHA256SUMS`。

下一步是保留严格贡献顺序的逐 MMA 一维/二维输出微块候选，以及真正的 phase ownership/transition 成本；必须继续用完整程序的 native-entry/GPU kernel timing 检验，不能把较低 prior 当作达到性能目标。
