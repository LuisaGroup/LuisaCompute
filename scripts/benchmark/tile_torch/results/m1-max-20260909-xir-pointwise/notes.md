# XIR guarded pointwise：实现与正确性检查点

新增通用的同域表达式 DAG streaming realization，默认关闭。
它不识别 RoPE/GELU 等算子名称，也不要求用户增加 noalias、ownership 或执行层级标注。
跨参数的实际 byte interval 不相交时进入 streaming 路径；否则保留原先的 Tile 快照。
同一参数上的多输出由坐标区间分离或相同的逐点 store 映射判定。

## 已验证

- 完整构建、XIR 结构测试、SIMD Runtime 与 LLM 回归；开启融合的 LLM 回归也通过。
- W2/W8/W16，aligned/ragged 域，分区输出、共享表达式、同址/错位 alias、相邻 byte interval、重叠输出。
  性能配置下的 full-packet specialization、predicated effects、cohort-private access 组合也通过 alias 测试。
- 负 origin、溢出 origin 和部分有效的 65 元素 Tile：明确断言融合被触发，避免只验证未优化路径。
- 六类算子 × 四档尺寸：48 份 Runtime 完整输出、72 份实际 native 入口输出均通过 FP64 reference 检查。
  Native on/off 在所有 24 个 case 上逐位相同；off 的 LLVM 和机器码与前一检查点完全相同。
- Doxygen、Sphinx 严格构建、链接检查以及桌面/390px 页面检查。分析和说明继续放在既有文档结构里。

数学及输入覆盖沿用前一 native cohort：FP32、固定 local=8、W8/B32、单 CPU 线程。
这不是任意输入数值分析、低精度验证或 held-out 性能评估。
别名测试使用单个逻辑 program 验证 overlap；没有引入违反 `parallel` 独立语义的数据竞争。

## 尚未建立新的性能结论

采集期间机器存在其它工作区的持续构建和很高的系统负载。
`native-smoke.json.gz` 的一轮短样本只用于实际入口正确性检查，明确设为 `timing_not_comparative=true`。
不得把这些数据用于速度比、Torch/MPS 胜负、solver 选择、cost model 拟合或更新性能排行榜。
正式的六次排列 paired native replay 已准备好，但本检查点没有运行合格的性能对照。

当前 cost prior 仍估计原快照路径；guard、fallback、code size、活跃值和尾部控制流还未校准。
完整方法、适用域与反例见 `docs/source/internals/tile/xir.md` 的 guarded pointwise 小节。
没有修改 Metal lowering、reduction tree、SIMD mapping 默认值或自动选优策略。

## 证据与下一步

`audit.json` 重新读取完整输出并检查 source/library/object 哈希，验证 on/off 位一致性；
`provenance.json` 标识 archive + overlay 的隔离构建来源，不能视作干净 git checkout。
Git 不存放大张量 payload；原始输入/输出留在记录的 raw 目录。回调的临时 guard 已在执行时检查，但不保留。
实际新生成的 LLVM、ORC object、native dylib 和反汇编已压缩归档；Inductor 和 C++ timer 沿用前一检查点的已验证二进制。

下一步是在无并行构建的窗口运行 on/off/Inductor 的完整成对计时，保留所有尺寸的回退和负结果，
再决定哪些 realization-derived features 应进入 backend policy 与 solver。
`origin/next` 已 fetch，仍为 `4b0c02384`；未把上游合并混入这个对照。
