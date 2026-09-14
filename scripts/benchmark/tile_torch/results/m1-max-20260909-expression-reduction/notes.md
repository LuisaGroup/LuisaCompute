# 表达式—归约融合：结构改善，性能尚未验收

2026-09-09，Apple M1 Max（10 CPU cores、64 GiB），macOS 26.6.2。

这次实现了一条通用的 producer/consumer 融合规则，而不是 softmax 特判。
**正确性和代码结构已经验证；没有新增合格的加速或 Torch/MPS/BLAS 胜负结论。**
默认仍关闭，成本系数未校准，有限求解器暂不自动选择这个开关。

## 实现了什么

`enable_expression_reduction_fusion` 让已经需要物化的纯 elementwise
生产者加入第一个合法的 unordered reduction 遍历。生产者每个点只计算一次，
归约内重复读取复用同一个标量；后续还有消费者时，同步保存原 Tile 快照。
例如 `exp`、保存临时结果、累加可以发生在一次遍历里，后续使用不会重新算 `exp`。

```text
不可变的输入定义
       │
第一个 reduction 遍历
  producer[i] ──> 标量复用 ──> 原有 reduction tree
       │
       └── 后面还要用？──> snapshot[i]
```

planner 和 lowering 共用准入：维度/范围双射、直接归约坐标、闭合的
unordered 更新、不越过写入和 stage。严格 fold、反向索引、非 unit 包装和
非 elementwise 生产者保留原实现。没有增加 DSL primitive、noalias 假设、
算子名称分支、归约单位元、重结合规则或数学近似。

成本提取仍将生产者计算计入一次，只移除第一个消费者的私有读取，以及确实
不再需要的快照写入。这是现有相对 work prior 的一致性改进，不是机器周期模型。

## 实际代码变化

固定 W8、local=8、block=32、单 CPU 线程、FP32、fast math 关闭；旧 load fusion
和 pointwise fusion 均关闭。每类四个尺寸：`17×65`、`129×768`、`257×1538`、
`1024×4097`；RoPE 将奇数列数补成 `66/768/1538/4098`。

| 算子 | 每个 case 融合的生产者 | Schedule blocks：768 列 / 其余尺寸 | 结论边界 |
|---|---:|---|---|
| Masked softmax | 2 | 25→19 / 41→31 | mask/max、exp/sum 两处合并 |
| LayerNorm | 1 | 22→19 / 36→31 | 共享表达式加入方差归约遍历 |
| RMSNorm、SwiGLU、GELU+residual、RoPE | 0 | 不变 | 16 个 case 的 LLVM 和 ORC object 均逐字节相同 |

不能只看 block 数：softmax 的 768 列 full-packet clone 从 1,782 增至 2,252 条
静态指令，LayerNorm 从 1,740 增至 1,882 条。两者保留的快照和 native workspace
大小未减少。静态 store 节点数还可能因 partial seeds、展开和尾部路径增加，
不能当作动态写入次数。实际 LLVM 已同时出现 `exp → snapshot store → fadd`，
但代码大小、活跃值和吞吐的取舍仍需合格的 native 计时。

## 验证与来源

- 基线为 `2634be45d`，已包含 `next@8911828eb`；递归 Git archive 固定 19 个仓库，
  只叠加本次八个 C++ 源文件/头文件。现有 11 个用户修改文件和所有依赖 checkout 保留。
- 隔离配置完整构建通过；35 项 Tile CTest、79 项 XIR/SIMD CTest 通过。
  Metal/TIRx 仍保留为回归路径，但没有重新测量其性能。
- 额外 W1/W2/W4/W8/W16 测试均实际执行，覆盖融合关闭、仅表达式融合、与已有两种融合组合；
  也覆盖别名写入、保留/消除快照、严格 fold、零次/多次循环、stage、边界填充及维度置换。
  初次通配符筛选空跑不计入结果；精确名称重跑要求一个执行的 case 和非零断言数。
  第一次精确名称审计错误地要求 reporter 输出逐 case 的 PASSED 文本，记录也保留并被复跑替代。
- 24 个 case 的 48 个实际 ORC capture 输出和 72 个 native smoke 输出通过完整 FP64 参考检查；
  24 对 off/on native 输出逐位一致。重放也检查输入不变、输出和 private workspace guards。
- 独立审计重新读取 138 份输出：前述 120 份加被排除计时的 18 份输出；
  拒绝八类内存中人为破坏的采样、校验记录或输出。临时 guards 由执行时检查，审计不声称重新读取已释放的 guards。
- 实际 ORC objects、LLVM、反汇编、生成的 Inductor C++、脚本、命令、校验和、测试日志
  随 checkpoint 保留。TorchInductor 2.14.0 的已冻结库直接复用，未宣称本轮重新编译 Torch。
  TVM 依赖也复用本机已有构建。clangd 检查没有错误；planner 全文件仍有 15 条既有风格/转换警告。

## 为什么没有性能成绩

采用共同 C++ timer 调用真实 native entry：计入入口的 block traversal/reset、
内部临时分配和 libc；排除 Runtime、Python、JIT 及调用方分配。拟用两套代码加
Inductor 的六种顺序、每次七个样本，100 ms warmup、约 30 ms/样本。

60 秒预检查只排除了已知渲染/构建进程，仍可见其他持续的后台 CPU 活动；
随后渲染任务也重新出现。因此主动终止了**自己的** native benchmark（exit 143），
没有停止任何其他工作。部分 cohort 的 96 个 timed visits 全部排除，不挑选其中
看起来稳定或更快的子集。第二个 cohort 未启动。

进程采样也不能证明硬件独占或频率稳定；后续准入必须同时检查持续的非 benchmark
CPU 活动，而不是只依赖进程名黑名单。这里的数据只能支持实现和正确性结论，
不能用于更新性能排名、选择默认开关或拟合成本系数。

## 下一步

固定这套源码和机器码，在无并发负载窗口补做完整交错计时；重点看 softmax 的
遍历收益是否抵消 partial 展开和活跃值开销。之后再考虑生产者 DAG 中的 load
共同融合、资源敏感的 cost policy 和求解器选择。SwiGLU/RoPE、矩阵和 attention
差距仍未由本次修改解决。

机器可读入口为 `audit.json`、`provenance.json`、`capture.json.gz` 和
`verification-widths-v2.json`。`excluded-replay.json.gz` 不是可引用的性能排行。
