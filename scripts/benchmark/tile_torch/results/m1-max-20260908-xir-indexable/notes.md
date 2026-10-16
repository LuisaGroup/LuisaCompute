# XIR 可索引 Tile 快照：归约更快，但仍须消除整行展开

## 结论

2026 年 9 月 8 日，Apple M1 Max。这次把
[Torch SIMD 检查](../m1-max-20260908-torch-simd-inspection/notes.md)指出的一个结构性问题落实到了
**通用 Tile→XIR lowering**：动态 Tile 提取不再在每次归约迭代里扫描整块 Tile。
无需新增 DSL 原语或手写 Memory，适用于有动态提取的 Tile SSA，不按算子名字匹配。

冻结二进制的两轮交错 A/B 中，七个归约相关案例的同步 Runtime 吞吐提升 **1.17–5.32×**；
不含动态提取的 SwiGLU 对照基本不变。32 次完整 FP64/guard 检查及独立 Python FP64
复核通过。**不是与 Torch/MPS 的性能对比，也不是纯 kernel 时间。**

负面结果同样明确：新方案的 JIT 更慢，64×256 RMSNorm 的代码和栈更大；宽度
1537/4096 未在 60 秒内完成探测，16384 触发静态展开预算。下一步仍是有界向量循环、
贡献维分布及按阶段规划临时值，不能把这次修复描述为完整 planner 或性能目标达成。

## 1. 同一语义程序，改变的是物理值表示

修改在 `src/tile/bridge/xir/`，不修改 Tensor/MemoryRef、Nest、reduce 的语言契约。

```text
旧：Tile 的 D 个 SSA 元素
             ↓
    for r in reduction:
        比较 index(r) 与 0…D−1，选择一个元素
        更新 accumulator                         O(D×R) 选择工作

新：在 Tile 定义处，把 D 个元素保存成局部快照
             ↓
    for r in reduction:
        带边界保护地读取 snapshot[index(r)]
        更新 accumulator                         O(D) 保存 + O(R) 索引
```

常量提取和可证明的 Tile-map 坐标直接投影到 SSA，不先生成 SELECT 链再等后续优化。
动态提取的 compiler-owned array 在定义处生成，**不在首次使用处懒加载、不在每次归约
迭代里重复保存整块输入**。未知的静态表达式仍保守回退；单元素/空 Tile 不分配数组。

loop-carried Tile 使用完整 PHI 集合完成同时更新后再保存自己的快照；loop result
在 exit 保存，覆盖零次迭代。输入 View 即使是 const，也允许与另一个可写参数重叠；
后续 store 不得改变之前得到的 Tile。测试显式覆盖这种别名，不添加无依据的 noalias。
严格 fold L/R 的贡献次序、操作数方向和浮点配置保持不变。

新增的 `LowerOptions::max_local_bytes` 默认 256 KiB，是**每个逻辑 worker 的静态
快照分配总和上限**，不是目标总栈大小或峰值存活量；W-wide packet 会放大这份存储。
原有 SSA 展开预算也计入 allocation/GEP/store。超过预算直接失败，不截断或降精度。

planner 与 lowerer 共用坐标类别分析；成本从“每次提取 D 个选择”修正为定义处保存和
运行时索引读取。**没有拟合新系数，没有新增贡献维求解候选**；当前相对成本仍未校准，
也没有把 JIT 或实际 spill 成本纳入优化目标。

## 2. 吞吐收益跨多个归约表达式，但不外推到其他路径

以下单位为每次调用的 **µs，同步 host-wall batch 时间**。计时排除编译、初始分配上传；
保留正常 Luisa Runtime command-list、CPU 线程池调度和同步。另存逐次调用 latency，
不把它与 batch 时间混合。每轮 7 samples、目标 30 ms/sample、预热 100 ms；每个
case 两轮顺序为 `旧→新`、`新→旧`，不是从多次尝试里取最快值。

表中时间是两轮 p50 的中位数；加速是两轮“旧/新”配对比值的中位数，所以不一定等于
显示时间的相除结果。两轮只支持描述性结论，不构成置信区间或跨设备泛化证明。

| 算子与尺寸 | 旧 XIR | 新 XIR | 旧/新配对加速 |
|---|---:|---:|---:|
| RMSNorm 17×7 | 0.392 | 0.335 | 1.170× |
| RMSNorm 17×127 | 29.492 | 8.140 | 3.623× |
| RMSNorm 64×256 | 225.980 | 87.514 | 2.582× |
| RMSNorm 1024×256 | 874.441 | 268.943 | 3.252× |
| RMSNorm 64×513 | 768.288 | 144.550 | 5.315× |
| LayerNorm 64×256 | 390.033 | 117.339 | 3.338× |
| Masked softmax 17×65 | 11.688 | 3.875 | 3.017× |
| SwiGLU 17×65（不含动态提取） | 4.356 | 4.293 | 1.015× |

原始数据在 [A/B report](ab/report.json)，独立重算结果在 [audit](audit.json)。七个
归约 case 的两轮方向均一致；SwiGLU 的约 1.5% 差异不解释成该变换的结构性收益。
这个修复同时包括静态提取投影和动态提取快照，尚未通过独立消融把两者的收益拆开。

W=8、请求 8 CPU workers，各 case 的 root order 和 block 大小在 A/B 间保持一致
（1024-row 案例为 128 workers/block，其余为 32）。
请求线程数不意味着全部利用：17 个 programs 只有一个 block，64 个只有两个。
每个变体都使用自己的冻结目录；可执行文件与 23 个共享库在整个 cohort 前后指纹不变。
不是用可变工作区的库冒充旧二进制。

## 3. 机器码证明选择链消失，也证明资源问题未解决

单独捕获的新 64×256 RMSNorm 通过两次全输出/guard 验证。
[实际 ORC object 的反汇编](code/object.asm.gz)中 `0x13724..0x13810` 的归约循环
只有 59 个静态指令槽：8 个带 lane mask 的加载点、两个四宽 NEON 累加、字节 index
每轮加 4，到 1024 结束。它仍处理 256 项贡献，但没有每轮比较 256 个可能下标。
[LLVM](code/kernel.ll.gz)也保留一次动态 local gather，而非整块 Tile 的选择链。

这不是理想的连续 NEON feature 循环：lane 仍代表不同 row，本地数组仍由后端按 worker
布局并通过 gather/scatter 访问。大量静态 load/store、mask 和 spill 继续存在。
捕获 object 的 `__text` 从前次证据的 214252 增至 **262772 字节**；相应 kernel
frame 从 33104 增至 **37376 字节**。不能拿“去掉选择链”推导代码体积或栈一定下降。
完整 `.ll/.s/.o` 原样 gzip 保存；`audit.py` 直接解析 Mach-O section 并检查实际循环。

## 4. JIT 回退与大尺寸失败是下一步的约束

普通、无汇编导出的 A/B 同时记录了 `compile_ms`。两个中等 RMSNorm 案例分别是：

- 64×256：约 **3.645→5.684 秒**；尽管热吞吐快 2.58×，编译变慢。
- 64×513：约 **14.851→24.113 秒**；热吞吐快 5.32×，编译仍变慢。
- 小型 17×7：约 **42.2→132.6 ms**；不能用很小的热吞吐收益掩盖启动代价。

这些是有限次数的实际编译耗时，不是编译器复杂度证明。若优化目标包含启动成本，
需要评估 `T_compile + N_calls × T_runtime`；不能只按热吞吐选默认候选。

[大尺寸探测](large/report.json)单独保存，不进入上表。17×1537 和 1024×4096 两个
版本都超过 60 秒进程预算；64×16384 两个版本都以静态 SSA expansion budget 错误退出。
没有完整产出或数值验证，故不报告这些尺寸的 kernel 吞吐。

1024×4096 的两个 2 秒 native sample 分别位于普通 ORC lookup 的
[MachineSinking](code/baseline-4096.sample.txt)和
[MachineCSE](code/candidate-4096.sample.txt)。**这次是正常 JIT，不是额外汇编复制**；
不能把此前 assembly-copy 超时与本次混为一谈。采样只证实当时所在阶段，不代表整个
60 秒的精确时间分布；线程池在等待编译结束，不是 reduction kernel 死锁。
一次对已超时退出的 1537 进程的采样没有取得数据，不作为热点证据。

## 5. 验证、复现和下一步

独立 CMake 完整构建后，**35/35 Tile CTests** 通过；额外 W=1/2/4/16 各跑完整
`test_tile_xir_runtime`，每次 728 assertions / 7 tests 通过。覆盖 FP32 cancellation、
负零、fold L/R、零次循环、多元素 simultaneous carries、别名覆盖、offset views、
ragged packets、空/单元素/越界提取。两个早期带 wildcard 的测试命令实际跳过了全部
测试，已保留日志且不计为验证；最终使用无过滤运行。

实现和测试的四个实际 translation units 经 clangd/clang-tidy 检查无错误；保留既有
style 警告及必须按值传递的 DSL 签名警告。首次测试还抓到非法 XIR 调试名字
`tile.snapshot`，修正为 `tile_snapshot` 后完整重建重测，失败日志未删除。

基线是 `45fc2e2cd` 时已验证的隔离构建：生产 Tile/XIR 来自 `f5daf25e6`，benchmark
helper 已含 `db52dbc59`。新版本只镜像本次 XIR bridge/header/tests 变化；没有纳入主
工作区未完成的 Metal matrix 实验。LLVM 21.1.8 编译，LLVM 22.1.8 反汇编。
构建/测试日志及 [源文件指纹](code/source.sha256)补足二进制收据。它不是全部外部
依赖均可离线重建的证明。111 个 benchmark Python 测试通过；无 Torch 的首次运行
有两个 skip，最终带 Torch 的运行复用本机缓存的 Torch 2.14.0，111 个均执行通过。
Doxygen XML 重新生成（保留其 warnings），Sphinx `-W` 构建通过；50 个页面、4166 个
本地链接/资源、199 个兼容锚点通过。1280/390 像素页面检查无全页横向溢出；表格保持
原生横向滚动，快照图改为窄屏可读的竖向结构。页面继续沿用仓库原有结构。

`run_ab.py` 需要 numpy，接受冻结的 `--baseline/--candidate` 目录、全新 `--output`
与 `--tensors` 目录以及 `--cases`；`capture_code.py` 单独捕获机器码。
`audit.py` 不调用 runner，重算配对统计并用六个变异测试拒绝错误口径、NaN、缺少数值
验证、不同输入及伪造加速。全量 tensor 保存在 receipt 指向的临时目录，未提交到 Git；
公开收据中的哈希不能替代重新运行全输出比较。

下一步按依赖推进：**有界局部向量循环 → 输出/贡献维分布 → scratch/重计算与阶段
存活期 → 带 JIT、mask、spill 和线程粒度的候选成本**。保留这里的快照路径作为
效果/别名证据不足时的语义 fallback。先把 CPU native entry 与 Runtime E2E 的计时
边界补齐，再同输入比较 Inductor native entry；不以本轮胜过旧 XIR 作为胜过 Torch。
需要回答的开放问题是：哪些程序能合法融合到同一有界循环，哪些需要跨阶段快照，
以及候选能否在 held-out 尺寸和算子上同时降低 JIT 与热运行成本。

报告结构说明：遵循用户指定的仓库 Markdown/Sphinx 交付，不另建报告 App。作用域和
指标定义移到表格之前；方法与负面结果分段说明，下一步与开放问题合并。本表用于
精确 case 查阅，两轮数据不足以画可靠的规模趋势或统计置信区间，所以不添加趋势图。
