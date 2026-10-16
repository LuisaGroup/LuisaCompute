# 2026-09-10：SIMD native pointwise fusion 固定候选复验

## 结论与证据边界

这次检验的是现有通用 guarded pointwise fusion 的收益和代价，不是新写一个
RoPE/SwiGLU 专用 lowering，也不是让 solver 自动选择融合。**完整 24-case
矩阵与四 case pilot 均完成并通过独立审计，收益显著依赖尺寸；默认开关不变。**
不能由“热路径少了拷贝”推出“所有尺寸都更快”。

完整矩阵覆盖六族、四种尺寸：17×65、129×768、257×1538、1024×4097；
RoPE 的两个奇数宽度调整为 66、4098。全部 432 个 native visits 成功；
[完整时间、配对范围及胜轮表](tables.md) 保留每个 case。主要结论是：

- RoPE 的三个较大尺寸 On/off 为 0.527–0.695，全部 18 轮改善。
  257×1538、1024×4098 的 On/Inductor 配对中位数为 0.943、0.880，
  分别胜 5/6、6/6 轮；129×768 仍慢于 Inductor，不能概括为 RoPE 全面领先。
- 小 RoPE 17×66 明显回退至 off 的 1.702 倍；小 LayerNorm 17×65 回退至
  1.457 倍，且由 off 胜 Inductor 变为 on 输。两者六轮均回退。
  LayerNorm 257×1538 也有 1.012 倍的轻微回退，六轮均保留。
- SwiGLU 四种尺寸全部 24 轮改善，On/off 中位数为 0.883–0.943，但
  On/Inductor 仍为 1.032–1.172，全部 24 轮仍输给 Inductor。
- RMSNorm 与 masked softmax 共八个 off/on 对象完全相同，是控制项，
  不把时间波动当作 fusion 收益。RMSNorm 和 GELU+residual 的 on 均胜
  Inductor 全部 48 轮；masked softmax 仍为 Inductor 的 1.344–1.638 倍。

完整矩阵与先完成的四 case pilot 保持独立，不合并样本、不相互替换。
例如大 RoPE 的 pilot On/off 为 0.627，完整矩阵为 0.695；保留这一区别，
不择取更漂亮的数字当作同一实验结果。

测量使用 `202955f7d` 加一项正确性测试的源码基线，没有本轮生产 compiler 改动。
当前 off/on 均为 LLVM22；9 月 9 日留档的 off 是 LLVM21，且实际 LLVM/object
身份均不同。因此，本次收益必须取同一轮 off/on 比较，不能用旧报告的绝对时间
拼接出新 speedup。冻结的 Inductor、输入和共同 C++ timer 可复用，Tile 编译器
身份不能因此视为相同。

这不是默认路径、自动 cost-model 优化或无后台负载的性能验收，不用于成本拟合。
也没有新测 Runtime E2E、GPU、Metal/TIRx、MPP、MPS、GEMM 或 attention。

## 固定实验合同

- Apple M1 Max，FP32，单 CPU worker；packet W8、local=8、block=32 固定。
- full-packet specialization、predicated effects、cohort-private access 开启；
  load/reduction fusion、expression/reduction fusion、map fusion、fast math 关闭。
  唯一 A/B 开关为 pointwise fusion。
- actual ORC 导出的对象被链接为 native entry，和冻结 TorchInductor 2.14.0
  entry 由相同 C++ callback timer 调用。helper 不实现算子算法。
- 每个 case 使用 off/on/Inductor 的全部六种排列；每次 visit 100 ms warmup、
  七个自适应重复样本，每样本目标 30 ms。表中时间是六个 visit 中位数的中位数；
  配对比值是六个同轮比值的中位数，不是显示时间相除。
- 指标为 `single_thread_native_entry_host_wall_us`：计入 native entry、必要的
  block traversal/reset、编译器生成的 libc 调用和内部 allocation；不计入
  Runtime dispatch、Python、JIT 和调用方 allocation。它不是硬件周期计数。
- payload 64-byte 对齐，固定输入；每次调用检查完整 FP64 oracle
  (`atol=rtol=5e-5`)、输入不变及 allocation guards。数值通过不等价于全定义域
  精度证明，不要求 Tile 与 Inductor 在所有 reduction/math 实现上 bitwise 相同。
- `host_before` / `host_after` 保留桌面进程与负载观察。记录并没有证明机器安静，
  六种访问顺序也不能完全消除后台干扰；范围仅是描述性统计，不是置信区间。

完整矩阵计时前后一分钟 load average 为 9.120 / 9.731；pilot 为
12.074 / 9.818。这只是桌面负载观察，不是对后台干扰的校正或每轮热状态证明。
八个同对象控制项的 On/off 配对中位数也在 0.993–1.015 之间波动。

## 完整矩阵审计

capture、replay 和独立审计均已完成，而不是用部分成功的矩阵发布结论。
24 cases × 3 variants × 6 orders = 432 visits；每个 visit 检查完整输出、
输入不变和 guards。独立 `matrix-audit.json` 在原始制品处重新读取 72 个
native output snapshots，以原输入重新计算 FP64 oracle，复算全部统计，
检查 1,413 个 artifact/runner 身份条目，并拒绝 11 类证据篡改。
整理本报告时还从逐轮结果复算了 24×3 种配对中位数与胜轮，全部相符。
24 项 off/on 输出均 bitwise 相等；72 个 native snapshots 对 FP64 oracle 的
最大绝对误差为 `1.0828684917640885e-6`。这是这些有限输入的结果，不升级为
所有数据或所有可配置 reduction 顺序的 bitwise 保证。

`off_on_object_identical` 的八项为四种尺寸的 RMSNorm、masked softmax。
本次所有 24 项的 off 对象均不同于旧 LLVM21 archive，所以旧表不充当本次
on/off 对照。原始未舍入统计与哈希保留在审计和 capture/replay records 中。
On 共 13/24 个 case 的配对中位数胜 Inductor、77/144 胜轮；这不是逐 case
选择较快开关后的分数，也不是自动 planner 的成功率。

附带的正确性回归由主线在计时之外运行：Runtime 19 tests / 5,086,319 assertions、
LLM 6 tests / 2,370,853 assertions，两个 CTest 均通过，总计 75.13 s。
新加的 RoPE alias focused case 在 W8/W16 下各有 422 assertions；每次另有
18 个不匹配 filter 的 groups 跳过，不能把 focused 运行描述成整个 suite 通过。
构建、CTest 和 focused 日志随 `validation.tar.gz` 保留；不把这些正确性结果
当作计时样本或 performance qualification。

## 四 case pilot：负结果必须保留

| Case | Off µs | On µs | Inductor µs | On/off | 六轮范围 | On/off 胜轮 | On/Inductor |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoPE 17×66 | 0.147 | 0.252 | 0.115 | 1.708 | 1.690–1.716 | 0/6 | 2.182 |
| RoPE 1024×4098 | 1461.131 | 988.435 | 1117.134 | 0.627 | 0.560–0.754 | 6/6 | 0.842 |
| SwiGLU 17×65 | 1.659 | 1.569 | 1.330 | 0.941 | 0.921–1.000 | 5/6 | 1.171 |
| SwiGLU 1024×4097 | 6039.880 | 5265.664 | 5013.831 | 0.881 | 0.795–0.951 | 6/6 | 1.054 |

小 SwiGLU 唯一变慢轮的精确比值是 1.000312，不能因表格舍入写成全胜。
大 RoPE 的 On/Inductor 六轮均小于 1；小 RoPE 与两个 SwiGLU 的六轮均大于 1。
pilot 共 72 个 native visits，完整输出、输入和 guard 检查均通过。
原始 `pilot-audit.json` 只拒绝七类证据篡改，原样保留；之后的
`pilot-audit-v2.json` 重新读取 12 个 native outputs，重新计算 FP64 oracle、
统计，检查 253 个 artifact/runner 身份条目，并拒绝 11 类证据篡改。
这是对原 pilot 的重审，不是重新计时；没有把新版审计能力倒填进旧文件。
两个 cohort 的 guards 仅有运行时检查记录，释放后的 guard storage 未留档，
不能称为事后重新读取过 guards。

## 留档索引与可复验边界

- 原始 capture/replay JSON 逐字节 gzip 留档，不重写其中路径或原 SHA：
  [pilot capture](pilot-capture.json.gz)、[pilot replay](pilot-replay.json.gz)、
  [matrix capture](matrix-capture.json.gz)、[matrix replay](matrix-replay.json.gz)。
- 独立审计：[原 pilot 审计](pilot-audit.json)、
  [pilot 重审](pilot-audit-v2.json)、[matrix 审计](matrix-audit.json)。
  原审计被新版补充，不伪装为同一份结果。
- 实际 native entry、ORC object、LLVM、反汇编及对应命令保存在
  [pilot native 制品](pilot-native.tar.gz) 和 [matrix native 制品](matrix-native.tar.gz)；
  构建/测试记录见 [validation](validation.tar.gz)。压缩包成员为相对路径。
- [provenance](provenance.json) 与 [inventory](inventory.json) 记录归档身份；
  `runners/` 保存计时、审计和验证工具的源码快照。

**本 Git checkpoint 不含 `.f32` / `.f64` 张量 payload，也没有另做持久张量
归档，因此不是可脱离原目录直接重跑输出审计的完整数据包。**审计确实在当时
读取过原输入与输出；仓库可追溯其记录和哈希，但不能仅从记录重新读取未归档的
数值。原始临时目录是 `/tmp/luisa-pointwise-native.WEtcvZ/pilot`、
`/tmp/luisa-pointwise-native.WEtcvZ/pilot-replay`、
`/tmp/luisa-pointwise-native.WEtcvZ/matrix`、
`/tmp/luisa-pointwise-native.WEtcvZ/matrix-replay`（macOS `/private/tmp` 为同一位置）。
这些目录不保证长期保留。输入 fixture 可由代码重生成，但相同输入字节需要重新
hash 验证；未来运行必须产生新 capture/replay/audit，不能替换此次输出或冒充
此次原始运行。实际 entry/object/LLVM 的字节则包含在本次持久归档中。

## 实际生成物：已知事实与待证伪解释

文档另经 Sphinx `-W --keep-going` 构建及 `check_docs.py` 检查：72 个
HTML 页面、5,430 个本地链接/资源、199 个兼容锚点全部通过。桌面 1440 px
与手机 390 px 渲染已检查；手机表格使用横向滚动，正文无横向溢出。
[文档验证日志与截图](docs-validation.tar.gz) 单独留档，不属于原性能 evidence
inventory，也不改变原 capture/replay 的身份与计时。

证据来自本轮 pilot 的 `assembly.stdout.log`，不是旧 LLVM21 静态输出。
下列 offset 均相对对应反汇编符号的起点；full-packet fast path 与 alias fallback
必须分开读，不能仅搜索整个对象是否仍有 `memcpy`。

- **大 RoPE：**融合 helper 的 `+0x110…+0x188` 是四组输入向量 load、两组
  output store 的单循环；fast path 不再先做 snapshot copy。`+0x1dc` 起的
  alias fallback 仍有四次拷贝。helper 仍预留 33,056 B frame，不能说快路径
  删除拷贝等于去掉了整个保守 workspace。
- **小 RoPE：**融合 helper 的 `+0xe0…+0x2c4`（共享尾 store 在 `+0x4f8`）
  分组直接读四输入、写两输出，没有旧 fast path 的数据 spill/reload；fallback
  自 `+0x2c8` 起仍会 spill。helper 未 inline；外层 entry frame 从 1,904 B
  变为 2,368 B。热路径内存访问减少与调用/guard/code-size 代价可以同时存在。
- **SwiGLU：**融合 fast path 是条件分支的跳转目标，不是 fallthrough：大 case
  自 `+0x49c`、小 case 自 `+0x4e0` 起。核心循环直接读取原输入，不再调用
  snapshot-copy；向量 exp/div 仍在，fallback 的两次拷贝仍在。不能把剩余差距
  无证据地归因为“exp 慢”。

整个保留对象的 AArch64 指令字节数如下（反汇编指令数 × 4；包含 fast/fallback
和 partial-packet 路径，不含对象元数据或数据池；不是 hot instruction footprint）：

| Pilot case | Off 指令 B | On 指令 B |
|---|---:|---:|
| RoPE 17×66 | 9,968 | 14,500 |
| RoPE 1024×4098 | 6,028 | 7,720 |
| SwiGLU 17×65 | 4,420 | 7,572 |
| SwiGLU 1024×4097 | 4,376 | 7,532 |

小 RoPE 回退与 helper 未 inline / guard 和代码体积增长同时出现；这只是可证伪
的盈利性假设，不是采样 profile 或周期归因。后续应在固定 IR/math/资源语义下
分别改变 helper inlining、guard 提升和 fallback 布局，再考察多算子 holdouts。
不能据此添加按算子名或特定尺寸强行开启/关闭的规则。

## 如何落实到通用 planner

本次只切换已有 realization。共享 DAG、多个同 ownership 输出、效果与别名边界
是合法性依据；“某个算子叫 RoPE”不是依据。将 fusion 纳入联合候选仍需比较：

1. 消除的 load/store traversal、snapshot 字节与共享表达式计算；
2. 引入的动态 alias guards、helper 调用、代码增长与 conservative fallback frame；
3. packet/full-tail 比例、执行几何和后端 codegen 对上述选择的实际响应。

资源可行性与盈利性是两层问题。静态 frame 或 allocation 不是精确寄存器压力、
峰值 live workspace 或耗时；本轮未拟合这些权重，也未证明自动选择可以重现
人工固定开关的最佳结果。完整矩阵保留未改变对象的控制项，避免把噪声算作优化。
