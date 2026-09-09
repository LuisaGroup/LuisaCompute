# next 集成与 native replay 验证

这个检查点将上游集成与性能实验分开记录。`next` 已合并，合并后的完整配置构建通过；
合并前两轮 native 对照均捕获到其它任务的渲染活动，因此没有新增可采纳的性能结论，
不改变当前排名、pointwise fusion 默认值或 planner/cost model 参数。

## 源码与依赖边界

- 上游：`8911828eb234a37d1059bb9b01e82d2915624f80`。
- 合并提交：`360d9791e344da06bcbde4dc871526aa336900b9`；保留两个父提交，没有 squash 或重写历史。
- 隔离副本从合并提交及递归固定的子模块提交导出，共 19 个仓库、18,830 个文件，逐文件记录 SHA-256。
  它没有带入工作区中尚未完成的 TIRx matrix 实验，也不复用旧的 source-overlay 构建目录。
  随后仅添加一处 Metal 测试入口参数检查修复；最终源码是固定基线加此显式 overlay，
  并非仍与初始 snapshot 完全相同。生产编译器、库和 Tile 源码未因这个修复改变。
- 工作区原有 11 个未提交文件的内容哈希及全部子模块 checkout 保持不变。
  因上游推进了 gitlink，`imgui` 和 `magic_enum` 的原 checkout 相对新索引会显示差异；没有擅自切换它们。
- 首次导出缺失 SPIRV-Tools 的固定提交对象；从配置的 origin 补取对象后重新导出，原失败日志保留。
  未改动该子模块的本地 HEAD。

配置使用 RelWithDebInfo、Apple C++ 编译器、LLVM 21.1.8、SIMD + Metal + TIRx；Metal4 等其它设备后端关闭。
外部 TVM 继续使用记录中的已有本地库，归档其哈希；“递归固定依赖”指 Git 子模块，不表示重建了 TVM 或系统工具链。
`configure-command.json`、`source-snapshot.json.gz` 和 `verification.json` 分别记录配置、初始源码和最终验证结果。
最终验证记录包含 overlay 哈希，并引用完整保留的两份早期失败记录。
这些构建/正确性检查不能证明合并前后性能不变，也不是未启用后端的验证。

完整构建后，35 个 Tile CTests 全部通过，包括 CPU/Metal 的 TIRx、native Runtime 和 XIR/SIMD 路径。
此外，79 个 XIR/SIMD CTests、两个 Metal 回归测试，以及额外的 Runtime A/B 和融合 LLM 配置均通过。
Tile 与 XIR/SIMD 两组在测试入口修复前完成；修复后重新完整构建，再运行 Metal 和相关配置复核。
不能把这些不同阶段、存在重复测试的计数直接相加，声称一次完整 CTest 全集通过。

## 失败原因与复核边界

- 新上游 Metal 测试在没有第三个参数时崩溃。LLDB 栈定位到 `string_view(argv[2])`：
  Boost.UT 重载的比较/逻辑运算让原本预期的短路失效。拆成两层条件，不改编译器、
  不删测试或断言；完整测试及 `--local-only`（780 条断言）均通过。
- 我们的测试 supervisor 最初强制打开 pointwise fusion，覆盖了 Runtime 测试自己构造的
  “关/开”对照，导致 metadata 断言失败。第二次已去掉强制打开，但仍强制关闭 load/reduction
  fusion，干扰了另一个 A/B fixture。最终调用清除继承的 SIMD 环境，并不强制这两类融合；
  W8、full-packet、predicated memory 与 cohort-private 配置保留，完整 Runtime 测试通过。
  原失败不是数值断言，未通过放宽断言来消除。
- 改动行的格式检查通过，整个翻译单元语法检查 0 错误、1 个原有 unused-include warning。
  整文件格式检查仍有 7 处上游已有诊断；与 pristine 合并提交逐位置核对一致。
  不为此重排无关代码，也不把全文件格式检查标成通过。
- 第一次严格 Sphinx 构建拒绝了上游新增的 13 份未接入目录树的验证记录。
  现将它们统一归属到已有 performance/validation 导航，保留原路径、内容与边界说明，
  不关闭目录归属校验，也不将历史跨设备结果宣称为本次重测。

`verification-initial.json`、`verification-recheck.json`、原始 JUnit、调试栈和 supervisor 源码
保留失败过程；`verification.json` 是有明确范围的最终成功记录，不覆盖原始失败。

## 两轮完整计时均不进入成绩

被测 native 代码来自合并前的 `67330a633ee7901034ab4a9360d9cd15be904b1f`，
不是本次合并后的新二进制。实际 LLVM、ORC object 和 native dylib 沿用
`../m1-max-20260909-xir-pointwise/` 的已冻结检查点；Inductor 和共同 C++ timer 的来源不变。

六类算子、四档尺寸，固定 FP32 / W8 / local=8 / block=32 / 单 CPU 线程，
比较融合关、融合开与 Inductor，覆盖全部六种排列；每次七个样本，100 ms 预热，30 ms 自适应批次目标。
计时仍为实际 native 入口的 host-wall 时间，排除 Runtime/Python/JIT/caller 分配，
保留必要的入口遍历、launch reset 和生成代码内部的调用。它不是硬件周期计数或 E2E 延迟。

第一轮启动后出现新的外部渲染；第二轮增加一分钟无构建/渲染的预检，
预检因新活动重置，达到条件后才启动，但计时中又出现渲染。
两轮都完整保留并整体排除，不挑选“看起来干净”的算子、轮次或更好的比值来更新性能结论。
五秒一次的进程观察不能排除采样间的短活动，更不是独占硬件或固定频率的证明。
公开记录保留匹配进程的名称及数字观测，省略其它应用清单和其它工作区路径；完整原始观察留在本地。

两轮共 864 次计时访问（每次包含重复的 native 调用），每次访问后检查完整输出、输入不变和保护区。
独立审计重新读取 144 份输出，48 个 on/off case 均逐位相同，并拒绝八类内存中的证据/输出篡改。
这些是新增正确性证据，不是速度提升、默认策略收益或全输入数值等价证明。
FP64 容差、有限输入集合、LayerNorm/reduction/math 实现差异沿用前一检查点。
临时保护区在执行中检查，没有保存供事后重新读取。

## 对优化工作的约束

同域共享 DAG 的 streaming 路径已存在，但 LLVM 最终的内联、回退栈帧和尾部处理仍影响实现成本。
例如已归档的小尺寸 RoPE 机器码在融合后保留独立 helper，而关闭融合的对象没有该独立函数边界。
这是静态代码证据，不是本轮已验证的性能归因。

后续应将数据搬运节省与 guard、fallback、活跃值、尾部 CFG、调用/内联一起作为 realization 的成本特征，
再用合格的完整对照及未参与调参的尺寸/算子验证策略。不能根据这里被排除的时间样本拟合模型，
也不能因为融合合法便默认它更快。本检查点没有实现或宣称新的成本校准。

原始张量保留在 `provenance.json` 指定的本地目录，Git 保存哈希、审计、运行源码及日志。
所有新证据继续由既有 `docs/source/performance/tile/` 文档索引，不另建一套报告入口。
