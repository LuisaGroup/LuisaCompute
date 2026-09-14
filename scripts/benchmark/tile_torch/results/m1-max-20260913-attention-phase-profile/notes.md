# Attention 原生采样：K/V 快照搬运比 MMA 指令更值得先优化

2026-09-13，Apple M1 Max，macOS 26.6.2，LLVM 22.1.8。

## 结论

这次没有新加速比，也没有修改生产默认。复用[上一轮 native MMA 实验](../m1-max-20260913-attention-native-mma/notes.md)的实际编译器产物，采样 MHA decode、长 KV decode 和 batch GQA prefill 的 off/on 两臂，六次都通过前后完整输出、guard、输入不可变及本臂 capture 逐 bit 检查。

**MHA-on 的主要采样热点是 K/V 定义快照搬运，不是未内联的 MMA 调用。** 最终统计中，5574个原始 Timer Fired 样本里，2381个落在 K 拷贝循环、2349个落在 V 拷贝循环，合计84.9%；长KV-on的对应两段合计81.5%。这是采样位置占比，不是可消除时间比例，也不是DRAM流量/带宽计数。

最初仅关联派生表的MHA计数为K2380、V2347，84.8%；最终纳入4个没有派生行但有原始PC的样本后，K增加1、V增加2，另1个仍在未分类kernel区域。初版统计保留，不补造它们缺失的派生权重。

因此优先实现通用的连续 snapshot-transfer 候选，再组合 QK 的输出寄存器分组。选择依据应是域、stride、mask、物理布局及相邻 phase，不应匹配 attention 名称。**目前只完成定位和设计依据；MPS/Torch/BLAS 的总体目标仍未达成。**

## 样本与原有性能结果分开

| 采样 case | B / Hq / Hkv / Q / KV / D / Dv | query×KV block | 采样臂 | 前后校验 |
|---|---|---|---|---|
| decode-mha-d64 | 1 / 8 / 8 / 1 / 2048 / 64 / 64 | 1×16 | off、on | 均通过 |
| decode-long-kv | 1 / 16 / 4 / 1 / 8193 / 128 / 128 | 1×16 | off、on | 均通过 |
| batch-gqa-q4 | 2 / 6 / 2 / 17 / 67 / 40 / 48 | 4×16 | off、on | 均通过 |

它们分别代表上一轮普通负结果、长KV负结果、唯一正结果，不是按本轮profile挑出的最快样本。上轮六配置的未插桩ABBA数据保持不变；本轮三配置不能替代其余三配置，也不构成全形状覆盖。没有重新测试GPU队列或声称Metal恢复。

| case / arm | 原始 Timer Fired | 有派生表权重 | kernel映像叶PC | 其他叶PC |
|---|---:|---:|---:|---:|
| MHA off | 5394 | 5390 | 5393 | 1 |
| MHA on | 5574 | 5570 | 5571 | 3 |
| 长KV off | 5400 | 5396 | 5356 | 44 |
| 长KV on | 5409 | 5406 | 5397 | 12 |
| batch GQA off | 5408 | 5404 | 5407 | 1 |
| batch GQA on | 5399 | 5395 | 5396 | 3 |

合计32584个定时样本、32561个精确关联行，23个缺失关联全部保留原始PC；每次另有2个Stackshot，均不加入定时样本分母。六次匹配行中的派生frame均比原始PC大1，原始PC均四字节对齐。不同run的样本数差异不是速度比较；未匹配行没有伪造weight。

每次复用冻结的 `kernel.o → kernel.dylib` 和 `replay.dylib`，不从LLVM源码重新编译。native MMA off/on仍为0/4，program packet仍为W8；block32、local1、R4及各case的cap/2D request均来自原manifest。这里的两个向量宽度不是同一个概念。

## 采样方法与地址身份

- 子进程先核对manifest、输入/输出和实际dylib哈希，完成一次native调用和全量FP64/guard检查，然后发出READY。父进程只对这个自己启动的PID采样，不采样其他用户进程。
- GO之后在原C++ replay helper里执行10秒warmup，再完成独立后验检查；通过 `dladdr` 记录实际kernel/helper路径、加载基址和入口地址。完整FP64 reference沿用上轮独立 `einsum(optimize=False)` 交叉核对过的oracle；本轮再次对完整输出逐元素检查，不宣称重新证明了所有浮点输入的等价性。
- Time Profiler附着5秒；启动、附着、保存有额外时间，命令墙钟不是有效采样窗。PROFILE_BEGIN也不是精确warmup边界，helper内部还有分配、预校验和校准。用原始样本时间和实际kernel映像归属过滤，不把所有进程时间叫作纯kernel时间。
- 普通 `/usr/bin/sample` 的MHA-off pilot成功，但把函数内多个PC合并，无法恢复频率；保留为分辨率不足的pilot，不伪装成数值Error。后加的Time Profiler是独立协议，不继承pilot的显式1ms/启动延迟选项。
- 原始 `time-sample` 的 Timer Fired 才是计数单位；Stackshot单列。`time-profile`通过精确时间戳和目标线程关联，使用其显式weight，缺失/重复不得补造。MHA-on中派生frame地址比对应原始指令PC大1；真正的反汇编匹配使用原始对齐PC，不猜测减1或四舍五入。
- `PC − dladdr base` 是映像相对地址。先核对Mach-O可执行节，再映射到实际linked指令；不能把它直接当ORC object offset。MHA-on函数在linked image的0x350，wrapper在0x1d690。保存的对象/映像反汇编和链接重定位差异支持该对应关系。

MHA-on的很多栈在kernel后出现无映像的小地址，native helper/wrapper回溯不完整。**叶PC归属可以使用，缺少kernel-rooted libc样本不意味着没有复制、分配或调用成本。** 采样扰动、skid、桌面活动和单次有限窗口也不允许从某条load的样本数推导该load的延迟或cache miss率。

## 代码证据与通用优化方向

MHA-on的两个half-open linked-image区间是：K `[0x9c10,0x9e08)`，V `[0x9e50,0xa01c)`。源基址分别来自kernel参数K、V；每个active program各拷贝1024个FP32元素到定义时快照（循环1024次，每次至多8个program lane）。实际机器码含逐program的 `tbnz` mask分支与 `ld1.s/st1.s` 标量访问，而native MMA随后需要的是每program连续布局。没有把大段未分类代码强行算成softmax或spill。

长KV-on的实际函数 `_llm_attention.full_packet` 中，K `[0x13c4,0x14d8)`、V `[0x14e8,0x15fc)` 各为每个program遍历2048元素，分别命中2221和2189个原始样本（合计4410/5409，81.5%）。满包特化去掉了这些copy的lane mask，但仍是element→program的标量访问顺序。这是不同代码形态中的结构佐证，不应把连续copy优化和满包特化视为互相替代。

batch GQA的跨块mask控制流尚未建立完整阶段映射，保持unclassified，不发布其copy/QK/PV占比。

```text
逻辑视图       (program p, tile element e)
                       │
当前搬运       e 外层 → p packet 内层 → masked scalar gather/scatter
                       │
定义时快照     snapshot[p][e]（每 program 内 e 连续）
                       │
native MMA     QK 沿 D 向量化；PV 沿 Dv 向量化

待实现候选     在同一 load 定义点：p 外层 → 连续 e 向量块
                       └─ 保留 bounds / inactive program / 尾块处理
```

这直接支持execution-first的设计：资源和SSA快照语义不变，搬运与计算可以选择不同的执行方向。连续copy、原packet-interleaved布局、逐program向量MMA各有适用条件，应联合计入转换、mask、live state和代码量；只给MMA向量指令一个奖励系数会漏掉这里的大部分工作。

下阶段先做封闭的 `view-load → fresh private snapshot` 连续片段准入，非volatile、类型一致，其他layout保留原fallback。读取仍在定义点完成，不能延迟至MMA消费，不能通过用户buffer越界宽读来消除mask，也不能给互相别名的输入错误添加noalias。之后测试普通GEMV/strided contraction、ragged输入和不同attention尺寸；未采样的ABBA完整kernel时间才用于判断收益。详细语义与cost分解见[设计记录第16节](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md#16-原生采样先优化搬运的执行映射而不只优化-mma)。

## 归档边界

只公开本任务的命令/校验收据、目标线程采样表、对应机器码和校验输出。完整Instruments trace、含进程环境的TOC、原始sample系统报告不入库；原始文件在本机保留。导出的目标采样表、派生统计和机器码区间各自保留哈希，不能把脱敏派生数据称作原始trace。原始目标表保留本次PID/ASLR地址和已审核的本任务、Python及系统动态库路径，不含继承环境或其他进程。

`evidence.tar.xz` 携带六臂manifest、linked image/object/LLVM、helper及capture/reference输出，能够离线重新提取原始PC、核对映像、重算六组统计和三组分类。完整输入tensor与623成员编译器源码复用上轮已提交归档，依赖及SHA写在包内 `dependency/native-artifact.json`；本包不是独立运行全部native数学验证所需的完整依赖闭包。

在本报告目录运行以下命令，只做安全解包、哈希检查、离线解析重算与6项纯Python合成测试；不会加载native代码、启动profiler或测试性能：

```sh
shasum -a 256 -c SHA256SUMS
python3 -B verify_profile.py .
```

完整导出表解析、相对PC及阶段统计应与归档逐字节一致。缺少TOC导出导致的五次初期解析失败、MHA初版matched-only统计和机器分析早期区间修正均保留，不能改记为native数值失败或删除后假装只有成功尝试。

本轮未改C++，未重新构建或重复宣称全仓测试通过。实际镜像来自上轮已通过完整选定构建和11项CTest的隔离导出，并非混入受保护TIRx WIP的主树全量重建。新代码只用于采样、离线解析及归档，native输出检查均复用原C ABI。
