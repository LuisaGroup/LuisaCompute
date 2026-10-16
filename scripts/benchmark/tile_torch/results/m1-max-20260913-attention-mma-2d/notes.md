# 通用二维 MMA 输出分块与 attention native-entry 实验

2026-09-13，Apple M1 Max，LLVM22.1.8，precise FP32。基于 `cdd6d51cc`
的工作量修正，扩展 XIR bridge 的可选生成方案。默认仍关闭；本轮不宣称
完成自动 planner 搜索、Metal4 GPU 验证或超过 Torch/MPS。

## 实现与实验变量

requested `mma_output_block=4` 表示最多四个累加器。新选项
`enable_mma_2d_blocking` 允许两个输出轴具有互补操作数广播时，使用
`2×2` 而不是原来的 `1×4`；R1/R2、单行 decode 和不满足准入的布局
保留原方案。规则依据 typed contraction 的轴和访问关系，不识别算子名字。

K 不分割、不重排，原操作数乘法次序及 MUL→ADD 不变；小输出按原
flat index 回填 SSA Elements，较大输出写入原 snapshot。definition-time
别名快照、carry、类型和资源容量不变。前导 batch 与 singleton 不误作分块轴。
共享准入同时供 lowering、资源分析与工作提取使用；成本接口保留独立的
lhs/rhs 投影及循环计数，没有增加统一的经验加速系数。

固定 W8、block32、local1、全局 Tile 阈值64、region budget4096、R4、
QK/PV=MMA、full-packet 请求开启；每组固定 cap，仅切换二维选项。
root order、blocks-per-task、snapshot/workspace、ABI、输入与数学 helper
在每个 A/B pair 内完全匹配。cap0 继承全局展开阈值，cap8 提前保留 K 循环。

## 完整六组结果

12 captures、72 ABBA visits、504 samples。每组三轮 ABBA，每次七个样本，
warmup30 ms、target15 ms。时间是**单线程实际 ORC native-entry host-wall**：
包含入口、launch reset、block 遍历和生成的 helper/内部分配，排除 Runtime、
Python、JIT、调用方分配及校验。不是硬件 cycle，也不是 Runtime E2E。

| 配置 | cap | 一维 µs | 二维 µs | 配对 on/off | 六对范围 |
|---|---:|---:|---:|---:|---:|
| prefill-q4-u0 | 0 | 137.557 | 128.236 | 0.932238 | 0.930354–0.944205 |
| prefill-q4-u8 | 8 | 159.915 | 156.251 | 0.977718 | 0.967504–0.983726 |
| heldout-q8-u0 | 0 | 527.324 | 470.696 | 0.894059 | 0.884893–0.897296 |
| heldout-batch-gqa-q4-u8 | 8 | 3353.047 | 3274.755 | 0.977572 | 0.970897–0.979137 |
| heldout-gqa-q8-u8 | 8 | 6001.021 | 5416.505 | 0.902307 | 0.893403–0.906633 |
| decode-m1-u8-control | 8 | 1986.925 | 1983.745 | 0.996147 | 0.987271–1.017823 |

展示时间为 visit 中位数的中位数；比例是每 cycle 两个邻接配对比的中位数，
不是展示时间相除。全部样本保留，范围不是置信区间。机器仍有桌面/系统 CPU
活动；已停止本任务的构建、测试、LSP、归档和其他性能任务，未干预用户应用。
load averages 留档，不声称机器处于完全隔离或静默状态。

shape 顺序为 `B,Hq,Hkv,Q,K,D,Dv`。以下矩阵在采集前冻结，不因结果删改：

| 配置 | shape | query/key block |
|---|---|---|
| prefill-q4-u0 / u8 | 1,4,2,32,65,32,32 | 4×16 |
| heldout-q8-u0 | 1,4,2,64,129,32,48 | 8×16 |
| heldout-batch-gqa-q4-u8 | 2,6,2,17,67,40,48 | 4×16 |
| heldout-gqa-q8-u8 | 1,6,2,33,131,48,40 | 8×16 |
| decode-m1-u8-control | 1,8,2,1,2053,80,96 | 1×16 |

五个 prefill/batch/GQA 配置均实际将 QK 和 PV 改为二维，配对时间减少
约2.2%–10.6%，各自六对均小于1。decode 不准入二维，LLVM 与 ORC object
逐字节相同；其跨1的范围说明不能把约0.4%的中位数差异算成代码优化。
这是有限矩阵的候选证据，不是所有形状或非 attention 算子的性能保证。
同一 prefill 的 cap8 仍明显慢于 cap0 的描述性时间；两种 cap 并未彼此交错，
不能声称二维方案消除了此前的展开/表示退化，更不能将多轮增益相乘。

## 代码生成与成本边界

| 配置 | snapshot B/worker，两边相同 | workspace B，两边相同 | Schedule blocks off/on | 实际满包 clone off/on |
|---|---:|---:|---|---|
| prefill-q4-u0 | 6688 | 0 | 59 / 59 | 0 / 0 |
| prefill-q4-u8 | 6688 | 0 | 110 / 110 | 0 / 0 |
| heldout-q8-u0 | 12720 | 101760 | 98 / 98 | 0 / 0 |
| heldout-batch-gqa-q4-u8 | 9120 | 72960 | 114 / 114 | 0 / 0 |
| heldout-gqa-q8-u8 | 12976 | 103808 | 108 / 108 | 0 / 0 |
| decode-m1-u8-control | 12864 | 102912 | 53 / 53 | 1 / 1 |

满包 clone 由实际 LLVM definition 核对，而非只看请求开关；decode 的
cloned-instructions 计数两边均2787。五组有效二维案例没有新生成满包
路径，故本批收益不是新增 clone 的效果。Schedule block 计数不变也不代表
指令数、寻址或 cache 行为相同；静态容量不等于物理寄存器/spill。

对完整四输出微块，投影总数从每 K 的5次变4次，但行侧读取增加、列侧减少。
实际代价还取决于 stride、representation、索引、代码量和 backend 向量化，
因此不能从20%的投影下降直接预测20%的完整 kernel 加速。
本轮准确计数支持候选评价，但没有自动枚举/选择二维，也未拟合 native cost。
下一步应联合探索输出/贡献维 ownership 与 QK→softmax→PV 的 phase 转换，
保留合法参考方案；微块调整不能代替完整时空映射求解。

## 正确性、构建与证据范围

全部输入、完整输出和独立 dense FP64 bottom-right causal GQA oracle 留档。
独立 NumPy 重算 reference 使用 FP32 scale，核对 expected.f64 为1e-12，
FP32 输出误差界为 `5e-5 + 5e-5×abs(expected)`；A/B 输出逐 bit 一致，
72个重放输出均与对应已校验 capture 的哈希一致。C++ helper 的 guards 与
输入不变检查通过；guard payload 未单独保留，不能把 receipts 冒充独立
重放的完整分配区内存审计。准备阶段只链接实际捕获的 ORC object，不重编 LLVM。

首次完整构建在新增测试缺少 `format` 声明处失败，原日志保留。补头文件并使用
已有类型 predicates 后，完整重建及四项 CTest 通过（142.88 s）；七项精确名称
host 回归均有非零断言。新增96组 plan/lower 配置与60个严格数值 dispatch，
覆盖零K、转置/交换、奇数尾块、预算 fallback、SSA carry、snapshot alias。
60次 dispatch 包括 R1/R2 和 M1 负控制，其中16次实际准入二维方案。
七个变更 TU 的 clangd/tidy 均无错误，warning 原样保留；格式和 diff 检查通过。
两个历史 host SSA-budget fatal 测试仍不在该门禁内，不宣称整仓测试全绿。

原始目录 `/tmp/luisa-attention-mma-2d.HIXgEL`。源码使用独立构建树的实际文件，
十个本轮变更与主树逐 hash 一致；并非声明隔离树每个文件都等于主树 HEAD。
受保护 TIRx WIP 和未合入 upstream 更改未掺入本轮构建。源码快照、配置与
producer/tool fingerprints 不是完整动态加载依赖闭包或 bit-reproducible build
证明。本轮无新的 Metal4/TIRx/Torch/MPS 配对计时；默认选项不变。

## 离线复查

`evidence.tar.xz` 包含原始输入、完整输出、actual ORC objects、准备后的
dylib/helper、命令/样本、验证日志与脚本。`sources.tar.gz` 和 `provenance.json`
另行提供源码/配置快照。源快照在 captures 进行中完成，竞争性 replay 在
两者都完成后才开始；这不是事前保存全部源码的证明。十个变更文件与工具
身份的检查见记录，不能延伸为完整 loader 认证。

原 provenance 沿用了过时的测试状态；原附带 patch 又缺末尾 LF，无法解析。
两项均保留，详见 [补充更正](provenance-corrections.md) 与
[更正 receipt](provenance-correction-receipt.json)。完整归档源码文件不受影响；
证据包内的 `owned-changes-exact.patch` 只补回该 LF，已作只读解析，未实际 apply。
补丁原文保留在包内，不对其必需的 diff 空白上下文作格式化。

复查只需 Python 和 NumPy，不读取当前 build/LLVM，也不会加载或执行 native code：

```sh
shasum -a 256 -c SHA256SUMS
mma_2d_review=$(mktemp -d /tmp/luisa-attention-mma-2d-review.XXXXXX)
tar -xJf evidence.tar.xz -C "$mma_2d_review"
python3 -B "$mma_2d_review/luisa-attention-mma-2d.HIXgEL/audit.py" \
    "$mma_2d_review/luisa-attention-mma-2d.HIXgEL" > "$mma_2d_review/audit-replayed.json"
cmp audit.json "$mma_2d_review/audit-replayed.json"
```

审计核对514个源码成员、644个证据文件、90条命令、24份 capture/prepared
输出、72份 replay 输出及全部504个样本；重新计算独立 FP64 oracle 与全部
配对统计，不删除慢样本或 decode 负控制。它同时检查实际 LLVM clone
definition/call、资源与 root 映射，历史 build/test 结果仍属于保留日志的 receipts。

归档 QA 已从独立新目录解包：7项顶层 SHA256 全部通过，raw/extracted 在审计前后
逐文件相同，提取版脚本输出与冻结 `audit.json` 逐字节一致。证据包5,530,068 B，
源码包2,200,028 B；成员扫描未发现绝对/逃逸路径、重复名或链接。
