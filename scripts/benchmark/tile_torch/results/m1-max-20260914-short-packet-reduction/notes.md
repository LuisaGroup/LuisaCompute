# XIR 短域协作规约：实现与验证

日期：2026-09-14；设备：Apple M1 Max。此记录不替代多路径性能排行榜，也不表示完整 attention 映射或 MPS/Torch 性能目标已完成。

## 本次改变了什么

旧公共轴 realization 不接收 `1 < N < W` 的协作分布。现在短轴使用每 lane 一个带有效位的 local slot，只有实际 owner 计算贡献；全组汇合后交换 payload 与有效位，再广播 canonical root，用户 init 只合并一次。合法 bounds fill 仍是贡献，不能和逻辑域外的 lane 混淆。长度1保持复制求值与 leader store；已有完整 packet 的计算顺序不变。

这是 bridge 的通用 realization 改进，不匹配 RMSNorm、softmax 或 attention 名称。资源分析与 emitter 共用短轴决策，planner 加入有效位通信的相对工作量；尚不是校准过的硬件周期模型。

`program_team.h` 同时提供逐值 Replicated/Cyclic 几何及 checked owner/slot 映射，作为后续多轴分布基础。**它尚未接入 whole-program SSA/carry/通信调度。** 当前 attention 的 pipeline/MMA 仍不能因此自动使用不同轴的协作分布。设计背景见 [attention 映射审查](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md)。

## 验证结果与失败边界

| 检查 | 最终结果 | 限制 |
|---|---|---|
| 完整选定构建 | 通过 | pinned 基线加10个明确 overlay，不含其他工作树改动 |
| target info / resources | 21项，24,580次断言通过 | 包括W32/W64、短轴存储、实际XIR Alloca与有效位shuffle |
| program-team几何 | 7项，338,525次断言通过 | 几何正确不等于完整emitter已接入 |
| 原XIR host测试逐例执行 | 18通过，2个既有失败 | 两例在未修改基线同样SIGABRT，未删除或放宽断言 |
| SIMD runtime | 26项，5,195,002次断言通过 | 使用分离LLVM版本的测试配置 |
| Metal4六套逐例执行 | 6项，238,189次断言通过 | 包括五种行算子、短/长规约、RoPE、资源准入；不是整套CTest计时通过 |

短规约检查覆盖长度1/2/7/16/31/32/33/65、非identity初值、signed zero、输入覆盖后的旧快照及guarded BufferView。SIMD另保留4096/16384长域。`logical width=7 / physical width=1` 检查要求填充后乘积为0，`sum(init=2.5, x+3)=24.5`。

保留的异常及处理：

- 原Metal整套CTest在240秒总预算处超时，不能标通过或据此推定GPU hang。最终逐套执行不减少任何子用例，使用每套600秒上限。每个过滤进程显示6个注册项，其中仅1个执行、5个过滤；表中没有将过滤项累计为通过。
- 两个host失败分别是 `tile_xir_large_tiles_have_bounded_code_and_eager_load_snapshots` 与 `tile_xir_expansion_budget_is_fail_closed`。候选/基线均报告 `XIR realization exceeds its static SSA expansion budget; choose smaller Tiles` 后SIGABRT。可恢复错误契约仍有缺口。
- 初始SIMD进程同时链接LLVM21的TVM与LLVM22的SIMD，基线和候选均在LLVM AnalysisManager崩溃。使用已有 `LUISA_COMPUTE_TILE_XIR_TEST_TIRX_COMPARISON=OFF`，完整重建后26项通过；独立TIRx bridge仍开启。这支持链接冲突诊断，但没有证明具体弱符号的绑定归属。
- 三次Metal汇总脚本失败均保留；native进程退出0，失败来自成功行前缀/ANSI边界解析。最终v4先对全部六份实际日志做离线检查，再完整重跑六套。未改C++或数值容差来消除这些解析错误。

## 性能口径

编译耗时和上表测试耗时不是kernel性能。本次固定block256，比较同一新版二进制的local1/local32映射，分别观察RMSNorm与masked softmax的4096×7/31/65；这不是旧版对新版或MPS/Torch的直接比较。

吞吐/延迟各保留插桩dispatch区间、无计数器command-buffer GPU时间及同步host wall三种指标。不同口径不混排；插桩区间也不称零开销“纯kernel时间”。

48个visit已完整执行，输出/guards通过，运行前后源码与产物指纹不变。每个case两个ABBA序列，共4个不重叠相邻配对；每visit每口径3个样本、吞吐16次dispatch。下表是配对visit中位时间之比的中位数，统一为 **local32/local1，小于1表示协作更快**。

| 算子，4096行 | 吞吐：插桩dispatch | 吞吐：无计数器CB | 吞吐：host wall |
|---|---:|---:|---:|
| RMSNorm ×7 | 2.041 | 3.268 | 0.865 |
| RMSNorm ×31 | 0.971 | 0.747 | 1.805 |
| RMSNorm ×65 | 0.343 | 0.337 | 1.549 |
| masked softmax ×7 | 2.406 | 1.421 | 1.345 |
| masked softmax ×31 | 0.822 | 0.957 | 1.741 |
| masked softmax ×65 | 0.197 | 0.383 | 0.789 |

| 算子，4096行 | 单调用：插桩dispatch | 单调用：无计数器CB | 单调用：host wall |
|---|---:|---:|---:|
| RMSNorm ×7 | 2.718 | 1.891 | 0.680 |
| RMSNorm ×31 | 1.232 | 1.257 | 1.315 |
| RMSNorm ×65 | 0.465 | 0.287 | 0.890 |
| masked softmax ×7 | 1.726 | 1.758 | 1.165 |
| masked softmax ×31 | 0.631 | 0.634 | 3.186 |
| masked softmax ×65 | 0.171 | 0.597 | 1.461 |

这只是有限样本的描述性诊断，不是稳定排名或统计显著性证明。完整36组配对范围和864个原始时间样本在归档的 `raw/probe-pairs.json`、`raw/probe-results.json` 及各visit文件中；没有删掉离群点。例如RMSNorm×65的插桩吞吐比值范围为0.311–0.349，host吞吐却为1.217–1.971；其无计数器CB范围0.309–1.337，也有反转。masked softmax×31的host吞吐比值范围0.593–13.483，不能仅报中位数为稳定回退。

## 对planner的启发

- **短域支持不等于短域收益。** 宽度7的两类算子在所有4个插桩吞吐配对中都回退。有效位通信和每program占用整组的代价必须保留；不能按“工作量除以32”直接优惠。
- 宽度31接近交叉区，收益依赖算子与目标口径。宽度65的GPU协作收益明显，但这是本来已支持的对照路径，**不是这次新增代码相对旧版的提速**。
- 固定block256仍会使线程组数从16变成512，因为根program数固定而每program从1变成32个worker。这正是执行资源映射的变化，不能误述为仅替换了一个intrinsic。
- 后续可评估一个物理packet容纳多个较小logical teams，而不是强制每program占完整packet；必须明确子team的shuffle索引、参与mask和尾部契约后再准入。另需接通每个SSA值的分布与跨phase通信，并以资源受限的完成时间建模。它们目前是候选设计，不是已实现/已校准能力。

因此，本轮支持“规约映射存在结构性限制”的判断，但未证明所有Metal差距都来自规约，也未建立MPS/Torch性能优势。

## 证据与复核

独立离线审计重新计算了48个visit、864个原始时间样本、24个相邻配对及36组汇总，并核对192个张量文件与6,750,208个FP64 oracle输出元素。时间由原始tick差、command-buffer区间和host样本重算，不复用已汇总的中位数。运行前后及审计时的2,899个源码/产物/门禁文件指纹一致。

原生执行每次检查完整输出及两侧guard；导出的tensor文件只包含有效区间，因此离线审计只能核对guard检查收据，不能重读未导出的guard字节。不会将这一点写成独立guard内存验证。

`evidence.tar.xz` 保留完整原始收据、失败尝试、审计脚本/结果、四个计时与数学参考helper、192个张量文件及10个最终源码overlay；`package-inventory.json` 提供成员大小与SHA256、基线递归导出身份、最终配置和二进制指纹。它不是自包含的全部依赖构建包。审计入口位于归档的 `raw/audit/audit_probe.py`，结果为 `raw/audit/audit-result.json`；脚本记录原执行环境路径，移机重放需按清单重映射路径。
