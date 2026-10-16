# Top-K / sort 覆盖已经落地，参考实现的性能尚不合格

2026-09-13 首轮筛查完成 **60/60 OK**：6 个 R/N/K 组合 × 升降序 × XIR/SIMD、XIR/Metal4、TIRx/Metal、Torch CPU、Torch MPS。另有 5/5 OK 的链路 smoke。这里的 OK 表示完整数值与计时协议检查通过，不表示速度达标。

例如降序 `topk(128×257, K=16)` 的 SIMD E2E batch 是 6341.646 µs，Torch CPU 是 111.688 µs；`sort(17×1025)` 则为 40925.250 / 184.140 µs。GPU 的同一 sort case，无计数器 command-buffer 区间为 XIR/Metal4 83164.750 µs、TIRx/Metal 121150.833 µs、Torch MPS 103.875 µs。差距非常大，但本轮不用于精确倍数或 planner 校准：只有一个访问顺序，且桌面负载明显。

## 测量与正确性

完整 [分路径表](tables.md) 分开呈现 E2E 与 GPU control。原生和 Torch 都预分配 FP32 values、int64 indices；E2E 排除 JIT、setup allocation/upload，包含 Runtime 或 Python/framework 调用和同步。GPU dispatch、compute-pass 探针与 command-buffer control 保留为不同字段。**没有测纯 CPU native-entry 时间**，也没有把 Python wall time 重命名为 kernel time。

每个 case 每路一轮、3 样本，目标 10 ms、warmup 30 ms；原生超时界限 30 s。SIMD 请求 1 worker、W8，采用当前 complete-program 默认映射。原生每路独立进程，Torch MPS CPU fallback 关闭。输入为 `((column*37 + row*17)%31 - 15)*0.25`，包含负值、零和重复值；每行最多 31 个不同值，不代表任意 logits 分布。

Tile 使用原下标打破同值 tie，Torch sort 也要求稳定顺序；Torch top-k 允许不同的合法并列下标，但必须满足完整值多重集、排序、阈值、唯一性、范围和源值对应。C++ 在首个 dispatch 和全部计时之后检查完整输出、输入不变及三块 buffer 的前后 guards。

独立 [审计](audit.json) 使用 NumPy stable argsort 重读全部 **65 份输入及 65 对输出**，验证 payload hashes、完整 values/indices 与 host 中位数。它不执行 kernel，也没有独立重算 GPU timestamp 协议；guards 是执行期检查记录，导出的 payload 不含 guards。

计时前后一分钟 load average 记录为 11.40 / 8.29（`evidence.tar.gz` 中原始 uptime）。本任务没有同时运行 build/tests/profile，但不能因此声称机器安静。小 GPU case 的 E2E 和设备探针波动很大，所以不挑几个偶然胜点宣称优势。

## 新回归也暴露了编译问题

新增 C++ 测试包含 26 个 kernel 配置 × 4 类输入（重复、唯一、全相等、正负零），共 104 次 dispatch。Metal4 和 TIRx/Metal 分别通过 473 assertions；SIMD 首次 CTest **300 s 编译超时**，失败日志原样保留。

2 秒调用栈采样显示当时正在 LLVM MachineScheduler／RegPressureTracker，而不是执行 kernel。当前 tile 元素数展开阈值没有计入 map 内嵌 reduction 的工作量，可能复制出巨大代码；这是结构性编译预算问题。后续共享 map 结构预算的修改与重测必须作为新记录追加，不能覆盖这个失败或把这张修复前的表当作修复后的成绩。

现有 `topk`/`sort` 是 O(N²) 的组合式参考算法。编译预算只解决表示膨胀，不会把算法变成高效 selection/sort。下一层应以现有 primitives/shuffle 组合更合适的库算法，再按 K/N、数据规模与硬件资源选择；不能用算子名字伪装成一般映射优化。

## 与 attention 优先级的衔接

Attention 是接下来的性能重点，尤其 decode。已保存的不同历史 cohort 显示：prefill 的 Tile/Torch GPU 比约 2.894；phase-specific QK reduction 的三个 decode case 仍约 9.245、3.798、9.640。它们不是本次新测结果，见性能文档中的 [attention 证据](../../../../../docs/source/performance/tile/results.md#composed-reductions-need-phase-specific-contraction-distributions)。

当前工作区已经有受保护的 matrix-initializer materialization 改动。它可能让 prefill 的 PV contraction 也 tensorize，尚不能据此宣布新性能结果，不能重复覆盖。优先实验为同一构建、固定 64 threads 与 8×16 block，只切 initializer materialization；随后重点设计 decode 的 `output_partition × reduction_partition` 候选，分别描述 QK、PV 的分布及阶段间重分布成本。单一全局线程数不足以同时处理这两个阶段。

## 归档边界

`evidence.tar.gz` 保存原始运行目录，包括 JSON、日志、command/process metadata、全部 tensor payload、生成源码、测试与 sample。`tested-sources.tar.gz` 是修复前测试源码及相关 XIR representation/planner；`runners.tar.gz` 保存当时的 driver 与依赖。原始 JSON 不改路径和 hashes。

构建已完成全量配置；初始 native test 的 clangd 为 0 errors，两个 benchmark 翻译单元各有 4 条既有计时转换 warning，78 个 Python units 通过。保存了 build/bin 动态库和 LLVM hashes，但没有保存完整 Runtime/LLVM/TVM 二进制闭包或实际 ORC native-entry 产物；因此不是自包含、可直接执行的环境镜像。审计可在解压目录只读重跑，无需原始 `/tmp` 路径。
