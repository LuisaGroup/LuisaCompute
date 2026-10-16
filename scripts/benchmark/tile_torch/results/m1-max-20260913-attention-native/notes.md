# SIMD attention：PV → reduce 不是通用优化

2026-09-13，Apple M1 Max，FP32。这是 **XIR/SIMD native-entry 单线程实验**，没有新 Metal、MPS 或 Torch 性能比较，也没有修改生产 planner 默认。

## 结果

PV 改为 `reduce(probability * value, n, add)` 在两个尺寸、两种固定 QK 表示下都变慢。QK 改成 reduce 在 decode 上只改善约 1.6%，在 prefill 上则变慢约 33%。保留默认 MMA，不能用全局表示替换代替真正的执行映射搜索。

| Case | 固定项；唯一变化项 | baseline µs | candidate µs | 配对 candidate / baseline |
|---|---|---:|---:|---:|
| Prefill | QK=MMA；PV MMA→reduce | 138.914 | 185.731 | 1.332 |
| Prefill | QK=reduce；PV MMA→reduce | 188.840 | 234.069 | 1.252 |
| Prefill | PV=MMA；QK MMA→reduce | 141.512 | 188.219 | 1.329 |
| Decode | QK=MMA；PV MMA→reduce | 3130.297 | 3532.771 | 1.129 |
| Decode | QK=reduce；PV MMA→reduce | 3090.826 | 3272.365 | 1.060 |
| Decode | PV=MMA；QK MMA→reduce | 3137.724 | 3082.836 | 0.984 |

每行是独立 ABBA 实验，3 cycles × 4 visits，每 visit 7 samples，warmup 30 ms、target 15 ms；比值是六个相邻配对比值的中位数，不是两个表格中位数相除。共 **8/8 实际对象捕获成功、72/72 计时 visits 通过完整 oracle、输入不变和内存保护检查**。原始 captures 的 Runtime 时间只用于验证捕获链路，不与单线程时间作开销相减。
存在桌面背景负载（见 `replays.json`），这是局部诊断，不是 quiet-machine 校准或全 workload 泛化证明。

## 语义、尺寸与控制变量

shape 顺序为 `B,Hq,Hkv,Q,K,D,Dv`：prefill 为 `1,4,2,32,65,32,32`、block `4×16`；decode 为 `1,8,2,1,2053,80,96`、block `1×16`。两者均为连续 KV、bottom-right causal GQA、完整 online-softmax 状态、FP32 precise math、`unordered_tree`，容差仍为 `5e-5 + 5e-5 × abs(reference)`。
每个配对固定输入/FP64 oracle 字节、block、源 reduction policy、请求的 group/input-view/reduction-candidate 设置，只改变表中的 QK 或 PV。实际 mapping 均为 W8、32 workers/block、local lanes=1；这不意味着 LLVM 指令相同。

## 计时与证据

复用 `native_tile.py` / `native_tile_replay.cpp`，链接 Runtime JIT 捕获的实际 ORC `.o`，**不从 LLVM 文本重编 kernel、不重新实现 attention**。计时包含 native entry、launch-record reset、block traversal 及 compiler-emitted libc/内部 allocation；排除 Runtime、Python、JIT、调用方分配、输入复制与数值校验。不是 CPU cycles、多线程 Runtime 吞吐或 GPU 时间。

`evidence.tar.gz` 保留输入、C++ FP64 oracle、完整输出、LLVM、实际对象、链接库、prepared manifests、replay 输出及日志。`audit.json` 独立重算配对比值并核对输出、固定控制项和文件哈希。`sources.tar.gz` 冻结测试/runner 与隔离构建的 Tile/XIR/SIMD 源；`provenance.json` 记录 source/build 状态与二进制指纹。
测试源为 `32bfbe106` 加本轮三个 C++ 文件改动的隔离构建，**不含远端新增 19 commits 或主目录未提交的 TIRx WIP**。prepared copies 保留捕获时的 runner；最终 runner 另收紧固定控制项检查，没有改变 emitter、ABI 或数学语义。ABI admission 加实际数值执行不是任意对象的独立 ABI 证明器。

## Compiler 启发与边界

PV 两种表示的 snapshot 字节数相同：prefill 为 6688（QK=MMA）或 6944（QK=reduce），decode 为 12864 bytes/worker。因此**不能直接把 PV 变慢归因于更多 snapshot 容量**；还需对照 loop organization、private accesses 与实际指令。静态 totals 不是峰值 live bytes 或物理寄存器数。Decode private workspace 为 102912 bytes，replay 在计时外配置并验证它。

TIRx unmatched MMA 的 collector 只计 output-domain 次数，贡献维 K 到 planner 前已丢失；`matrices.empty()` 又直接返回 reference。只改 cost coefficient 不会产生 output × reduction 候选。下一步仍是依据 typed contraction/domain/access 生成逐阶段候选，并联合考虑 QK/PV 转换与状态成本，见 [设计侧记录](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md)。

## 同轮验证，不混作性能成绩

- 完整隔离构建成功；SIMD LLM 和两个非法参数 CTest 均通过。新增 12 个小配置复用 shader 跑 24 次，含 FP64 oracle 和 zero-Q/K 的独立因果 prefix-mean oracle。
- 通用 native replay helper 的 6 个验证测试及 LLM admission 的 7 个纯 Python 测试通过。
- QK/PV controls 也可通过 XIR/SIMD 和 XIR/Metal4 benchmark 显式传入；本轮只执行 SIMD，不推断 Metal4 新增 cases 已通过。
- `compare_llm.py` 将 Torch 移入独立 worker，捕获 C/C++ fd stderr；即使 exit=0，已知 GPU failure 也令整轮无效，余下 GPU visits 保留 NotRun。它不证明 GPU 已恢复，也不是穷尽驱动错误的 decoder。
- 21 个 runner 单测通过，包括真实 `os.write(2)` + exit=0 和失败后 1 Error/7 NotRun 的执行检查。实际 Torch CPU worker 的 attention+RMSNorm 混合 smoke 为 8/8 OK；这批短预热时间仅作协议验证，不是 Torch 性能表。

性能目标仍未完成；negative probes 不提升为生产默认。
