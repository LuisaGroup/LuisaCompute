# GPU preflight：完成/正确性检查通过，性能比较暂不验收

此文件是对本目录既有 GPU 运行的只读审查注释，不重写任何原始 JSON，也不将后续 CPU MMA-cap 候选的 producer 身份套到先前 GPU 数据上。审查未运行 GPU、重测性能或重建二进制。

`gpu-acceptance.json` 的独立验收结论是 `performance_comparison_accepted=false`、`cost_calibration_accepted=false`。此前发生 GPU hang 的 cohort 仍然无效；这些小探针不能追认旧数据，也不能证明整个设备已恢复稳定性能。

## 证据与检查边界

- `gpu_health.py`、`gpu-health/health.json`：3 个新 Torch 进程，各做 3 轮 empty-sync / copy / tiny-op，共 27 个阶段。copy/tiny-op 使用 37 个 FP32 元素；worker 检查 CPU roundtrip 的精确相等并同步。全部完成、exit 0、无 timeout；保存的 stdout 事件与总表一致。这是小规模完成/正确性准入，不是吞吐基准或全设备稳定性证明。
- `native_health.py`、`native-health/health.json` 及两个 case 子目录：Metal4 GELU-residual 与小 attention 均完成，原始输入、输出、FP64 参考仍保存。审计独立重算了完整 NumPy FP64 oracle。
- `tirx-mps-small/results.json` 及逐 visit 日志：单个 attention shape、两轮交错顺序、四次访问。完整张量被现有 runner 清理，只保留 oracle receipts 与内容 hashes；这里不能独立重放其全部数值检查。
- 以上共 9 个进程（3 health + 2 native health + 4 cohort），18 份 stdout/stderr 的大小和 SHA256 已与保存 receipts 核对；组合日志、process JSON 和生成源码的 hash 一致。用后来修复的 compact `GPUHangError` 规则重新扫描全部日志，未发现错误诊断；9 份 stderr 全为空。absence of diagnostics 不等于 absence of stalls。

## Metal4 小 native 探针

| case | 完整输出元素 | 独立 FP64 最大绝对误差 | E2E throughput 样本（µs） | E2E latency 样本（µs） |
| --- | ---: | ---: | --- | --- |
| GELU-residual，2×17 | 34 | 2.2304376212645138e-7 | 111802.917, 1819.375, 77660.417 | 199637.75, 407699.083, 191515.834 |
| attention，1,2,2,5,5,5,3，block 2×3 | 30 | 8.463576617323554e-8 | 207732.792, 191748.917, 207738.334 | 591710.5, 207711.083, 151741.792 |

二者保留 producer 的两次完整 oracle 检查和每次 34 个 guard 元素 receipts；`atol=rtol=5e-5`。独立 FP64 数值检查通过，但上述 host-wall 延迟已经说明不能把“跑通”解释成“性能健康”。guard 分配本身没有保存，guard 完整性仍依赖原 C++ 检查 receipts。

## 小 TIRx/Torch-MPS cohort：四次访问全部保留

固定数学：`B,Hq,Hkv,Q,K,D,Dv = 1,4,2,16,33,32,32`，block `4×16`，FP32，QK/PV 均 `mma`，native `fast_math=false`，bottom-right causal GQA attention。Native 是 `tile_tirx_metal`，不是 XIR-Metal4，128 threads/group、`mpp=false`。Torch 2.14.0 使用 MPS functional SDPA、GQA 与显式 bottom-right mask，没有 CPU fallback。

配置：2 rounds × 3 samples，target 5 ms、warmup 10 ms；顺序 r0 native→Torch，r1 Torch→native。每次访问的中位数均从原始样本重算；单位均为 **µs**。

| visit | 路径 | 无 encoder instrumentation 的 GPU command-buffer throughput control | GPU command-buffer latency control | E2E throughput | E2E latency | host batch reps | GPU control reps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r0 | TIRx Metal | 28.17773565765625 | 32.7082816511 | 31.1666640625 | 250.958 | 128 | 64 |
| r0 | Torch MPS | 28.62501423805952 | 28.666690923273563 | 1200123.333 | 391959.209 | 1 | 1 |
| r1 | Torch MPS | 19.21224065881688 | 25.250017642974854 | 53.7283670886076 | 313.291 | 79 | 64 |
| r1 | TIRx Metal | 34.4999134541 | 34.4583531842 | 208013.625 | 1191890.541 | 1 | 1 |

GPU control 的 method 是 `metal_command_buffer_timestamps_v1`，scope 为 `sum_of_command_buffer_gpu_intervals`，`encoder_instrumentation=false`。这里统计 command-buffer GPU intervals 的和，并按调用数归一化，**不是 isolated pure-kernel 时间**。Native 含两个 command buffer（包含 empty completion），Torch 随 batch 有一个或两个；repetitions 也不完全相同。另有 counter-instrumented 数据，但 instrumentation/control 的比例约 1–2.56，不能替代上表 control 宣称 kernel 性能。

异常不是稳定地归属于某条路径：r0 Torch E2E throughput 的原始样本是 `[1200123.333, 7852.167, 1400022.75]`；r1 native 是 `[207959.792, 399973.333, 208013.625]`。与此同时 GPU interval 只有约 19–35 µs。数据支持“显著等待发生在这里测量的 GPU intervals 之外”，但没有足够证据把根因指定成 runtime、driver、操作系统调度或 GPU 恢复状态中的某一个。

因此不汇总为稳定的 E2E 胜负或 cost-model 拟合数据。GPU throughput 的两轮 native/Torch 比值也从 0.984374555 变成 1.795725656；取中位数并不能消除这种漂移。短 warmup、两轮数据不足以验收稳定性能，不能声称已打败 MPS/Torch。

### 数值完整性与可复现性限制

四个访问均 `status=OK, valid=true`，各检查完整 2048 个元素。Native 最大绝对误差 2.0780765297434556e-7；Torch 最大绝对误差 2.5049095830897983e-7。Torch pre/post 检查与 no-fallback receipts、native 两次 guarded 检查都存在。原始张量已经清理，所以上述是经一致性检查的运行 receipts，而不是本次独立重算出的 GPU cohort 数值结论。

两轮之间同一路径的输出 hash 一致；跨路径 hash 不同但误差均在原 oracle 容差内：

- Native：`a9ddbff42662a96d61f23a3b7d6d46090193c67a63be5c10bc056c7306b82d8a`
- Torch：`6bd127aee18492017fdfec70e9336f663cb9515a2b248f33087b02a27611d687`

四次访问相同的三个 input hashes：

1. `36f12731fc5285eca8f7b596424f5982111c19c591f3af419774a8431be5e33f`
2. `d4b4166de09e6a57a60a84b9446778c8da2bf4e3d67cc778a4da37ffa16d028c`
3. `3313771d79ae053849ae7a235679f9df12c44ad0288be826585468655cb5f147`

Torch E2E 包含 functional SDPA output allocation，native 预分配输出；这也是口径差异，不能称为完全一致的 isolated-kernel 对照。

## Runner 源身份：明确保留旧 hash，不改写历史

此 cohort 启动时父进程已加载旧版 `compare_llm.py`。其 metadata 记录的 SHA256 是：

`7f9b884a387df40ebf17899f21e5b0f8c38f5f3e1cbeee75699be4e040cd5ea7`

后来只修补 compact GPU-hang spelling 的源文件 SHA256 是：

`46899aed04460e88d319fa35641e1d6325cf335aa9a86efa875f0c939583b10f`

只逆转这一处正则变化就能从新文件恢复 cohort metadata 中的旧 hash。旧规则要求 `GPU`、`Hang`、`Error` 之间有空白；新规则对 Hang 专门使用 `GPU[\s_-]*Hang[\s_-]*Error`，从而接受无分隔、空白、下划线和短横线写法，同时其他诊断规则保持不变，普通 `max_abs_error` / 空 error JSON 不因此误报。数学及计时路径没有随此修补改变。

源文件修改时间为 2026-09-13 17:35:53 +08；两个 Torch worker 的 stdout 分别在 17:35:40、17:35:41 已完成；cohort 结果文件 17:35:57 才写完。保存的 worker 时间线支持它们在补丁前完成，但当前 schema 没有逐 worker 的独立源码 hash，不能假装有更强的源加载证明。主线程已把旧文件按 metadata hash 冻结在 `preflight-sources/compare_llm.py`，应随此 GPU preflight 一起归档。

`artifacts_unchanged` 原本覆盖 binary/extra-artifact identities，不覆盖 Python 源文件。原始 `complete/cohort_valid=true` 表示 runner 数值、日志及其 binary-identity 检查完成，不代表本次独立性能稳定性验收通过。请同时保留原始 JSON、旧 runner identity 与 `gpu-acceptance.json` 的独立拒绝性能验收结论。

## 后续 CPU 实验不与此 GPU cohort 混池

本目录 CPU MMA-only cap0/cap8 实验在后来的 producer/source freeze 之后执行；其 `sources.tar.gz`、`provenance.json`、`audit.py`、`audit.json` 有各自独立证据链。GPU preflight 使用先前 15a9b475c producer 与旧 Python runner，而 CPU provenance 含新 cap 候选及 compact-regex 修补。两者可以一同归档，但不能使用同一个 source hash 或合并计时统计。
