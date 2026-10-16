# Attention 优先级与 PV 探针：GPU 执行异常，性能结论暂停

2026-09-13 增加了独立于 QK 的 `--attention-pv mma|reduce`。默认仍为 MMA；实验分支用现有 DSL 表达 `acc * alpha + reduce(probability * value, n, add)`，用来观察 KV 累加方向能否更好地映射到 collective。它不是 production planner 优化，不引入新 primitive，也不承诺与 MMA 的浮点求值顺序逐位相同。

四个新增 SIMD 数值测试通过：prefill / ragged decode × QK MMA / reduce，均采用完整 FP64 oracle 和 output guards。整个 SIMD LLM 回归通过，Python 88 项 measurement/oracle/metadata tests 通过。对显式 PV probe，旧 binary 没有 acknowledgment 或返回错误 mode 都会拒绝。未在此次 checkpoint 执行 PV-reduce 的 GPU 性能测试。

## 尝试的 control 与实际失败

先运行既有 QK-reduce/PV-MMA control，两组 `B,Hq,Hkv,Q,K,D,Dv` 为 `1,8,2,1,2048,64,64` 与 `1,8,2,1,2053,80,96`；固定 block 1×32、1024 threads、subgroup reductions、readonly input views、FP32/precise math。每 shape 两轮 native/Torch 交换先后顺序，3 samples、10 ms target、30 ms warmup，native 单进程限时 30 s。Torch 2.14.0 / MPS 的 functional SDPA 包含其分配；native 输出预分配。GPU compute-pass 探针、无计数器 command-buffer control 和 E2E 分列。

八条 raw records 中六条被 runner 标记 `valid=true`，两条 native 超时；两个 shape 的 summary 均 `complete=false`。此外，控制台在第二个 shape 的 Torch 测量阶段记录了：

```text
The Metal Performance Shaders operations encoded on it may not have completed.
Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)
```

驱动错误没有转成 Python exception，runner 随后仍将该条记录标为 PASS。不能据此把它当作成功的性能样本。**整个 cohort 的 GPU/E2E 性能数据均不接受，不能只删除出错行再计算速度比。** 原始 JSON/console 保留不改，由 `acceptance.json` 显式记录整体不接受的判定。它也不能证明是某一个 Tile kernel、MPS 算法或桌面负载造成了驱动故障。

因出现这一执行异常，没有继续 PV-reduce arm、扩大矩阵或拟合 cost model：这些后续测量状态为 **NotRun**。不重启设备、不关闭用户其他应用，也不把重跑中偶然成功当作稳定性恢复。

## 下一步实验的前置条件

1. 用有界的 empty-sync / copy-only / tiny-kernel 检查确认 queue 与 host completion 稳定；如仍迟滞，用原生 SharedEvent 的 GPU 进度与反馈/清理时间区分，不依赖同一 host callback 链。
2. 再执行相同构建、相同输入、固定配置且交错顺序的 PV off/on 对照；检查实际 QK/PV collective、局部/共享中间数组与同步数量。
3. 将有收益且可泛化的选择落实为每个 contraction 的 `output_partition × reduction_partition` 候选，以及阶段间重分布/存储成本。不能靠 kernel 名字决定映射。

更完整的已实现/未实现区分、历史同口径表和 held-out 矩阵见 [attention mapping review](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md)。Prefill 的 matrix-initializer/guarded-input 工作区改动是另外一组受保护 WIP，本次隔离构建**没有包含它们**，所以本记录也不验证那些改动的正确性或速度。

## 归档范围

`evidence.tar.gz` 保存原始 control 目录和 console；`sources.tar.gz` 保存测试时的 fixture、benchmark、TIRx bridge 源码及测量脚本。CPU/build 验证日志位于相邻 [map-budget checkpoint](../m1-max-20260913-map-budget/notes.md)。构建基于 `147700bd431eb054694ba649817b7f50a335b2b1` 的隔离源码导出，工具链沿用该配置，未同时加载 TVM LLVM21 与 SIMD LLVM22 backend 做 in-process 对比。

现有 LLM driver 在临时目录逐项校验完整输出后清理 tensor exports；本 archive 保留 hashes/oracle receipts 和生成 Metal 源码，**不含完整 tensor payload 或所有依赖二进制**。它不是可自包含数值重放包。原始 `source_sha256` 不是整个工作区的身份声明；源码快照用于区分本次实际测试源与受保护的未合入 WIP。
