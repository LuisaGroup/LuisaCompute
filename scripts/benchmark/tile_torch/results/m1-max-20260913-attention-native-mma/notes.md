# Attention：每 program 的原生向量 MMA 候选

2026-09-13，Apple M1 Max，macOS 26.6.2，LLVM 22.1.8，FP32。

## 结论

这次实现的是 **Tile → XIR → SIMD 编译器路径**，不是前次手写 NEON/Accelerate 探针。它保留 typed contraction，在同一 LLVM module 内生成私有向量 helper：本批 QK 沿贡献维向量化，PV 沿输出维向量化。选择不依赖 attention 或轴名称，没有新增 DSL primitive、Runtime callback 或 BLAS 导入。

**候选有数值正确性和代码生成证据，但不是普遍的性能改进。** 六组中，batch GQA 的配对时间减少约25.8%；另外五组增加6.2%–60.4%。默认保持关闭，也没有把这些结果拟合成自动 cost policy。MPS/Torch/BLAS 的整体性能目标仍未完成。

| case | B / Hq / Hkv / Q / KV / D / Dv | query×KV block | cap / 2D request | off µs | on µs | 配对 on/off | 六对范围 | on 较慢对数 |
|---|---|---|---|---:|---:|---:|---|---:|
| decode-mha-d64 | 1 / 8 / 8 / 1 / 2048 / 64 / 64 | 1×16 | 0 / off | 1034.232 | 1593.974 | 1.5236 | 1.4570–1.5930 | 6/6 |
| decode-gqa-d80 | 1 / 8 / 2 / 1 / 2053 / 80 / 96 | 1×16 | 8 / off | 1920.786 | 2789.068 | 1.4515 | 1.4273–1.4721 | 6/6 |
| decode-long-kv | 1 / 16 / 4 / 1 / 8193 / 128 / 128 | 1×16 | 8 / off | 15214.875 | 24403.062 | 1.6037 | 1.5762–1.6069 | 6/6 |
| prefill-q4 | 1 / 4 / 2 / 32 / 65 / 32 / 32 | 4×16 | 0 / on | 123.800 | 176.118 | 1.4210 | 1.4064–1.4359 | 6/6 |
| prefill-q8 | 1 / 4 / 2 / 64 / 129 / 32 / 48 | 8×16 | 0 / on | 461.137 | 489.758 | 1.0619 | 1.0426–1.0840 | 6/6 |
| batch-gqa-q4 | 2 / 6 / 2 / 17 / 67 / 40 / 48 | 4×16 | 8 / on | 3194.094 | 2368.216 | 0.7417 | 0.7408–0.7472 | 0/6 |

每个 µs 是六次 visit 中位数的中位数；比率是六个相邻匹配 on/off 比率的中位数，不是表中两个中位数相除。范围是描述性范围，不是置信区间。全部负结果保留。

## 测量与正确性边界

- 两臂都使用实际捕获的 ORC object，同一个 C++ 原生计时器，单调用线程；不是重新编译生成的 LLVM 源码近似物。
- 指标是 `single_thread_native_entry_host_wall_us`。包含原生入口、launch reset、block traversal 和编译器内部工作；排除 Runtime dispatch、Python、JIT、调用者分配及输出校验。不是硬件周期计数，也不是 Runtime E2E。
- 所有六组、十二次 capture/prepare 完成后才开始 replay。每组3轮 ABBA、每 visit 7个样本，共72 visits / 504 samples；warmup 30 ms，目标每样本15 ms。没有事后筛掉慢样本。
- 两臂固定 QK/PV MMA、R4、W8、32 workers/block、local lanes=1、global tile budget=64、region budget=4096、FP32、global fast math=false，full-packet specialization 都请求开启。只切换 `LUISA_SIMD_NATIVE_MMA_VECTOR_WIDTH=0/4`；cap 和2D request按表预先固定。
- root order 与 blocks/task 在每对中相同，但**实际快照、交错存储、代码和 clone 准入允许变化**。本批是完整 realization 对比，不能解释成单一向量指令的因果收益。
- 全部输入、完整输出和分配 guard 均检查。独立 FP64 `einsum(optimize=False)` 重新构造 bottom-right causal GQA attention；使用 FP32 scale，容差 `5e-5 + 5e-5*abs(reference)`。所有 replay 必须逐 bit 等于本臂 capture。两个臂之间因获准的 QK 重结合，不要求逐 bit 相同；本批六对实际都不同。
- 每次 native replay 都检查输入未变、guard 通过、输出有限且完整。最大绝对误差不超过2.010e-7；这只覆盖本批受控有限 FP32 输入，不是所有输入等价证明。
- 测量时没有并行构建/测试，但桌面有用户及系统后台活动，未停止用户应用。不是安静隔离主机，不用于微小成本系数标定。历史手写 BLAS/NEON 结果不是本批匹配基线，不能跨会话拼成加速比。

## 资源变化及结构性启发

| case | snapshot bytes/worker off→on | allocations off→on | interleaved arrays off→on | replay workspace bytes off→on | 实际 full-packet clone off→on |
|---|---:|---:|---:|---:|---:|
| decode-mha-d64 | 8320→9216 | 4→9 | 4→1 | 66560→73728 | 0→0 |
| decode-gqa-d80 | 12864→13376 | 8→11 | 8→3 | 102912→107008 | 1→1 |
| decode-long-kv | 18560→19200 | 8→11 | 8→3 | 148480→153600 | 1→1 |
| prefill-q4 | 6688→7712 | 10→13 | 10→5 | 0→0 | 0→0 |
| prefill-q8 | 12720→14768 | 16→18 | 14→8 | 101760→118144 | 0→1 |
| batch-gqa-q4 | 9120→10400 | 10→13 | 10→5 | 72960→83200 | 0→0 |

这些是编译器快照容量和 replay workspace，不是物理寄存器数、DRAM流量或占用率。workspace=0也不代表没有栈/私有状态。prefill-q8 从没有满包 clone 变为一个（3786 cloned instructions）；其6.2%回退不能只归因于 helper。

已证实的静态变化：输入和seed在定义处形成可寻址快照，输出使用新storage；调用参数会让对应数组退出 packet-interleaved private-array 准入。每个 active program 执行自己的内维向量 helper。QK确有连续贡献向量乘加及水平合并，PV确有广播乘连续输出向量；没有 FMA 或任意外部原生 callback。更少的算术指令不保证完整 phase 更便宜。

对实际MHA-on对象的反汇编进一步确认：helper已经内联，kernel body内没有`BL/BLR`，因此不能把该组回退归因于未内联的helper调用。QK的16-output循环仍在每个score内加载同一Q的八对向量，K指针每个score推进256字节；机器指令确为分离的`fmul.4s/fadd.4s`。这暴露了贡献向量化尚未组合原有多输出复用的问题，但没有量化其耗时占比。该机器码结论限MHA，详见evidence中的code-review及反汇编收据。

尚未通过采样/计数器量化的原因：快照写入、AoS/SoA访问变化、跨program gather/scatter、调用/循环开销和寄存器压力各占多少。不能仅凭相关性宣布其中一个已被证明是瓶颈。后续应先 profile 相同原生入口，再比较保留 packet 布局的向量实现与当前 per-program 连续布局，单独验证转换边界。

另一个小候选是把已验证的output不别名事实显式传为helper参数属性，同时仍允许三个输入相互别名。它尚未实现或计时，而且O2/内联可能已恢复部分事实；不能先验声称加`noalias`就会提速。

对 planner 的直接要求是联合选择 `(phase realization, physical layout)`，并计入相邻 phase 的转换与 live storage，而非仅给向量 MMA 一个收益系数：

```text
min Σ compute_cost(phase, realization, layout)
  + Σ transition_cost(producer_layout, consumer_layout)
  + snapshot / call / code-size / live-state costs
s.t. coverage、依赖、数学权限、实际容量和 target 能力
```

这是待标定的目标分解，尚不是实现完成的预测模型。本版只把 native calls、输出/贡献向量组及准确的快照工作交给 backend cost policy，并明确报告 `native_mma_cost=unmodeled`。同样的候选可用于普通 GEMV/strided MMA；性能泛化还需非 attention 和 held-out 尺寸验证。

## 实现、验证与重放

`StridedMmaMD` 是必需的 typed XIR 语义，不能靠函数名称识别。clone、文本/bitcode保存所有字段；XIR及SIMD边界拒绝错误placement、重复metadata、类型、容量、非local引用和输出别名。Schedule持有descriptor副本。严格MMA仍保持ascending K和separate MUL/ADD；贡献树要求局部reassociation许可。kernel-wide fast math不能覆盖strict MMA。默认width=0，backend target info首版只显式接受2/4/8；没有把这些数字设为execution hierarchy上限。

完整选定构建 `full-build-5` 通过，随后11项CTest全过（143.45秒）：Schedule IR、warp uniformity、XIR→Schedule、LLVM codegen、Tile DSL、target info、Tile Runtime、LLM、XIR verifier、interchange、passes。23个修改的C++ translation units精确syntax检查全部通过。数值回归覆盖strict/default、K=0/1/3/4/5、尾包、signed zero、FMA敏感样本、输入覆盖后的定义时快照及carry。Metal/Metal4只验证构建及strict-math forwarding，未声称GPU执行已验证。

历史失败不删除：前三次full build分别暴露printer整数重载、Schedule引入不应有的AST依赖、BoostUT byte-vector诊断问题；第4次完整构建通过。第一次回归10/11通过，target-info测试对K=1/N=1退化布局的期望错误；修正测试oracle后完整重建、第2次11/11通过。旧失败日志保留，不能改记为成功。此次11项通过也不是“全仓所有测试都通过”。

`sources.tar.gz` 在第一次capture前冻结，包含623个实际选定源文件/配置/验证证据；36个owned编译器及测试文件在主树、隔离导出、成功门禁和实验plan之间哈希相同。构建来自 `/tmp/luisa-next-integration.roZbN8/source`，不是整个主树HEAD的重建，未混入其他人的TIRx WIP。producer/plugin/工具都有指纹，但不是全部动态loader依赖闭包或可复现构建证明。

`evidence.tar.xz` 保存plan、冻结/运行脚本、全部门禁尝试、capture/prepare命令及stdout/stderr、对象/汇编、全部输入输出、72次visit和审计。离线复核不加载native代码：安全解包evidence，将单独的sources.tar.gz复制进对应raw目录，再用归档audit.py重新校验。`SHA256SUMS`及package inventory用于核验文件集合；具体审计命令见audit.py说明。

独立离线审计已通过：重算完整FP64 reference和每份输出、全部统计，核对623源码成员、53次历史门禁记录及90条运行/准备命令。batch GQA使用实际packet-batch ABI2，其余五组使用block-batch ABI0；每对内部ABI一致，不是强行假设六组入口完全相同。审计器前四次失败也原样保留：分别是macOS临时路径别名、早期构建实际只有35个owned文件、batch的ABI差异、审计脚本自身路径别名；修正记录解释后第5次通过，没有改输入、输出、样本或编译器产物。临时uv Python启动路径被记录，但解释器二进制未独立归档/指纹验证。
