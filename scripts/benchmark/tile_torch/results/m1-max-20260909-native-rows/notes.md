# SIMD 六类行算子的真实 native 对照

2026-09-09，Apple M1 Max，FP32，单 CPU 线程。沿用 `4f46d03f3`
对应的已验证编译器与隔离构建；本次新增测量设施与证据，**没有改编译器或自动 planner**。

## 结论

已有通用 lowering 的收益不限于 RMSNorm：固定 local=8 时，RMSNorm、
LayerNorm、GELU+residual 的 12 个尺寸均快于实际 TorchInductor 2.14.0
native entry，72/72 配对轮次胜出。但 masked softmax、SwiGLU、RoPE 的
另外 12 个尺寸全部落后，0/72 配对轮次胜出。**总体性能目标未完成。**

下表是各算子四个尺寸的配对中位比值范围；不是置信区间，也不是四个尺寸的平均加速。
分子为 Tile XIR/SIMD local=8 时间，分母为 Inductor；小于 1 表示 Tile 更快。

| 算子 | Local/Inductor | 配对胜出 | 当前判断 |
|---|---:|---:|---|
| RMSNorm | 0.337–0.469 | 24/24 | 延续已有 native 优势 |
| LayerNorm | 0.374–0.735 | 24/24 | 此次补齐 native 证据；须保留方差算法差异 |
| GELU + residual | 0.568–0.645 | 24/24 | 此次补齐 native 证据；不能一概归因于数学函数慢 |
| Masked softmax | 1.350–1.607 | 0/24 | 仍慢约 35–61% |
| SwiGLU | 1.166–1.244 | 0/24 | 仍慢约 17–24% |
| RoPE | 1.247–1.944 | 0/24 | 最大相对缺口；适合检查共享输入、多输出融合 |

完整 24 行的 whole/local/Inductor 时间、配对比值和六轮范围见 [tables.md](tables.md)。
所有大小、所有败例均保留。本次既没有新 Metal/MPS/BLAS/GEMM/attention 结果，
也没有把八线程 Runtime 历史时间与单线程 native 时间拼接起来。

## 如何测量

### 固定的实验与选择边界

- 六类算子，每类固定 `17×65`、`129×768`、`257×1538`、`1024×4097`；
  RoPE 需要偶数宽，分别用 66、768、1538、4098。最大主张仍只到约 420 万 FP32 元素。
- 两个 Tile mapping 分别固定 `local_lanes=1`（whole）和 `8`（local）；
  packet W8、block=32、一个 CPU worker。只比较这两个预定候选，不按实测挑赢家。
- full-packet specialization、predicated effects、cohort-private access 开启；
  load/reduction fusion 关闭，fast math 关闭，unordered reduction partitions=4。
  这些都是已存在的显式开关，**不是新的自动/default 路径胜利**。
- 实际 Inductor 由 `torch.compile` 捕获完整图后调用 `torch._inductor.compile`，
  使用其生成的 C++ 和 `.so`。不是手写仿制 Torch kernel，也不是 eager Python API。
- 源码及相邻 Luisa 二进制闭包与前驱记录逐项校验；先完整构建，再捕获全部 48 个
  Tile entries；Torch 编译、动态库链接和反汇编全部完成后才开始 native 重放。
  无构建、测试、文档渲染或其他本任务的 benchmark 与正式计时并行。
- 每个尺寸运行 whole/local/Inductor 的全部六种排列；每次访问先 warm 100 ms，
  自适应 repeat 以约 30 ms 为采样目标，取七个样本。保留 repeat 数与全部样本。
  最大 repeat 有上限，因此不声称每个小 kernel 样本必然达到 30 ms。

### 计时边界

```text
计时前：JIT / link / ABI 解析 / host 分配 / 64-byte 对齐 / poison 输出
                            |
                  共同的 C++ callback timer
                  /                       \
     实际 ORC block/packet entry      实际 Inductor kernel entry
     必要的遍历 + launch reset        原生代码内部的 scratch 分配/库调用
                  \                       /
计时后：完整 FP64 oracle / 输入不变 / 前后 guards / 重复输出 hash
```

计时是 warm 单线程 **native-entry host-wall µs**：排除了 Runtime dispatch、
Python、JIT 和 caller 分配，但没有扣除共同 callback、函数入口、必要遍历、
launch-record reset 或生成代码自身的 libc/分配。这不是硬件周期计数，也不是
E2E dispatch 延迟。输出与 scratch 在同一 entry 的重复调用间复用，工作集是 warm 的。

特别地，Inductor 的四个 masked-softmax kernel 均在函数内生成一个
`std::make_unique<float[]>(width)`；这部分仍在 timer 内。
不能把“caller 分配排除”写成“所有分配排除”，也不能手动移走它来美化 baseline。

### ABI 不能按算子名字猜

新的 `native_rows.py` 解析实际生成 wrapper 的 AST，恢复图输入顺序、
指针 constness、scratch、reuse 与返回 view；只接受一个静态 FP32 C++ entry，
拒绝未知 wrapper effect、额外调用、动态尺寸、不匹配签名和越界 alias。
`native_rows_replay.cpp` 只调用真实 entry，不包含任何算子算法。

| 实际情况 | 重放需要处理的内容 |
|---|---|
| RMSNorm | 图中 gamma 排在 x 前，不可假定是声明顺序 |
| 小尺寸 LayerNorm | 第一个参数是 mutable reused scratch，不全是 const 输入 |
| 大尺寸 LayerNorm | 另一个六指针 ABI；与小尺寸的 const-mask 不同 |
| Masked softmax | 只有一个有效图输入，额外 scratch 与 native 内部分配均保留 |
| RoPE | 两个输出指针指向同一 allocation 的不同半区，保留原始 stride/offset |

这些是实测生成器输出的差异，不是要求 DSL 用户增加实体。

## 正确性与数值语义

所有输入沿用相同确定性 FP32 数据，Tile 与 Torch 读取同一组原始位。
每次 native visit 后检查全部输出，误差界为 `atol=rtol=5e-5`；检查全部输入
未被写入，并检查每块 host allocation 共 128 个 guard floats。
Tile 的 native private workspace 另有字节级 guards。输出先填 NaN，防止漏写。

固定 entry 的六次访问必须保持相同输出 hash；不同 mapping 或不同框架只要求
通过 FP64 tolerance，**不声称跨实现位相等或全数值域等价**。
本轮不是大偏置/消去、NaN/Inf、subnormal 等数值压力测试，不能据此放宽 DSL 的数值约束。

需保留的算法差异：

- Tile LayerNorm 使用先求均值、再归约中心化平方；大宽度 Inductor 实际代码采用 Welford。
  性能胜出不证明前者在所有数据分布下与后者有相同数值性质。
- 宽行 Inductor RMSNorm/softmax 保留 cascade reduction；归约树、除法/倒数乘法可能不同。
- GELU 双方都使用 tanh 近似公式，但底层 tanh 的实现不同。
- Tile masked softmax 显式把无效位置的指数贡献设为零；Torch 图先填 `-1e30`
  再 softmax。在此次有限输入范围内完整 oracle 一致，不推广到任意特殊值语义。

独立 [audit.json](audit.json) 重新读取 48 个 capture 输出和 72 个唯一 native
输出快照，复算 432 次访问的统计、配对顺序、比值、完整性及固定 entry 的位稳定记录。
guard 数组没有持久化；其检查发生在每次重放现场，不能声称事后重新读取了它们。
九种破坏证据的 mutation 检查均被拒绝。

## 对通用优化的启发：已有证据与待验证假设分开

### 1. 优先检查 RoPE 所代表的多输出、共享输入 DAG

`native/rope-129x768/inductor.cpp` 在同一 vector loop 读取四个输入向量，
共享这些值并写出两个分区输出。对应的 XIR bridge 当前只延迟纯 single-use
elementwise；load 与多 consumer 值仍在定义处 materialize，两个 store 独立发射。

这是明确的编译结构差异，但还不是硬件 profile 的因果归因。
下一候选应基于共同迭代域、访存 effect、snapshot 生命周期和输出分区关系，
比较“共享 SSA DAG 的流式融合”与“保留 materialization”。不要按 `rope` 名字识别，
也不要因为 `parallel` 无迭代间冲突就忽略迭代内部的 load/store 顺序。
该候选还应覆盖其他多输出 pointwise 图，并保留别名/覆盖输入等拒绝测试。

### 2. Softmax 的候选和代价必须按 phase 组织

Inductor 显式组织 max、exp+sum、normalize，并复用一个宽度大小的 scratch。
Tile local 仍需多个 Tile snapshot 与 predicate/归约 phase；四个尺寸全部落后。
仅把载入改为 contiguous 并不足以解决这类组合程序。

应比较合法的 phase fusion、scratch/materialization、full/tail 分区和 mask 实现；
把 producer/consumer 间的重读、临时 live range 与归约依赖放进同一个候选。
不得把所有中间量消除当作必然获益：前驱 load/reduction fusion 已有真实 native 退化反例。

### 3. 不要把 exp、tanh 与普通加法使用同一粗糙权重

SwiGLU 仍慢 17–24%，但 GELU 在本轮已经获胜。因此“所有 transcendental 都慢”
不是数据支持的结论。实际生成的数学实现、内联/库调用、除法、向量宽度、
predicate 与 live-state 都应进入 backend policy；不能只按 TileIR opcode 个数解释时间。
先做独立 primitive/code-shape 测量，再用未参与拟合的组合图检验泛化，避免直接
把这 24 个结果拟合成算子/尺寸查表。

### 4. 保持 mapping、realization 与 Runtime grain 三层成本

本轮 metadata 均为 fixed single candidate、`custom_cost_policy=false`，不是 solver
重新发现了胜出方案。建议的求解结构仍是：在语义/资源约束下，联合选择逻辑域到
packet/lane 的映射、phase 内存实现，以及独立的 CPU task grain。

```text
semantic legality / layout + access facts
                    |
       mapping × phase realization candidates
                    |
          native cost / live resource budget
                    |
        independent task-grain + E2E cost
```

下一轮先在 frozen native timer 中验证 RoPE 类共享-DAG候选，再分开验证 dispatch。
最后才校准可替换的 cost policy，并给 solver 独立的未拟合尺寸/组合图验收。
本报告仅提出候选与验证顺序，**没有实现或验证新的 cost model**。

## 复现、保留记录与仓库同步

工具入口与三阶段命令见 [benchmark README](../../README.md#actual-native-row-entries-without-runtime)。
该版本的 capture 有意绑定前驱 provenance 与二进制闭包；源码更新后应显式建立新
基线和记录，不能绕过检查沿用旧结果。单纯把路径改成另一个 build 不足以复现本实验。

- [provenance.json](provenance.json)：源码继承链、二进制闭包、374 个归档的原始/压缩 hash。
- `capture.json.gz`、`manifest.json.gz`、`replay.json.gz`：完整原始记录，包含实际临时目录。
- `captures/`：全部 48 个 LLVM、ORC object、realization metadata。
- `native/`：实际链接的 Tile/Inductor libraries、生成 C++/wrapper、完整反汇编。
- `runner-sources/`：本次真正执行的工具快照；`build-capture-prepare-logs.json.gz` 保存完整构建及子进程记录。
- `preflight-failures.json.gz`：最初把 backend 标签 `cpu` 误验为 `simd` 的捕获失败、
  首轮结果写 JSON 时 NumPy 整数不可序列化的失败；均不参与最终 24 组比较。
- 大块输入/输出仅保留在 provenance 指向的临时目录；仓库记录其 hash 和逐项验证。
  没有删除旧实验或改写前驱证据。

工具的九个单元测试与 C++ 严格语法检查通过；benchmark Python 全套 116 个测试中
114 个通过、2 个因当前依赖环境跳过。实际三个阶段完成并通过独立审计。
此次未改编译器，没有把旧 CTest 当作新跑过的结果。文档构建及最终桌面/窄屏渲染
检查记录于 `validation-final.json` / `validation-final-visual.json`；第一版检查也保留，
随后把页面摘要表精简为三列，使手机宽度不必横向滚动就能读到全部比值和胜出轮数。

按用户要求已 fetch `origin/next` 到 `4b0c02384`，本轮冻结计时之前未合并它。
上游包含新的 `yyjson` gitlink，而工作区该子模块已有未提交变化；不在本次归档/提交
中覆盖它。后续同步应作为单独集成步骤，重新完整构建及回归后再发布性能结果。
