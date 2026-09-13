# CPU attention：实际 Tile native entry 与原生参照

日期：2026-09-13；Apple M1 Max，64 GiB，10 CPUs。源状态 `a8d81c9f3`
之后的 benchmark-only 探针；实际 producer 来自已验证的独立构建目录。
本轮没有生产 compiler/planner 改动，没有宣称达到 Torch/MPS 性能目标。

## 结论与边界

attention 仍有结构性差距，不只是几个展开参数：三组 decode 中，当前
Tile 分别约为 online NEON 的5.3、6.6、4.2倍，为 dense Accelerate 的
5.8、8.7、6.0倍。两条参照都比当前固定 Tile 配置快，但**手写 probe
不等于 Tile lowering，也不能把整体比率直接变成某个 MMA 的 cost 系数**。

这不是全局最优 Tile 配置搜索，也不是 Torch、MPS、Metal4、TIRx 或多线程
Runtime 排名。所有计时是单线程 native-entry host-wall：排除 Python、JIT、
Runtime、调用方分配/复制和验证；包含 entry、launch reset/block traversal、
编译器 helper，以及 BLAS 内部的 packing/分配（若有）。不是 CPU 硬件 cycles。
主机仍有用户/系统背景活动；我们的构建、回归测试在计时前结束，未停止用户进程。
仅作大差距诊断，不对微小系数作精细校准或置信度承诺。

## 六组预声明配置

shape 顺序 `B,Hq,Hkv,Q,K,D,Dv`；Tile QK/PV 均为 MMA，FP32、fast-math=false、
unordered-tree，W8 / block32 / local1，R4、全局 Tile 阈值64、region 阈值4096。
所有 Tile 都请求已有 full-packet specialization；实际 clone 与资源记录保留在 metadata。
cap/二维选项在计时前固定，不根据本批结果重选。

| case | shape | attention block | MMA cap | 二维请求 / 实际 MMA 数 |
|---|---|---|---:|---|
| decode-mha-d64 | 1,8,8,1,2048,64,64 | 1×16 | 0 | off / 0 |
| decode-gqa-d80 | 1,8,2,1,2053,80,96 | 1×16 | 8 | off / 0 |
| decode-long-kv | 1,16,4,1,8193,128,128 | 1×16 | 8 | off / 0 |
| prefill-q4 | 1,4,2,32,65,32,32 | 4×16 | 0 | on / 2 |
| prefill-q8 | 1,4,2,64,129,32,48 | 8×16 | 0 | on / 2 |
| batch-gqa-q4 | 2,6,2,17,67,40,48 | 4×16 | 8 | on / 2 |

## 完整性能表

每格单独比较 Tile(A) 与参照(B)：3 cycles × ABBA × 7 samples。
共12组比较、144 visits、1008 samples，无失败子集删除、无选择性重试。
每次 visit 的中位数，再取同实现六个 visit 中位数作为表中 µs；比值是
六对邻接 B/A 的中位数和完整范围，**不是表中两个汇总数简单相除**。
两种参照的 Tile 基线各自来自本组交错计时，不混合成一个基线。

| case | 参照 | Tile µs | 参照 µs | 参照/Tile，配对中位数 [min,max] |
|---|---|---:|---:|---|
| decode-mha-d64 | online NEON | 1037.176 | 195.929 | 0.188869 [0.188259,0.193033] |
| decode-mha-d64 | dense Accelerate | 1044.869 | 180.037 | 0.172474 [0.167744,0.175158] |
| decode-gqa-d80 | online NEON | 1932.406 | 290.954 | 0.150526 [0.150385,0.150650] |
| decode-gqa-d80 | dense Accelerate | 1932.062 | 221.296 | 0.114391 [0.114276,0.115103] |
| decode-long-kv | online NEON | 15249.188 | 3610.188 | 0.236737 [0.236201,0.243578] |
| decode-long-kv | dense Accelerate | 15261.542 | 2556.773 | 0.167532 [0.165634,0.176534] |
| prefill-q4 | online NEON | 123.855 | 45.497 | 0.367313 [0.362681,0.367837] |
| prefill-q4 | dense Accelerate | 123.855 | 29.552 | 0.238412 [0.237188,0.241195] |
| prefill-q8 | online NEON | 460.943 | 213.198 | 0.462501 [0.449725,0.463111] |
| prefill-q8 | dense Accelerate | 460.490 | 104.635 | 0.226285 [0.225645,0.230890] |
| batch-gqa-q4 | online NEON | 3194.609 | 111.551 | 0.034920 [0.034603,0.035299] |
| batch-gqa-q4 | dense Accelerate | 3197.000 | 60.531 | 0.018937 [0.018584,0.019082] |

## 数学、算法与资源差异

- 输入是留档的连续 FP32 Q/K/V，bottom-right causal GQA；`kh=h/(Hq/Hkv)`，
  `key<=query+K-Q`。完整独立 NumPy dense FP64 oracle，scale 按 FP32计算，
  atol=rtol=5e-5；每次验证所有输出、不可变输入、buffer/scratch guards。
  同一实现跨 visits 要求输出 hash 不变，不要求跨算法 bitwise 一致。
- online NEON 保留 KV16 `(max,sum,acc)` 递推。QK 使用显式四路贡献维归约，
  PV 沿连续 Dv 输出向量化；D/Dv 有标量尾部。它串行遍历 heads/queries，
  跳过全 masked KV 工作，并采用不同的 snapshot/packet 表示。
- dense Accelerate：Q=1 时 GEMV，否则 GEMM；QK alpha=1，随后显式 FP32
  scale、stable softmax、归一化，再 PV。每个 head 物化 Q×K scores，scratch
  跨串行 heads 复用；BLAS 内部 FMA/reassociation/packing 不是 strict-MMA。
- 两参照均声明并预分配 `4*max(Q*K,16)` 字节 scratch；online 实际只访问
  前64字节作为权重块。Tile 的实际 private workspace/静态 snapshot 容量另记，
  不把它们说成相同资源，也不从逻辑字节推断 cache/DRAM 流量。
- 胶水以 LLVM22 `-O3 -fno-fast-math -ffp-contract=off` 编译；这不限制
  Accelerate 的内部实现。每次计时前后在同一线程确认 BLAS mode=1，设置成功
  返回0；[Apple BLASSetThreading](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))
  是 calling-thread TLS 设置，不只是环境变量猜测，也不是独立 worker profiling。
- 仅对完整验证的受控输入报告。有限 FP32 仍可能发生 dot overflow；Tile 使用
  `-1e30` 初始 max，probe 使用 `-inf`。不宣称任意有限输入、全 masked row、
  任意 mask、paged KV、dropout、backward 或低精度语义已经覆盖。

## 实现方向

应在 typed MMA 被标量化之前保留每个 contraction 的轴/步幅/数值权限，
让后端选择 contribution-vector QK、output-vector PV 或矩阵 leaf。首版可以
窄范围准入，保留当前 fallback，但必须把 SSA snapshot、packing、调用频率、
跨 phase layout 转换与峰值 live storage 纳入共享 plan。
现有 SIMD 不支持普通 XIR CallInst，不能假设调用 BLAS 只是函数名替换。

详细约束与成本分解见 [attention mapping review 第14节](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md)。
下一次真正的优化结果必须由新 compiler 输出、普通 contraction 反例和
完整 attention 的同构建交错计时来证明；本次参照比率不会冒充该结果。

## 验证、来源与复现

完整选定构建通过；29项纯 Python 检查、24项原生 validation-only
（4形状×3模式×2参照）、四项现有 SIMD CTest 通过。原生 checks 无 timer loops。
六个实际 ORC capture 均先通过原 fixture 两模式检查，再由独立 oracle 复核；
随后所有 captures/preparation 完成才开始比较计时。无 GPU/Torch 执行。

`sources.tar.gz` 是实际独立 producer 的选定源码，加主工作树指定五个
probe/replay 文件与原始 validation；不是整个 HEAD 的导出。523个成员，
archive SHA256 `e4032e31fd334fe8b328e7418be10ac718d2911fbc9ddcfafb05136c5d2d76e7`。
`provenance.json` 逐文件记录来源与 hash；源码封存完成先于首次 capture。
producer、SIMD plugin、LLVM及生成对象/动态库均有指纹，但不宣称完整 OS
动态加载闭包或可重现构建证明。protected TIRx WIP 没有纳入编译器改动。

`evidence.tar.xz` 保存预声明计划、实际 ORC 对象/源、prepared probe、完整输入输出、
编译命令/stderr、逐 visit 样本、validation 和离线审计。保留原始路径字段作来源；
离线查看按归档相对路径解析。顶层 `SHA256SUMS` 校验归档和说明文件。
`run.py` 在 evidence 中记录固定构建位置；要复跑必须使用新输出目录，不能覆盖
本批证据。先完整构建与验证，再 freeze→capture/prepare→replay→offline audit。
