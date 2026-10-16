# Metal4：固定 TileIR/BQ 的线程组划分对照

2026-09-14，Apple M1 Max。结论：**不能把 block32 设成通用修复；完整 attention 仍需要逐阶段的协作分布。** 本轮没有改变生产 planner、lowering 或数学策略，也没有新的 Torch/MPS/BLAS 对照。

## 实验边界与来源

- 原生源码来自 `b21d33053` 及递归固定的19个 Git trees，导出到独立持久缓存；不复制主工作区未提交的 TIRx 等修改。Python timing harness 单独记录为工具修改。
- 宿主 AppleClang21、RelWithDebInfo，后端 LLVM22；Metal4 原生路径为 TileIR→XIR→LLVM/AIR。配置、构建命令、依赖来源和指纹见证据包；不是把其他 Metal/TIRx 路径改名为 native。
- 6个配置，每配置 `[auto,32,32,auto]` 两遍，共48 visits。每 visit 独立进程、3个样本、固定8次吞吐 dispatch，并保留单次 latency。每配置只有**4个相邻配对**，不是12个独立配对；没有按时间筛选、重跑赢家或删除慢样本。
- 固定 `local_lanes=1`、BQ/BK、root order、MMA QK/PV、FP32、fast_math=false，以及 reduction/临时表示选项。只改变请求的 block size。非 attention 对照也故意固定 local1，**不代表它们在自动 local-distribution 搜索下的最佳性能**。
- 每次检查实际 dispatch、所有计时阶段的 block size、同配置/实际 block 的 shader checksum、表示元数据、完整数值 oracle、输入指纹和输出逐位一致性。不同 block 的 shader 不要求相同：block 常量本身属于编译配置。
- GPU 串行运行，与完整构建不重叠。外层 `caffeinate -di` 的 assertion 有 `pmset` 收据；不宣称桌面没有其他应用活动，也不宣称 GPU 独占。48 visits 全部完成，无 timeout 或 GPU failure 诊断。

旧临时树缺失依赖的失败构建没有修补或掩盖。新树首次完整 `cmake --build` 的真实退出码为0，但误用了 benchmark 的 GPU 日志扫描器：路径 `metal-next/.../error.posix.c` 上两条普通编译 warning 被匹配为 GPU error，包装器因此标记失败。原始日志和失败收据保留。实验脚本将**仅 host 构建/检查**的退出状态与 GPU 诊断分开，再次完整构建确认退出码0；没有放宽任何 native benchmark 的 GPU 故障规则。纯 Python 协议测试40项通过。

## 实际执行几何

`N` 是独立 root programs 数，不是 KV 长度。packet width为32。

| 配置 | 形状 | BQ/BK | N | auto block→32 | threadgroups：auto→32 | 物理 SIMDgroup slots：auto→32 |
|---|---|---|---:|---|---|---|
| small-q4 | attention `1,4,2,16,33,32,32` | 4/16 | 16 | 32→32 | 1→1 | 1→1 |
| small-q1 | attention `1,4,2,16,33,32,32` | 1/16 | 64 | 64→32 | 1→2 | 2→2 |
| prefill-q4 | attention `1,4,2,64,128,64,64` | 4/16 | 64 | 64→32 | 1→2 | 2→2 |
| prefill-q1 | attention `1,4,2,64,128,64,64` | 1/16 | 256 | 256→32 | 1→8 | 8→8 |
| rmsnorm-tail | rows257 × width1025 | — | 257 | 256→32 | 2→9 | **16→9** |
| swiglu-wide-grid | rows8192 × width65 | — | 8192 | 256→32 | 32→256 | 256→256 |

Attention 形状顺序为 `B,Hq,Hkv,Q,K,D,Dv`。small-q4 两臂实际几何相同，是 no-change control。N257 也改变 padding：有用 packets 都是9，但 auto256 多启动7个完全越界的 SIMDgroup slots；这些 slots 做入口检查，不是执行7份完整 kernel。其收益不能单独归因于 group packing。

## 时间与完整负结果

下表的绝对时间是每臂4个 visit 中位数的中位数，单位µs；只用于量级展示。配对比由各 visit 的中位数分别相除，不能用显示的绝对中位数重算。

| 配置 | 插桩吞吐 dispatch：auto µs | 插桩吞吐 dispatch：32 µs | 32/auto 配对比，中位数 [最小,最大] |
|---|---:|---:|---|
| small-q4 | 1811.146 | 1819.969 | 1.019 [0.915,1.138] |
| small-q1 | 756.427 | 788.594 | 1.043 [1.040,1.044] |
| prefill-q4 | 10486.562 | 9689.948 | 0.888 [0.621,1.706] |
| prefill-q1 | 7034.042 | 6631.479 | 0.942 [0.940,0.945] |
| rmsnorm-tail | 295.010 | 244.833 | 0.832 [0.807,0.847] |
| swiglu-wide-grid | 121.104 | 129.010 | 1.065 [0.884,1.219] |

以下全部为**耗时比32/auto**，小于1较快。区间是4个相邻配对的范围，不是置信区间。

| 配置 | 吞吐：插桩 dispatch | 吞吐：feedback-only command buffer | 吞吐：host wall |
|---|---|---|---|
| small-q4 | 1.019 [0.915,1.138] | 1.011 [0.982,1.428] | 0.969 [0.609,1.374] |
| small-q1 | 1.043 [1.040,1.044] | 1.050 [1.009,1.532] | 1.488 [0.618,1.668] |
| prefill-q4 | 0.888 [0.621,1.706] | 0.950 [0.476,1.282] | 0.927 [0.843,1.138] |
| prefill-q1 | 0.942 [0.940,0.945] | 0.933 [0.768,1.620] | 0.719 [0.561,0.865] |
| rmsnorm-tail | 0.832 [0.807,0.847] | 0.827 [0.794,0.909] | 0.728 [0.337,0.850] |
| swiglu-wide-grid | 1.065 [0.884,1.219] | 0.934 [0.148,1.296] | **3.283 [2.982,5.195]** |

| 配置 | Latency：插桩 dispatch | Latency：feedback-only command buffer | Latency：host wall |
|---|---|---|---|
| small-q4 | 0.967 [0.881,1.300] | 0.973 [0.835,1.127] | 1.612 [0.350,2.797] |
| small-q1 | 1.364 [1.250,1.465] | 0.844 [0.760,1.107] | 0.645 [0.334,1.018] |
| prefill-q4 | 0.693 [0.127,1.476] | 0.893 [0.740,1.771] | 0.876 [0.487,1.343] |
| prefill-q1 | 0.947 [0.485,1.516] | 0.950 [0.931,0.975] | **1.352 [0.410,2.070]** |
| rmsnorm-tail | 0.850 [0.566,0.957] | 0.767 [0.695,1.025] | 0.981 [0.646,2.994] |
| swiglu-wide-grid | 1.089 [0.863,1.190] | 0.976 [0.397,1.293] | 1.572 [0.361,4.897] |

计时边界：插桩 dispatch 不包含 host Runtime 编码，但有时间戳插桩影响，**不是零开销 kernel 时间**；feedback-only 控制包含 GPU command-buffer 内的工作/空隙，不是单 kernel；host wall 是同步端到端调用。三者各自独立采样，不能相减得到精确 Runtime 开销，不能用更好看的 control 替代插桩时间验收。固定8次重复不意味着每批都达到工具元数据中的20ms目标。没有新的跨 framework 排名。

可以支持的结论：

1. prefill-q1 插桩吞吐的4个配对都缩短约5.5%–6.0%，但 latency 插桩/control/E2E 的方向并不全面一致，不能称为所有口径稳定加速。small-q1 的插桩吞吐反而增加约4.0%–4.4%。no-change control 本身也有波动，以上只是有限 cohort 的观测，不是统计显著性的证明。
2. RMSNorm 的两种 GPU 吞吐口径均改善约17%，但该点同时减少 padding 与改变 packing，而且固定了 local1；不能外推到所有规约或最佳自动计划。其 host latency 也没有一致改善。
3. prefill-q4 的插桩配对0.621–1.706跨越两种方向，不接受“中位数更快”作为稳定性能结论。SwiGLU 的 host吞吐4对全部变慢，而插桩与control方向不一致。所有这些负结果保留，不拟合一条“block32更快”的规则。
4. 因而本轮没有选择新默认 block，也没有根据这6点给 GPU cost model 填经验常数，更没有证明瓶颈百分比、寄存器 spill、具体 occupancy 或 MPS 性能胜利。

## 对 planner / reduction 的启发

源代码可直接推导当前 auto 排序。令 `P=ceil_div(N,32)`、`G=ceil_div(N,B)`，物理 slots 为 `L=G*B/32`。固定 local1/root order/表示后，Metal policy 的 arithmetic/memory 总工作估计不随B变化，只有正的 `block_dispatch*G` 变化；候选按32/64/128/256升序、同分取首个，所以天然偏向最少 groups。**这是相对工作 prior，不是 GPU 完成时间模型。** N257 中，L−P的额外无效 slots 也没有单独计价。

下一步应分别推进：

- 通用 GPU 成本：保留 P/G/L、每组 packet数、工作量和资源表示；在多种 root-count、block-size 和工作集上校准组服务时间/有效并行能力，目标估计完成时间。部分空组的服务成本应同时依赖实际分配宽度与有效工作，不能把物理宽组当作较窄组。静态 snapshot 不冒充实测寄存器，拟合容量也不冒充 GPU core 数。缺少校准或超出支持范围时明确回退。
- 最小后续校准可选 SwiGLU65 与 RMSNorm1538，N取64/256/257/8192，block取32/64/128/256；保留相同几何 control，attention作为算子/shape留出集。这个32配置集合是**未执行计划**，还须确认覆盖饱和转折；当前auto/32固定N对照不足以唯一识别服务时间和并发容量。
- 通用协作表示：program team 与每个 Tile 值的分布解耦，让输出域与贡献域分别映射。现在 attention 的 local1 把相邻lane分给不同program，不能直接插入 `simd_sum`。先支持 uniform row reduction 的短/尾部有效值树和 replicated/cyclic 转换，再闭合 pipeline carry 与 QK→softmax→PV。详见[映射审查§21.4](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md)。

这些是不同层面的缺口。block实验没有改变 reduction 的算法；较好的 block 也不会自动生成缺失的协作规约或 tensor atom。TIRx 既有 `simd_sum/max` 的事实不变，不能把 XIR/Metal4 的能力边界概括成“所有 Metal 路径都没用 warp intrinsic”。

## 证据与复现

- 原始实验：`/tmp/luisa-metal-block-mapping.eI6JXf`；完整执行由保存的 `probe.py` 和 `plan.json` 描述，原始目录禁止覆盖。
- [离线审计脚本](audit.py)已重新检查全部48 visits 的6种计时（864个样本值）、192个张量收据及完整FP64 oracle；最大绝对误差为 `5.0202757329e-7`，同配置跨block输出逐位一致。不加载原生库、不执行GPU。首个审计尝试因 macOS `/tmp` 与 `/private/tmp` 的同一路径别名失败，修正仅此前缀的路径比较后通过；失败尝试也有独立收据。
- [打包脚本](package.py)保留源快照、Python工具、固定依赖manifest、所有失败/成功构建和采样日志；binary只保存指纹，不包含虚拟环境或完整可独立重建的全部依赖。
- 离线审计/可移植归档结果见本目录 JSON 收据与 `SHA256SUMS`；不得用源码、JSON的自报OK替代对原始数据的核验。
