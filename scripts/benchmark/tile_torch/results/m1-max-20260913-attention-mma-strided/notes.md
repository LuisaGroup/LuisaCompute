# 通用 MMA 输出分组：带步幅的右操作数

2026-09-13，Apple M1 Max，LLVM22.1.8，precise FP32。实现提交 `7164b6d9d`。承接 [indexable snapshot](../m1-max-20260913-attention-mma-indexable/notes.md)，默认R1/cap0不变；仍为固定opt-in候选，不宣称完成自动planner优化或Torch/MPS目标。

## 改动：合法性与访存偏好分开

左操作数沿输出方向广播时，右操作数即使在该方向有步幅，也能为多个独立输出共享一个K循环。共享 `MmaEmissionPlan` 现在允许这一候选；每个输出保持原K次序、类型和MUL→ADD，不引入新的DSL实体或函数名特判。

现有 `mma_output_block` 取R1/2/4。另一种对称情形（右广播、左带步幅）此次仍保留原有fallback，不冒充覆盖全部布局。全局预算、局部分布限制、资源规划不放松。小输出仍保留完整SSA Elements，兼容现有pipeline carry，不改成storage-only输出。

## 六个预声明实验：R1/R4完整结果

每组固定cap与full-packet请求P；12 captures、72 ABBA visits、504 samples。每visit七样本，warmup30 ms、target15 ms。以下为**单线程实际ORC native-entry host-wall**，排除Runtime/Python/JIT/调用方分配/校验，包含入口、launch reset、block遍历及生成helper；不是硬件cycle或GPU时间。

| case | cap | P | R1 µs | R4 µs | 配对 R4/R1 | 六对范围 |
|---|---:|---|---:|---:|---:|---:|
| ragged decode D80 | 8 | on | 2071.242 | 1996.349 | 0.9645 | 0.9426–0.9812 |
| MHA D64 | 0 | off | 1148.225 | 1140.357 | 0.9968 | 0.8815–1.0544 |
| MHA D64 | 8 | off | 1246.421 | 1233.675 | 0.9863 | 0.9404–0.9996 |
| MHA D64 | 8 | on | 1253.268 | 1218.736 | 0.9680 | 0.9459–1.0093 |
| prefill | 0 | on | 141.769 | 135.799 | 0.9570 | 0.9445–0.9662 |
| prefill | 8 | on | 203.868 | 160.084 | 0.7860 | 0.7801–0.7895 |

时间取visit中位数的中位数；比例取六个同cycle配对比的中位数，并非展示时间相除。范围不是置信区间，所有波动/回退样本保留；未控制桌面为静默状态，load averages留档。

Shape顺序 `B,Hq,Hkv,Q,K,D,Dv`：decode=`1,8,2,1,2053,80,96`，block1×16；MHA=`1,8,8,1,2048,64,64`，block1×16；prefill=`1,4,2,32,65,32,32`，block4×16。固定QK/PV=MMA、W8/block32/local1、全局Tile阈值64、region budget4096与math/fusion策略。

## 能得出与不能得出的结论

1. 本轮R4对decode约少3.5%时间，对cap0 prefill约少4.3%；两组的六对比均小于1，但仅是这些预声明配置的有界证据，不推广成全局默认。
2. cap8 prefill约少21.4%，但其160.084µs仍高于cap0 R4的135.799µs。两种cap不是同一轮交错对比，不能把跨组绝对数再算成配对收益；更不能只摘21%当成超越旧最优的成绩。
3. MHA收益很小，两条配置的范围跨1，不能声称稳定明显改善。代码变小或循环变少不等于吞吐同比提升。
4. 本批R1/R4都使用新的compiler；R4同时对QK和PV分组，`blocked_mmas`从0变2。它测的是完整分组选项，不是单独隔离“放宽strided RHS条件”的旧R4/新R4 A/B。
5. 同cap两边snapshot容量、workspace、输入、math/helper一致，所有逻辑输出逐bit相同。物理寄存器/缓存行为没有被证明相同。full-packet实际生成与否由独立LLVM定义检查，不把请求P当作已生成。

## 验证与后续重点

首轮构建在新增host测试的 `Type::of<int64_t>()` 未实例化符号处失败；改用现有 `is_int64()` 后重新完整构建成功。四项CTest全过（140.27 s）：SIMD LLVM codegen、XIR target info、SIMD Runtime、SIMD LLM。不是整仓测试全绿。

host测试在K9/N5/cap8检查实际int64 PHI：R1/R2/R4产生5/3/2个K循环；反向布局负控制保持5个。新增36个转置布局Runtime dispatch，保留之前配置，完整核对严格FP32累积、FMA反例、seed/A的eager alias语义、输入B未变与guards。三个相关TU的clangd/tidy 0 errors（target38、runtime52、lower3 warnings保留）；格式/no-throw通过。

下一结构性重点是phase内部的输出/贡献维lane ownership与QK→softmax→PV之间的转换。现有XIR local分布准入只支持有限的一维程序族，MMA仍在bridge中变成标量循环；仅改善循环展开无法替代这些映射能力。后续应保留typed contraction与数值许可到候选选择阶段，成本同时包含访问表示、循环/代码量、layout转换与资源容量，严格MMA仍须保留ordered fallback。这里是下一步设计方向，不冒充已实现。

原始目录 `/tmp/luisa-attention-mma-strided.sFL4Ov`。源码、对象、完整输入/FP64 oracle/输出、命令、samples与测试记录将以独立证据包留档；生产二进制和工具指纹不等于完整loader闭包。没有新的Metal4/TIRx/Torch/MPS配对性能结论。

## 离线解包复审

归档已完成；在本目录运行下列命令即可复查。需要 Python 和 NumPy，不需要读取当前 build／LLVM 安装，也不会加载 native dylib、重建或运行性能。新建临时目录解包，避免覆盖原始实验。

```sh
shasum -a 256 -c SHA256SUMS
review_dir=$(mktemp -d /tmp/luisa-attention-mma-strided-review.XXXXXX)
tar -xJf evidence.tar.xz -C "$review_dir"
python3 -B "$review_dir/luisa-attention-mma-strided.sFL4Ov/audit.py" \
    "$review_dir/luisa-attention-mma-strided.sFL4Ov" > "$review_dir/audit-replayed.json"
cmp audit.json "$review_dir/audit-replayed.json"
```

归档 QA 已从独立临时目录复审：完整解包目录与原始 raw 的 `diff -qr` 无差异，审计 stdout 与归档 `audit.json` 逐字节一致。审计包含 502 个冻结源码成员、12 个独立 NumPy FP64 oracle、24 份原始/准备输出、72 个 replay 输出及 504 个计时样本；所有六组比较与跨1的范围均保留。

审计检查固定 cap 下的配置、snapshot/workspace 容量、数学 helper、输入与实际 captured ORC object/prepared dylib 的身份，并重算全部样本统计。历史 producer/tool SHA 只作归档 provenance 的内部一致性记录，不虚称独立证明了当时加载的全部二进制或 loader 闭包。guard payload 未单独保存，guard/input-immutability 仍依赖原 C++ 检查 receipts；静态容量一致不证明物理寄存器和缓存行为一致。R1/R4 都来自本版 compiler，本审计不把它扩展成旧/新 R4 的隔离比较。
