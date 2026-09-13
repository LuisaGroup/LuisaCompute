# MMA 动态读取：联合规划循环与 definition snapshot

2026-09-13，Apple M1 Max，LLVM22.1.8，precise FP32。实现检查点 `b74264bab`；这是 [独立 cap 实验](../m1-max-20260913-attention-mma-roll/notes.md) 的后续，不覆盖旧数据。默认 cap0 不变，尚未成为自动成本模型优化。

## 改动与语义边界

新 cap 使原本展开的 K 变成运行时循环时，小 SSA 操作数也需要合适的动态索引表示。共享 `mma_contraction_plan` 同时供 emitter 和 allocation/resource plan 使用：对直接 MMA 输入0/1的新增动态读取，在原 SSA 定义处生成可索引数组；不在消费时重新读用户 memory，不把 seed 算作贡献维输入。

既有 snapshot 不重复分配，常量与 lazy/splat 策略保持；不递归强制 materialize 整个表达式图。空输出、零K、默认cap0、既有大K循环和显式全展开诊断均不改变。资源上限先准入，后调用后端cost policy。**这是循环与存储表示的联合变化，不能再宣称容量相同。**

MHA 新增 Q 的64个FP32元素：每worker 8320→8576 B，4→5个snapshot；W8 workspace 66560→68608 B。其余本批case容量未变。规则由 typed MMA domain/use 分析决定，不匹配 attention 名字，也无新DSL实体。

## 六个预声明配置：全部 native-entry 结果

与前批相同六个配置、每个cap0/8：12 captures、72 ABBA visits、504 samples。每visit 7 samples，warmup30 ms、target15 ms。单线程直接运行实际ORC对象；排除Runtime/Python/JIT/调用方分配与校验，计入native入口、launch reset、block遍历与生成helper。不是硬件cycle时间，也不是Torch/MPS对照。

| case | P | R | cap0 µs | cap8 µs | 配对 cap8/cap0 | 六对范围 |
|---|---|---:|---:|---:|---:|---:|
| decode D80 | off | 4 | 3111.406 | 3162.385 | 1.0155 | 1.0132–1.0227 |
| decode D80 | on | 4 | 3140.568 | 2022.326 | 0.6462 | 0.6384–0.6544 |
| MHA D64 | off | 1 | 1185.439 | 1274.380 | 1.0672 | 1.0475–1.0994 |
| MHA D64 | on | 1 | 1254.924 | 1363.894 | 1.0980 | 1.0695–1.1834 |
| prefill | on | 1 | 141.771 | 203.468 | 1.4349 | 1.2436–1.4437 |
| decode tail-only | on | 4 | 2947.591 | 2750.221 | 0.9310 | 0.9265–0.9452 |

时间是visit中位数的中位数；比例另取六个配对比的中位数，不能用表格时间相除替代。范围不是置信区间。桌面活动未静默控制，load averages保留，波动样本不删。

Shape顺序 `B,Hq,Hkv,Q,K,D,Dv`：decode=`1,8,2,1,2053,80,96`；MHA=`1,8,8,1,2048,64,64`；prefill=`1,4,2,32,65,32,32`；tail-only=`1,6,2,1,2053,80,96`。Prefill block4×16，其余1×16。QK/PV=MMA、W8/block32/local1、全局Tile阈值64、region budget4096与math/fusion配置固定。

## 结论：修正严重退化，尚不是全面提速

MHA 在前批cap8中比同批默认慢约4.8倍；本批只慢约7%–10%。两个版本未在同一轮交错运行，因此不把两批绝对时间的比值冒充配对V1/V2收益。新可索引表示消除了一个明确的动态读取问题，但仍没超越本批cap0。

Prefill约43%的退化仍在，不能只靠snapshot解决。小输出展开时，MHA的QK/PV分别产生16/64组K循环；`rolled_mmas=2`只计两个MMA操作，不是两个实际循环。后续可由已有输出分块统一组织多个独立输出，保留逐输出K顺序；不能因右操作数的输出方向不连续，就把这种合法分组排除在候选之外。访存步幅应影响成本，未必是合法性限制。

Decode R4满包路径约36%的改善依旧是相对较慢的R4 baseline，之前R1满包已经约2 ms；未证明刷新最优成绩。既有默认保持，无新Metal4性能胜负、无Torch/MPS达标结论。

## 验证与证据

完整构建通过；四项CTest全通过（134.22 s）：SIMD LLVM codegen、XIR target info、SIMD Runtime、SIMD LLM。host测试核对真实alloca与资源分析一致、215/216 B准入边界、复用/carry/constant/lazy，以及动态输入SELECT链消除。既有严格FMA/alias/guard Runtime测试继续通过。不是整仓测试全绿。

三个直接相关translation units的clangd/tidy为0 errors（test38、lower3、planner16 warnings原样保留）；格式/no-throw通过。实际隔离源码与生产二进制指纹冻结于 `sources.tar.gz`/`provenance.json`，不混入受保护TIRx WIP和远端19个commit。原始目录 `/tmp/luisa-attention-mma-indexable.w2qP2u`；后续审计及完整归档单独记录，不读取后来改变的live build。

### 离线审计与复查

`audit.py` / `audit.json` 保留独立FP64全量oracle、12份capture、24次capture output检查、72次ABBA replay output检查、504个计时sample，以及502个冻结源码成员的身份核验。逐份核对真实LLVM的snapshot数组（workspace区间或显式stack alloca）、planner bytes/allocations和各自replay workspace ABI；允许且精确验证资源增量，不沿用V1的容量相等假设。MHA只增加Q64的256 B/worker；P已有indexable snapshot，未重复分配。

`v1-comparison.json` 独立重算两批原始LLVM/实际ORC对象与prepared副本的hash：六个cap0均逐字节相同；cap8只有两条MHA不同，decode/prefill/tail四条仍相同。此身份对照不比较跨批计时，重新验证V1字节需同时取得前批证据包。

`evidence.tar.xz` 保留全部原始capture/prepared/replay目录、输入与输出payload、期望结果、对象/LLVM/汇编、dylib/helper、命令与日志，以及构建/测试记录；源码归档单列。生产者和工具可执行文件只保留指纹，不宣称完整可重建的编译器/loader闭包。审计不执行对象、不重新link、不读取live build；guard与输入未变的额外保证仍来自保留的C++执行回执。

从仓库根目录复查（Python需要NumPy，脚本路径按本机Python安装调整）：

```sh
checkpoint="$PWD/scripts/benchmark/tile_torch/results/m1-max-20260913-attention-mma-indexable"
(cd "$checkpoint" && shasum -a 256 -c SHA256SUMS)
verification_dir="$(mktemp -d /tmp/luisa-attention-mma-indexable-verify.XXXXXX)"
tar -xJf "$checkpoint/evidence.tar.xz" -C "$verification_dir"
cp "$checkpoint/audit.py" "$checkpoint/provenance.json" "$checkpoint/sources.tar.gz" "$verification_dir/"
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -B "$verification_dir/audit.py" "$verification_dir" > "$verification_dir/rechecked.json"
cmp "$verification_dir/rechecked.json" "$checkpoint/audit.json"
```
