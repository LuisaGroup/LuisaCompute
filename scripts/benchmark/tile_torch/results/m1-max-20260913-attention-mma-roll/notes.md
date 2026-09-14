# MMA K 展开与资源规划：独立 cap 的正负实验

2026-09-13，Apple M1 Max，LLVM 22.1.8，FP32 precise。实现检查点 `cb2219e50`；实际隔离源码、对象和输入单独冻结。**本批仍是 opt-in 诊断，不改变默认策略，也不宣称达到 Torch/MPS 性能目标。**

## 为什么分离 MMA 展开阈值

原实现由 `max_unrolled_tile_elements=64` 同时控制 Tile 遍历／snapshot 与 MMA 贡献维展开。于是 D=64 的 QK 全展开，D=80 却保留循环；R4 的 K=16 PV 也会扩大函数，超过已有 full-packet clone 上限 4096。

新增 `max_unrolled_mma_terms` 是额外上限：默认 0 继承原策略，非零只允许更早保留 MMA K 循环，不会重新展开原来已 rolled 的大 K。显式全展开诊断 `max_unrolled_tile_elements=0` 仍优先。共享 `MmaEmissionPlan` 驱动 R1／R2／R4 的 K 循环以及寄存器分块工作预算；不改逐输出 K 顺序、MUL→ADD、其他 reduction 或本批 snapshot 分配。backend policy 能看到固定请求，但成本仍标为 `unmodeled`，没有自动搜索或折扣系数。

## 完整纯 native-entry 结果

六个预先列出的实验，固定 cap=0／8；共 12 captures、72 visits、504 samples。每实验三轮 ABBA，每 visit 七个样本，warmup 30 ms、sample target 15 ms。单 CPU 线程直接重放实际 ORC 对象，排除 Runtime、Python、JIT、线程池、调用方分配、数据拷贝与数值校验；计入入口、launch-record reset、block 遍历及生成代码内部调用。它是 native-entry host-wall，不是硬件 cycle counter。

所有实验固定 QK/PV=MMA、W8/block32/local1、全局 Tile 阈值64、region budget4096、math 和 fusion 开关。P 表示请求 full-packet 特化，不能替代实际 clone 计数。时间为 visit 中位数的中位数，比例为六个配对比例的中位数，不能直接用表格时间相除替代。

| case | P | R | cap0 µs | cap8 µs | 配对 cap8/cap0 | 六对范围 |
|---|---|---:|---:|---:|---:|---:|
| decode D80 | off | 4 | 3142.375 | 3200.237 | 1.0173 | 1.0105–1.0261 |
| decode D80 | on | 4 | 3148.346 | 2027.893 | 0.6439 | 0.6413–0.6487 |
| MHA D64 | off | 1 | 1214.814 | 5896.448 | 4.8681 | 4.7180–5.0760 |
| MHA D64 | on | 1 | 1219.288 | 5871.865 | 4.8233 | 4.6714–4.8980 |
| prefill | on | 1 | 141.845 | 202.846 | 1.4309 | 1.4277–1.4349 |
| decode tail-only | on | 4 | 2957.391 | 2764.857 | 0.9368 | 0.9265–0.9380 |

Shape 顺序 `B,Hq,Hkv,Q,K,D,Dv`：decode=`1,8,2,1,2053,80,96`；MHA=`1,8,8,1,2048,64,64`；prefill=`1,4,2,32,65,32,32`；tail-only=`1,6,2,1,2053,80,96`。Prefill block=4×16，其余1×16。

## 不是“代码小就快”

Decode R4 在 cap8 下生成 3195 条指令的 full-packet body，恢复了被旧 4560 条 body 拒绝的 clone；P-on 耗时约降36%，P-off却约慢1.7%。六个 head 的 tail-only case没有满包，约6.3%的改善不能解释为执行了满包 clone。

MHA 反而退化到约4.8倍，prefill约慢43%。独立 cap 保持了旧 snapshot，动态 K 投影因而可能落入 `_read` 对小 SSA Tile 的逐元素 SELECT 链；同时输出域仍可能完全展开出许多循环。实际 Schedule blocks：MHA 16→256，prefill 56→251；direct CFG 仍为 true。**这些是代码形态证据，不是动态 profile 对每项耗时的独立归因。**

不能把全局 Tile 阈值一起调低来隐藏交互，也不能全局打开新 cap。下一步应在统一 representation/resource plan 中，让新增的动态贡献索引获得可索引 snapshot；再与 cap0、保留 SSA 的 cap8 分别比较。这将改变资源分配，属于新实验版本，不能覆盖本批数据。

## 验证

完整构建成功；首轮仅因新增测试缺 `PhiInst` 头文件失败，修正后重新完整构建。四项 CTest 全通过（139.64 s）：SIMD LLVM codegen、XIR target info、SIMD Runtime、SIMD LLM。新增36个 Runtime dispatch覆盖零K、阈值两侧、动态读取、alias/snapshot、FMA反例及逐 bit oracle；host测试核对真实 alloca、资源分析和planner一致、默认／诊断不变、backend policy接收配置。不是整仓测试全绿。

七个变更 C++ translation units 的 clangd/tidy 无error，warning保留；格式和no-throw检查通过。另补紧凑拼写 `GPUHangError` 的日志检测，24个focused Python测试通过，包括真实exit0子进程的stdout/stderr故障与正常JSON负例。

## GPU 执行诊断，不能混作性能胜负

在本候选构建前，三轮独立MPS进程完成27个empty-sync/copy/tiny步骤；旧 `15a9b475c` 构建的Metal4 GELU与小attention也通过完整oracle。小TIRx/Torch attention的四次运行无GPU错误日志、数值检查通过，但E2E出现跨visit的大幅迟滞：有的样本为几十微秒，另一些到数百毫秒甚至秒级；未插encoder probe的GPU command-buffer control仍为19–35µs。

因此 `gpu-acceptance.json` 明确拒绝把这组结果用于稳定性能比较或cost校准。健康检查只证明这几次小任务完成，不证明整个设备恢复稳定；之前的GPU hang cohort仍然无效。GPU control是command-buffer区间，不是纯kernel时间。现有GPU runner清理了完整tensor，仅保留oracle receipts、hashes及日志；不冒充可离线数值重放包。

GPU父进程使用的旧 `compare_llm.py` 冻结于 `preflight-sources/`，hash匹配原metadata；后续唯一的compact诊断正则修复不能被标成这些GPU visits的生成版本。

## 归档与复查

`sources.tar.gz`／`provenance.json` 冻结隔离生产源码、构建配置和二进制指纹，不含远端19个commit或受保护TIRx WIP。`evidence.tar.xz` 保存真实对象、prepared dylib／helper、完整CPU输入／FP64 oracle／输出、samples、命令、测试／语法日志和GPU诊断。`audit.py`／`audit.json` 提供独立身份、数值及配对计算检查；`SHA256SUMS` 校验归档。

原始路径 `/tmp/luisa-attention-mma-roll.LHgRyP`；`run.py` 分 capture/replay 两阶段，复现应改用新ROOT/BUILD/OUT，不能覆盖已有结果。桌面共活动未控制到静默状态，load averages留在原始record；六对范围不是置信区间。CPU完整逻辑输出可复查，guard payload未单独保存，guard通过来自实际执行记录。

### 只读解包复审命令与证据边界

在本归档目录执行下列命令。依赖 Python 和 NumPy；无需加载 native dylib、运行 GPU、重新编译或访问当前 build／LLVM 安装。临时解包目录不会覆盖原始实验。

```sh
shasum -a 256 -c SHA256SUMS
review_dir=$(mktemp -d /tmp/luisa-attention-mma-roll-review.XXXXXX)
tar -xJf evidence.tar.xz -C "$review_dir"
python3 -B "$review_dir/luisa-attention-mma-roll.LHgRyP/audit.py" \
    "$review_dir/luisa-attention-mma-roll.LHgRyP" > "$review_dir/audit-replayed.json"
cmp audit.json "$review_dir/audit-replayed.json"
```

审计验证归档内源码、真实 ORC object、prepared dylib/helper 和所有 CPU 逻辑输出的 hash 与数值。历史 producer 与 LLVM 工具的 SHA 是 provenance 记录：只检查归档内关联与一致性，不冒充独立验证了当时加载的全部二进制，也不会读取后来重建的 live build。GPU 详细边界见 `gpu-audit.md`；CPU 审计不追认 GPU 的性能稳定性。`sources.tar.gz` 的 502 个冻结源文件逐项校验；所有六个预声明比较及其不利结果均保留。
