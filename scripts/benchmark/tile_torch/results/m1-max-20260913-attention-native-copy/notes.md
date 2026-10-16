# Attention：连续快照复制的执行映射

2026-09-13，Apple M1 Max，macOS 26.6.2，LLVM 22.1.8，FP32。

## 结论

这次是实际 **Tile → XIR → SIMD 编译器优化**：在原 load 定义点，把满足连续布局条件的既有私有快照改为每 program 内连续向量复制。不是手写 attention 替换、函数名称匹配或新的 DSL primitive。

固定 native MMA 开启，复制候选让三组 decode 的配对时间减少 **70.2%–78.8%**，prefill-q4 减少 **25.8%**，batch GQA 减少 **50.4%**。prefill-q8 则增加 **2.6%**，同时失去满包特化。默认仍关闭；这轮没有新的 Torch/MPS/BLAS 对照，也没有证明已经打败上一轮更快的 MMA-off 基线。整体性能目标尚未完成。

| case | B / Hq / Hkv / Q / KV / D / Dv | query×KV block | cap / 2D request | copy off µs | copy on µs | 配对 on/off | 六对范围 | on 较慢对数 |
|---|---|---|---|---:|---:|---:|---|---:|
| decode-mha-d64 | 1 / 8 / 8 / 1 / 2048 / 64 / 64 | 1×16 | 0 / off | 1587.456 | 449.195 | 0.28293 | 0.27533–0.28768 | 0/6 |
| decode-gqa-d80 | 1 / 8 / 2 / 1 / 2053 / 80 / 96 | 1×16 | 8 / off | 2840.323 | 602.510 | 0.21191 | 0.20979–0.21325 | 0/6 |
| decode-long-kv | 1 / 16 / 4 / 1 / 8193 / 128 / 128 | 1×16 | 8 / off | 24431.750 | 7288.667 | 0.29833 | 0.29825–0.30106 | 0/6 |
| prefill-q4 | 1 / 4 / 2 / 32 / 65 / 32 / 32 | 4×16 | 0 / on | 175.938 | 130.576 | 0.74178 | 0.73402–0.74305 | 0/6 |
| prefill-q8 | 1 / 4 / 2 / 64 / 129 / 32 / 48 | 8×16 | 0 / on | 489.964 | 502.758 | 1.02590 | 1.01640–1.03693 | 6/6 |
| batch-gqa-q4 | 2 / 6 / 2 / 17 / 67 / 40 / 48 | 4×16 | 8 / on | 2367.031 | 1174.733 | 0.49627 | 0.49208–0.49711 | 0/6 |

µs 是六次 visit 中位数的中位数；配对比率是六个相邻 ABBA on/off 比率的中位数，不是两个表中时间相除。范围不是置信区间。所有负结果保留。

## 比较合同与正确性

- 两臂固定 native MMA width4、QK/PV MMA、R4、W8、32 workers/block、local1、global Tile threshold64、region budget4096、FP32、global fast math=false；满包特化都请求开启。cap 和2D request按表预先固定。唯一开关是 `LUISA_SIMD_NATIVE_COPY_VECTOR_WIDTH=0/4`。
- 从真实 JIT 捕获 ORC object，再由同一个 C++ 计时器执行；没有重新编译 LLVM 文本来近似实际产物。指标为 `single_thread_native_entry_host_wall_us`：包含 native entry、launch reset、block traversal 和编译器内部工作，排除 Runtime dispatch、Python、JIT、调用者分配与验证。不是硬件周期、GPU计时或端到端 Runtime 延迟。
- 全部12 captures/prepare成功后，才运行每组3轮ABBA、每visit 7个样本，共72 visits / 504 samples。warmup30 ms，样本目标15 ms。源码、生产者和工具指纹在第一次capture前冻结。
- 使用独立 dense FP64 `einsum(optimize=False)` 重建 bottom-right causal GQA，FP32 scale，容差 `5e-5 + 5e-5*abs(reference)`。完整输出、输入不变、分配guards与有限值均检查。每个回放的最终输出必须逐bit等于本臂capture，且六组A/B capture也必须逐bit相同。本批最大绝对误差小于9.8e-8。
- C++ helper 在preflight、warmup、calibration及每个sample batch之后检查完整输出/guards/输入，**不是每次内部native invocation都校验**。计时区间内没有验证，也没有新增数学重结合权限。这只是本批输入与尺寸的证据，不是所有FP32输入的等价证明。
- root order、blocks/task、入口ABI、dispatch及所有输入在每对中一致。copy可能改变helper代码、interleaving、workspace和特化准入；这些被观察而非假定不变。桌面后台活动未被控制，load average从约3.60/3.98/4.78到4.82/4.22/4.85；未停止用户应用，没有同时构建、测试或profile，不做细小成本系数标定。

## 资源与结构性结论

| case | snapshot B/worker（两臂） | allocations（两臂） | interleaved arrays（两臂） | workspace B（两臂） | full-packet clone off→on |
|---|---:|---:|---:|---:|---:|
| decode-mha-d64 | 9216 | 9 | 1 | 73728 | 0→0 |
| decode-gqa-d80 | 13376 | 11 | 3 | 107008 | 1→1 |
| decode-long-kv | 19200 | 11 | 3 | 153600 | 1→1 |
| prefill-q4 | 7712 | 13 | 5 | 0 | 0→0 |
| prefill-q8 | 14768 | 18 | 8 | 118144 | 1→0 |
| batch-gqa-q4 | 10400 | 13 | 5 | 83200 | 0→0 |

六对容量、分配数、交错数组数和workspace都相同。workspace=0不意味着没有私有栈状态；这些也不是物理寄存器或DRAM流量。每个on kernel有Q/K/V三处静态native copy；K/V位于循环中，不能把三处当作三次动态调用。

之前[原生采样](../m1-max-20260913-attention-phase-profile/notes.md)显示K/V搬运占MHA-on约84.9%、long-KV-on约81.5%的raw timer observations。本次实测支持优先优化搬运映射，但采样占比不是可直接消除的时间，也不是带宽计数器。

真实ORC反汇编确认MHA/long的K/V变成16-byte步长的 `ldr q` / `str q` 循环，helper已内联，没有copy runtime callback；8193最后一块仍保留逐元素读取/置零fallback。详见evidence中的 `copy-code-review.md` 和30条工具命令收据，不能将优化前LLVM的helper存在误认为最终调用开销。

prefill-q8 的原始entry从3786条LLVM指令增至4146，超过冻结代码的4096门槛，因此copy-on不再生成full-packet clone；GQA/long的clone指令数则从2369增加到2608。这说明guard/fallback与typed copy会改变特化前代码预算，即使逻辑工作、容量和root hierarchy相同。其2.6%回退不能简单解释成向量复制本身慢；量化特化损失需要新的独立控制实验。

对planner/solver的启发是联合选择，而非只给vector copy一个固定折扣：

```text
candidate = (transfer mapping, compute mapping, physical layout, specialization)
cost      = compute + transfer service + transitions + live-state/code effects
subject to semantic legality, target capacity and actual specialization budget
```

实现已经把候选能力以及每program的调用数、full-path向量组、scalar tail与fallback元素提供给backend policy。fast/fallback是互斥路径，不能把二者相加当动态cycles；logical bytes与capacity也不能重复收费。默认 `native_copy_cost=unmodeled`，未拟合自动选择规则。下一步需要联合copy/MMA/满包候选、重新匹配此前MMA-off基线，再扩大prefill和非attention留出集；不能从这六组推断所有算子已泛化。

## 实现与回归

首版只处理静态FP32、local1、已经需要snapshot的完整连续view。非单位tile轴的row-major stride必须与源相同；在原load处逐轴检查整个逻辑view有效，满足才走连续复制，否则保留原逐元素bounds/fill。不是用flatten后的buffer容量代替逻辑行边界。producer/reduction fusion优先，不增加原本无需的snapshot。

必需的 `ContiguousCopyMD` 附在typed XIR external declaration；名字仅供调试。签名为typed buffer resource、uint64元素offset、完整root-local FP32数组reference。clone、text、bitcode、verifier与拥有型Schedule副本保留语义；普通external call、混合metadata、错误类型/容量/存储/placement均拒绝。SIMD backend capability首版接受2/4/8，不是hierarchy lane上限；其他backend默认不接受非零请求。

LLVM在同module生成private helper，以i32位模式做vector full chunks及精确scalar tail。inactive program不读用户地址，尾部不越读，snapshot在定义点完成。输入可与其他用户参数别名，不全局添加noalias/readonly承诺。它沿用合法buffer-read区间前提，不为非法宿主指针编造零值。

完整选定构建 `full-build-3` 成功后，11项CTest全过（148.68秒），25个C++ TU syntax全过。新测试含18组复制JIT配置的位模式、非前缀active mask、空cohort、保护页尾部与workspace guards；24次Tile runtime编译/48次dispatch覆盖负origin、跨行、ragged tail、输入别名/覆写后snapshot、小Tile carry；另有40组几何/资源对照及非法metadata/Schedule边界测试。不是全仓所有测试通过的声明，Metal/GPU没有新的执行结果。

失败尝试原样保留：`full-build-1`遇到Boost.UT不能打印Usage vector；修正后`full-build-2`通过。`regression-1`为9/11：一个malformed-input测试在公共factory提前assert，另一个fusion测试漏计动态输出map已有的260 B snapshot；修正测试夹具后重建，再得到11/11。`syntax-1-6`因Python缺orjson失败；后续环境修正后的三代完整检查通过。没有重写旧失败日志或丢弃慢的性能行。

性能捕获之后另补跨后端拒绝边界：Metal4直接XIR入口不得通过同名native_include绕过必需copy/MMA语义，HIP/CUDA也显式拒绝未实现的typed calls。新增host-only preflight测试覆盖16个普通/required、有/无native_include、live/unused声明组合；完整重建后该CTest及两个相关TU syntax通过，不使用GPU。其补丁源和包括首次block-size夹具失败在内的日志单独存入evidence的 `boundary/`，不修改最初42文件或性能样本。CUDA/HIP在本机配置中关闭，只做源码审查，不声称构建/设备验证。

## 证据与复核

`sources.tar.gz`在第一次capture前冻结，含685个源文件/构建配置/门禁成员；42个owned源文件在主树、实际选定导出和成功门禁间一致。构建来自 `/tmp/luisa-next-integration.roZbN8/source`，不是整个主树HEAD重建，未混入其他人的TIRx WIP。binary/tool指纹不是完整动态loader依赖闭包或可复现构建证明；Python launcher仅作为命令收据。

独立离线审计首轮通过：完整FP64/bit checks、所有统计、源码成员、11CTest、25TU最终syntax及90条capture/prepare/replay命令都重新核对。`table.md`是从审计结果生成的简表。`evidence.tar.xz`包含原始输入输出、对象/LLVM、全部72visits、脚本、命令日志与历史门禁；`sources.tar.gz`单独保存以避免重复归档。文件集合与哈希见 `package-inventory.json` / `SHA256SUMS`。

离线复核不加载native代码：将evidence安全解包，把顶层sources.tar.gz复制到解出的 `luisa-attention-native-copy.n6RmGZ/`，再使用带NumPy的Python运行其中 `audit.py`。它只读归档内容，不依赖原来的临时构建目录。实际机器码检查与命令收据也保存在evidence中；不把静态指令数当作时间占比。
