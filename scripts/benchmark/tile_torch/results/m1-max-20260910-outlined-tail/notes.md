# 将 SIMD 尾部代码外提没有带来普遍收益

2026-09-10 的 24-case 实验不支持启用这个候选。两侧都保留 pointwise fusion 和 full-packet specialization，只改变真实窄尾部调用上的 `NoInline`。多数变化很小；17×66 RoPE 六轮全部变慢，配对中位数回退 **4.62%**（范围 4.17%–4.75%）。生产代码中的实验开关和对应结构测试已撤回，测试过的源码和补丁仍完整归档。

## 结果与含义

完整结果见 [24-case 表](tables.md)。小 RoPE 的 batch wrapper 栈帧从 2368 B 减为 112 B，但保留的 full-packet helper 仍为 304 B；候选另有一个 2160 B 的通用尾部 helper。实际机器码变小、热 wrapper 栈帧变小，都不能单独作为收益的代理指标。这里没有 cycle sampling 或指令级耗时归因，不能据此断言回退由分支或 cache 引起。

1024×4098 RoPE 的 on/off 配对中位数为 0.9978，但轮间范围为 0.9925–1.1716，不应写成稳定的大尺寸收益。RMSNorm 1024×4097 的两列汇总时间与配对比值方向不同，原因是先按轮配对再汇总；不应把跨轮中位数之比替代配对统计。

对模型的启发是区分 **尾部实际执行频率** 与 **冷代码对 wrapper 的静态影响**。在固定 local_lanes=W 的这批映射中，外层 dispatch 已按 W 扩展，行宽不整齐并不意味着每个 packet batch 都执行窄尾部。将来应在 SIMD 后端的 packet realization policy 中结合实际 body、workspace、代码大小和 call boundary 评估这些选择，而不是给 Tile 算子加硬编码规则。本轮没有新 planner/cost-model 改进落地。

## 测量口径

- M1 Max，LLVM 22.1.8，Torch 2.14.0；6 个算子 × 4 个尺寸。固定 W8、local8、block32，1 个 CPU worker，fast math 关闭。其他 fusion 开关记录在 [provenance.json](provenance.json)。
- 共 432 个访问：24 cases × 6 种 off/on/Inductor 顺序 × 3 个实现；每次 7 个样本，warmup 100 ms，目标 sample 30 ms。
- 指标为 `single_thread_native_entry_host_wall_us`：共同的 C++ 计时器调用实际原生入口，排除 Runtime、Python、JIT 和调用方分配；包含入口遍历、reset、编译器生成的 libc 调用和内部 allocation。不是硬件 cycle 计数。
- 记录了桌面后台负载，不能声称是安静机器。结果仅针对固定候选，不代表默认 planner，也不能与其他日期、LLVM 版本或不同计时口径拼成一张排行榜。
- 全部 24 个 off/on object 均不同；全部 off/on 输出逐位相同。运行时检查了完整 FP64 oracle、输入不变和 guards。独立 auditor 复核 1413 项 identity、72 个输出、24 个比较契约，并拒绝 11 种统计篡改和 8 种比较契约篡改。

## 可追溯性与保留的失败

[capture.json.gz](capture.json.gz)、[replay.json.gz](replay.json.gz)、[audit.json](audit.json) 保留原始字段与路径。[replay-failed.json.gz](replay-failed.json.gz) 是首次使用缺少 Torch 的 Python 导致的错误：0 次计时访问，不是 kernel 错误，不进入性能汇总。

[candidate.patch](candidate.patch) 只含这次被拒绝的两个 C++ 文件差异。`validation.tar.gz` 含测试过的源码、日志、runner 和命令；`evidence.tar.gz` 含实际 LLVM、ORC object/assembly、入口 dylib、Inductor 产物和共同计时器。冻结脚本、inventory、provenance 和 SHA256SUMS 记录逐字校验方法。

归档有意不含 360 个 tensor 文件（约 1.52 GB）及整个工具链，所以 **不是自包含的离线数值重放包**。原始输出在审计时被重新读取；归档阶段验证的是保存的文件身份，不能用哈希代替缺失的输出内容。重新生成并重跑必须建立新实验记录。
