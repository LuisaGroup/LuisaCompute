# Native snapshot copy：离线审计结果

数据来自 [audit.json](audit.json)，首次审计通过；完整原始尝试保存在 `audit-attempt-1.json` 和空的 `audit-attempt-1.stderr`。这是同一冻结编译器下 native copy width **0 / 4** 的成对实验；两臂 native MMA width 均为 **4**，不是默认编译路径或 MPS/Torch 对比。

## 性能

单位为单线程 native-entry host-wall µs。Off/On 两列分别为该臂六次访问中位数的中位数；每次访问含七个样本。成对 On/Off 是三个 ABBA 周期产生的六组相邻 AB/BA 访问中位数比率的中位数，**不是两列相除**；小于 1 表示更快。区间仅为六组比率的实测最小–最大值，不是置信区间。

| Case | `(B,H,KH,Q,KV,D,Dv)` | Off µs | On µs | 成对 On/Off | 实测范围 |
|---|---|---:|---:|---:|---:|
| decode-mha-d64 | `(1,8,8,1,2048,64,64)` | 1587.456 | 449.195 | 0.282934 | 0.275330–0.287679 |
| decode-gqa-d80 | `(1,8,2,1,2053,80,96)` | 2840.323 | 602.510 | 0.211910 | 0.209792–0.213251 |
| decode-long-kv | `(1,16,4,1,8193,128,128)` | 24431.750 | 7288.667 | 0.298329 | 0.298254–0.301057 |
| prefill-q4 | `(1,4,2,32,65,32,32)` | 175.938 | 130.576 | 0.741775 | 0.734016–0.743052 |
| prefill-q8 | `(1,4,2,64,129,32,48)` | 489.964 | 502.758 | 1.025900 | 1.016397–1.036926 |
| batch-gqa-q4 | `(2,6,2,17,67,40,48)` | 2367.031 | 1174.733 | 0.496272 | 0.492077–0.497113 |

五组改善；`prefill-q8` 六组成对比率均大于 1，中位数约慢 **2.59%**，保留为负结果。不能据此声称所有形状都获益或默认开启已经合适。

## 审计范围与边界

- 核对 42 个 owned 源文件、685 个源码包成员、最终 11 个 CTest（148.68 秒）、25 个语法检查 TU、90 条命令收据，以及 12 captures / 72 replay visits / 504 samples；历史失败门禁记录仍保留。
- 全部输出通过独立 dense FP64 bottom-right causal GQA oracle；六组 captured A/B 输出逐位相同，每次 replay 最终输出与自身 capture 逐位相同。数值容限为绝对/相对 `5e-5`，不是全输入空间正确性证明。
- A/B 的 root order、task grain、ABI、输入、静态快照字节数和分配数一致；本矩阵实际 workspace 字节数也一致。每臂均有两个 native MMA，copy 静态调用点由 0 增至 3（Q/K/V），不等于动态只执行三次。
- 计时包含公共 C++ launch 遍历/重置及生成的 helper/libc 执行；不含 Runtime dispatch、Python、JIT、调用方分配和验证。oracle/guards 在 preflight、warmup、calibration 及每个计时样本批次之后检查，**不是每次批次内 native invoke 后检查**。
- 比率描述整个 lowering 候选，不能分离解释为 vector load 的净收益。`prefill-q8` 同时少了一个 full-packet specialization / 3786 条 cloned instructions；这只是共变证据，不是已证明的退化原因。
- 背景负载、亲和性和静默主机条件未受控；这些数据不是多线程吞吐、MPS/Torch 排名或跨硬件普适性结论。源码包为实际选定导出，不是完整 HEAD、可重现构建证明或 loader 依赖闭包。
