# Full11 新配对测量结果

四阶段均已终态通过，独占 native CPU 窗口已释放。无重试、参数变更或运行中源码/脚本修改。

| 阶段 | 案例数 | 阶段 wall time |
|---|---:|---:|
| capture | 22/22 passed | 13.725 s |
| prepare | 22/22 passed | 28.054 s |
| full11 vs full9 | 22/22 passed | 104.479 s |
| full11 vs Torch | 8/8 passed | 35.165 s |

## 同一 native_tile timer：full11 / full9

每个 edge 独立执行 3 轮 ABBA，每次 visit 5 个样本：总计 264 visits、1320 个原始 wall 样本、132 个相邻配对比率。下表的时间为 visit median 的 median；比率是 6 个相邻配对比率的 median，不等于两列 median 的简单相除。

| 案例 | local | full9 µs | full11 µs | full11/full9 配对比 |
|---|---:|---:|---:|---:|
| RMS 129×65 | 1 | 6.7262 | 6.7587 | 1.00548 |
| RMS 129×65 | 8 | 12.2160 | 12.2408 | 1.00328 |
| softmax 129×65 | 1 | 28.4580 | 28.3949 | 0.99715 |
| softmax 129×65 | 8 | 42.4750 | 42.5823 | 1.00275 |
| RMS 129×512 | 1 | 48.5376 | 48.6518 | 1.00225 |
| RMS 129×512 | 8 | 56.2486 | 56.4203 | 1.00450 |
| softmax 129×512 | 1 | 225.2804 | 224.1426 | 0.99688 |
| softmax 129×512 | 8 | 211.3231 | 210.8335 | 0.99763 |
| attention33 | 1 | 16.1593 | 16.2119 | 1.00162 |
| attention33 | 8 | 46.9638 | 47.0137 | 0.99861 |
| attention65 | 1 | 85.3224 | 84.9605 | 0.99496 |
| attention65 | 8 | 172.9059 | 172.9201 | 1.00099 |
| RMS 129×1024 | 1 | 99.2299 | 98.8748 | 0.99718 |
| RMS 129×1024 | 8 | 105.6807 | 106.0586 | 1.00217 |
| softmax 129×1024 | 1 | 464.2770 | 465.1185 | 1.00154 |
| softmax 129×1024 | 8 | 414.3571 | 411.8649 | 0.99490 |
| RMS 129×4096 | 1 | 428.7620 | 426.1716 | 0.99823 |
| RMS 129×4096 | 8 | 412.8122 | 411.9642 | 0.99845 |
| softmax 129×4096 | 1 | 1879.9831 | 1882.0261 | 1.00444 |
| softmax 129×4096 | 8 | 1639.6198 | 1641.2266 | 0.99835 |
| attention65-q17 | 1 | 262.2855 | 260.9240 | 0.99377 |
| attention65-q17 | 8 | 787.8672 | 790.1543 | 1.00264 |

比率范围 0.993765–1.005477；没有观察到明显回退，不能称为性能提升，也不是正式统计等价性证明。这里衡量的是 full11 epoch 正确性修复相对保留的 full9 默认实现。

## 另一套共同 native_rows timer：full11 / Torch

8 个独立 fresh ABBA edges，96 visits、480 个原始 wall 样本；另外 16 次有正确性检查的低时长 preflight，其测量全部丢弃。旧版 Torch timing 不参与计算。

| 案例 | local | Torch µs | full11 µs | full11/Torch 配对比 |
|---|---:|---:|---:|---:|
| RMS 129×65 | 1 | 5.6255 | 6.7712 | 1.20475 |
| RMS 129×65 | 8 | 5.6524 | 12.2446 | 2.16098 |
| softmax 129×65 | 1 | 16.8779 | 28.3303 | 1.67897 |
| softmax 129×65 | 8 | 16.7242 | 42.6060 | 2.54977 |
| RMS 129×512 | 1 | 35.4358 | 48.9068 | 1.37991 |
| RMS 129×512 | 8 | 35.6205 | 55.7521 | 1.56571 |
| softmax 129×512 | 1 | 117.1809 | 224.2677 | 1.91545 |
| softmax 129×512 | 8 | 117.1038 | 210.2282 | 1.79138 |

按本次较快 mapping，四种形状仍比 Torch 慢约 20.5%、67.9%、38.0%、79.1%。这些是单线程 native entry wall time，不是 Runtime 端到端吞吐；不能把该表的 Torch 分母用于上一套 timer 的样本。

## 正确性与证据

- 360 个正式 timed visits 全部 valid、guards passed、inputs unchanged；完整输出按独立 FP64 reference 检查，进入每次 visit 前 writable 输出为 NaN。
- Torch 还检查全部外露 writable ABI arguments，包括 RMS 平方和、softmax max/exp-sum scratch；96 visits 共 168 次完整 writable-array oracle 检查。
- 核心 phase JSON：`capture-results.json`、`prepare-results.json`、`replay-results.json`、`torch-results.json`。
- 原始 wall 样本：`replay-output-<case>-l<1|8>/results.json` 与 `torch-output-<case>-l<1|8>/results.json` 中的 `visits[].samples_us`；每次输出也独立存档。
- `joined-admission.json` 绑定 full9-immutable、独立复制的 baseline preparations、full11 source/binaries/gates 快照，以及两个未修改 timer 的哈希。
- 仅 source overlay 与关键 binaries 的身份得到闭合，不声称完整 Git/runtime/Torch 动态依赖归档。编译器内部临时分配不能从外部逐一 guard，但计时包含其开销。
