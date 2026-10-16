# Top-K / sort 首轮筛查：全部 60 个记录

2026-09-13，M1 Max，FP32 values / int64 indices。12 个 shape/K/direction 配置 × 5 条路径，每条仅一轮、3 样本。全部正确性检查通过；**不是顺序平衡的性能排名**。桌面负载和计时扰动明显，不据此报告优化收益。

## 端到端 batch，µs / invocation

包含 Runtime 或 Python/framework dispatch 与同步，排除 JIT/setup；各路预分配输出。CPU 列不是纯 kernel 时间，不能直接与下表 GPU 区间相除。

| Case | XIR SIMD | Torch CPU | XIR Metal4 | TIRx Metal | Torch MPS |
|---|---:|---:|---:|---:|---:|
| topk-1x31x1-ascending | 10.225 | 1.353 | 29655.167 | 3624.125 | 259.267 |
| topk-1x31x1-descending | 10.414 | 1.383 | 29568.167 | 9104.125 | 4536.958 |
| topk-17x65x8-ascending | 89.789 | 5.534 | 19104.666 | 2133.375 | 4690.000 |
| topk-17x65x8-descending | 87.822 | 6.159 | 544.098 | 1593.222 | 574.006 |
| topk-128x257x16-ascending | 6302.417 | 100.209 | 12776.042 | 15937.417 | 5720.604 |
| topk-128x257x16-descending | 6341.645 | 111.688 | 9737.688 | 17392.417 | 5665.708 |
| topk-17x1025x1024-ascending | 42658.500 | 382.511 | 96083.333 | 116620.875 | 387.869 |
| topk-17x1025x1024-descending | 42959.167 | 395.042 | 111857.000 | 142833.792 | 767.424 |
| sort-17x65x65-ascending | 175.714 | 9.591 | 44002.250 | 2730.028 | 266.471 |
| sort-17x65x65-descending | 176.898 | 9.850 | 4279.167 | 2752.028 | 493.724 |
| sort-17x1025x1025-ascending | 40537.125 | 193.243 | 103660.500 | 128739.458 | 2656.500 |
| sort-17x1025x1025-descending | 40925.250 | 184.140 | 108999.083 | 118605.542 | 5570.917 |

## 无计数器 command-buffer GPU batch，µs / invocation

这些 control 包含 GPU work 与 command-buffer 内部间隙，不是孤立 kernel 时间。Metal4 的 instrumented dispatch 与 legacy Metal/Torch 的 compute-pass 探针另外保留在原始 JSON，没有混入本表。

| Case | XIR Metal4 | TIRx Metal | Torch MPS |
|---|---:|---:|---:|
| topk-1x31x1-ascending | 72.958 | 118.396 | 78.983 |
| topk-1x31x1-descending | 145.542 | 121.417 | 98.375 |
| topk-17x65x8-ascending | 178.875 | 149.292 | 183.625 |
| topk-17x65x8-descending | 195.674 | 148.514 | 78.526 |
| topk-128x257x16-ascending | 3029.750 | 13780.583 | 166.625 |
| topk-128x257x16-descending | 3068.354 | 13971.125 | 97.333 |
| topk-17x1025x1024-ascending | 86040.125 | 163637.583 | 182.882 |
| topk-17x1025x1024-descending | 84043.750 | 112519.792 | 85.019 |
| sort-17x65x65-ascending | 530.750 | 352.264 | 81.253 |
| sort-17x65x65-descending | 470.292 | 369.194 | 80.528 |
| sort-17x1025x1025-ascending | 84142.625 | 120557.792 | 104.903 |
| sort-17x1025x1025-descending | 83164.750 | 121150.833 | 103.875 |

全部 single-call latency、instrumented GPU 样本、编译时间和生成源码都保存在 [原始矩阵](matrix.json.gz) 与 `evidence.tar.gz` 中。结论和局限见 [说明](notes.md)。
