# Attention joint copy/MMA：完整审计表

由 `package.py --render-table-only RAW` 从两份已通过的离线审计 JSON 生成；不重跑计时、不选最快臂。解释与限制见 [notes.md](notes.md)。

## 六个 case × 五条配对边

下表为 candidate/baseline 的六个相邻 AB/BA 配对比值的中位数。**小于 1 表示候选耗时更短**。每格属于独立 cohort，不能把跨格比值相乘或反推未测过的直接比较。

| Case | Copy，M0 | Copy，M4 | MMA，C0 | MMA，C4 | Simplified，M4C4 |
|---|---:|---:|---:|---:|---:|
| decode-mha-d64 | 0.6138 | 0.2850 | 1.5348 | 0.7175 | 0.9998 |
| decode-gqa-d80 | 0.3965 | 0.2132 | 1.4690 | 0.7901 | 0.9984 |
| decode-long-kv | 0.6562 | 0.2998 | 1.6058 | 0.7341 | 0.9906 |
| prefill-q4 | 0.8645 | 0.7422 | 1.4203 | 1.2224 | 0.7413 |
| prefill-q8 | 0.8846 | 1.0262 | 1.0637 | 1.2333 | 0.8693 |
| batch-gqa-q4 | 0.6287 | 0.4963 | 0.7411 | 0.5847 | 0.9953 |

## 全部 30 条边

时间单位 µs。每臂时间是六个 visit 中位数的中位数；每个 visit 有七个样本。比值是六个配对比值的中位数，不一定等于表中两列时间相除。范围是六对的实际 min–max，**不是置信区间**。

| Case | 边：baseline → candidate | Baseline µs | Candidate µs | 配对比值 | 配对 min–max |
|---|---|---:|---:|---:|---:|
| decode-mha-d64 | m0-c0 → m0-c4 | 1030.825 | 631.350 | 0.6138 | 0.5666–0.6150 |
| decode-mha-d64 | m4-c0 → m4-c4 | 1587.901 | 455.524 | 0.2850 | 0.2738–0.2924 |
| decode-mha-d64 | m0-c0 → m4-c0 | 1001.322 | 1537.763 | 1.5348 | 1.4718–1.5861 |
| decode-mha-d64 | m0-c4 → m4-c4 | 626.794 | 449.861 | 0.7175 | 0.7098–0.7403 |
| decode-mha-d64 | m4-c4 → m4-c4-simplified | 455.442 | 455.396 | 0.9998 | 0.9902–1.0726 |
| decode-gqa-d80 | m0-c0 → m0-c4 | 1934.174 | 767.251 | 0.3965 | 0.3960–0.3971 |
| decode-gqa-d80 | m4-c0 → m4-c4 | 2842.990 | 609.835 | 0.2132 | 0.2128–0.2155 |
| decode-gqa-d80 | m0-c0 → m4-c0 | 1933.578 | 2839.495 | 1.4690 | 1.4485–1.4867 |
| decode-gqa-d80 | m0-c4 → m4-c4 | 765.542 | 604.775 | 0.7901 | 0.7883–0.7973 |
| decode-gqa-d80 | m4-c4 → m4-c4-simplified | 604.551 | 603.644 | 0.9984 | 0.9974–1.0000 |
| decode-long-kv | m0-c0 → m0-c4 | 15220.791 | 10032.458 | 0.6562 | 0.6534–0.6622 |
| decode-long-kv | m4-c0 → m4-c4 | 24423.375 | 7326.484 | 0.2998 | 0.2966–0.3003 |
| decode-long-kv | m0-c0 → m4-c0 | 15399.167 | 24600.230 | 1.6058 | 1.5856–1.6225 |
| decode-long-kv | m0-c4 → m4-c4 | 9967.739 | 7314.120 | 0.7341 | 0.7286–0.7386 |
| decode-long-kv | m4-c4 → m4-c4-simplified | 7445.932 | 7331.422 | 0.9906 | 0.9760–1.0056 |
| prefill-q4 | m0-c0 → m0-c4 | 123.886 | 107.023 | 0.8645 | 0.8636–0.8831 |
| prefill-q4 | m4-c0 → m4-c4 | 175.957 | 130.715 | 0.7422 | 0.7324–0.7512 |
| prefill-q4 | m0-c0 → m4-c0 | 123.773 | 175.808 | 1.4203 | 1.4192–1.4221 |
| prefill-q4 | m0-c4 → m4-c4 | 107.265 | 131.431 | 1.2224 | 1.2206–1.2279 |
| prefill-q4 | m4-c4 → m4-c4-simplified | 130.796 | 97.581 | 0.7413 | 0.7386–0.7481 |
| prefill-q8 | m0-c0 → m0-c4 | 460.146 | 407.097 | 0.8846 | 0.8753–0.8868 |
| prefill-q8 | m4-c0 → m4-c4 | 495.537 | 508.561 | 1.0262 | 1.0191–1.0365 |
| prefill-q8 | m0-c0 → m4-c0 | 460.693 | 490.661 | 1.0637 | 1.0554–1.0764 |
| prefill-q8 | m0-c4 → m4-c4 | 407.479 | 502.574 | 1.2333 | 1.2199–1.2343 |
| prefill-q8 | m4-c4 → m4-c4-simplified | 503.183 | 437.509 | 0.8693 | 0.8591–0.8817 |
| batch-gqa-q4 | m0-c0 → m0-c4 | 3196.974 | 2009.763 | 0.6287 | 0.6209–0.6370 |
| batch-gqa-q4 | m4-c0 → m4-c4 | 2368.154 | 1175.400 | 0.4963 | 0.4959–0.5024 |
| batch-gqa-q4 | m0-c0 → m4-c0 | 3200.859 | 2367.211 | 0.7411 | 0.7379–0.7416 |
| batch-gqa-q4 | m0-c4 → m4-c4 | 2009.893 | 1174.861 | 0.5847 | 0.5779–0.5849 |
| batch-gqa-q4 | m4-c4 → m4-c4-simplified | 1180.625 | 1180.215 | 0.9953 | 0.9903–1.0090 |

## 固定 Tile 臂与 NEON / Accelerate 的六条补充对照

Tile 始终固定为预声明的 `m4-c4-simplified`，没有按 case 挑选最快臂。这里比值方向为 **reference/Tile，小于 1 表示 Tile 更慢**。补充 cohort 与上面的优化矩阵独立，不能拼接时间或相乘速度比。

| Case | Reference | Tile µs | Reference µs | Reference / Tile | 配对 min–max | 最大绝对误差 |
|---|---|---:|---:|---:|---:|---:|
| decode-mha-d64 | online_neon | 449.779 | 195.319 | 0.4346 | 0.4168–0.4437 | 1.175e-08 |
| decode-mha-d64 | dense_accelerate | 446.027 | 183.004 | 0.4060 | 0.4047–0.4235 | 1.152e-08 |
| decode-long-kv | online_neon | 7316.985 | 3597.167 | 0.4886 | 0.4869–0.4950 | 8.798e-09 |
| decode-long-kv | dense_accelerate | 7320.318 | 2475.352 | 0.3378 | 0.3375–0.3387 | 8.798e-09 |
| prefill-q8 | online_neon | 437.664 | 213.373 | 0.4875 | 0.4819–0.4922 | 8.261e-08 |
| prefill-q8 | dense_accelerate | 439.786 | 104.154 | 0.2369 | 0.2348–0.2381 | 1.602e-07 |

## 五臂实际资源与 full-packet 准入

Snapshot 单位为每个逻辑 worker 的字节，allocation / interleaved 均为数量；workspace 是所捕获入口的字节需求。clone 列记录实际生成数量，不用 requested flag 代替。Source/candidate 是准入用 LLVM 指令数（目标 O1/O2 前），不是机器指令数；`ineligible` 的 0 不代表函数为空。

| Case | 臂 | Snapshot B / alloc | Interleaved | Workspace B | ABI | Clone | Source → candidate | 准入结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| decode-mha-d64 | m0-c0 | 8320 / 4 | 4 | 66560 | 0 | 0 | 42529 → 0 | source_budget_exceeded |
| decode-mha-d64 | m0-c4 | 8320 / 4 | 4 | 66560 | 0 | 0 | 41527 → 0 | source_budget_exceeded |
| decode-mha-d64 | m4-c0 | 9216 / 9 | 1 | 73728 | 0 | 0 | 5536 → 0 | source_budget_exceeded |
| decode-mha-d64 | m4-c4 | 9216 / 9 | 1 | 73728 | 0 | 0 | 4460 → 0 | source_budget_exceeded |
| decode-mha-d64 | m4-c4-simplified | 9216 / 9 | 1 | 73728 | 0 | 1 | 4460 → 2168 | simplified_selected |
| decode-gqa-d80 | m0-c0 | 12864 / 8 | 8 | 102912 | 0 | 1 | 2787 → 2787 | legacy_selected |
| decode-gqa-d80 | m0-c4 | 12864 / 8 | 8 | 102912 | 0 | 1 | 3019 → 3019 | legacy_selected |
| decode-gqa-d80 | m4-c0 | 13376 / 11 | 3 | 107008 | 0 | 1 | 2369 → 2369 | legacy_selected |
| decode-gqa-d80 | m4-c4 | 13376 / 11 | 3 | 107008 | 0 | 1 | 2608 → 2608 | legacy_selected |
| decode-gqa-d80 | m4-c4-simplified | 13376 / 11 | 3 | 107008 | 0 | 1 | 2608 → 1524 | simplified_selected |
| decode-long-kv | m0-c0 | 18560 / 8 | 8 | 148480 | 0 | 1 | 2787 → 2787 | legacy_selected |
| decode-long-kv | m0-c4 | 18560 / 8 | 8 | 148480 | 0 | 1 | 3019 → 3019 | legacy_selected |
| decode-long-kv | m4-c0 | 19200 / 11 | 3 | 153600 | 0 | 1 | 2369 → 2369 | legacy_selected |
| decode-long-kv | m4-c4 | 19200 / 11 | 3 | 153600 | 0 | 1 | 2608 → 2608 | legacy_selected |
| decode-long-kv | m4-c4-simplified | 19200 / 11 | 3 | 153600 | 0 | 1 | 2608 → 1524 | simplified_selected |
| prefill-q4 | m0-c0 | 6688 / 10 | 10 | 0 | 0 | 0 | 21084 → 0 | source_budget_exceeded |
| prefill-q4 | m0-c4 | 6688 / 10 | 10 | 0 | 0 | 0 | 21389 → 0 | source_budget_exceeded |
| prefill-q4 | m4-c0 | 7712 / 13 | 5 | 0 | 0 | 0 | 6183 → 0 | source_budget_exceeded |
| prefill-q4 | m4-c4 | 7712 / 13 | 5 | 0 | 0 | 0 | 6495 → 0 | source_budget_exceeded |
| prefill-q4 | m4-c4-simplified | 7712 / 13 | 5 | 0 | 0 | 1 | 6495 → 3182 | simplified_selected |
| prefill-q8 | m0-c0 | 12720 / 16 | 14 | 101760 | 0 | 0 | 8002 → 0 | source_budget_exceeded |
| prefill-q8 | m0-c4 | 12720 / 16 | 14 | 101760 | 0 | 0 | 8355 → 0 | source_budget_exceeded |
| prefill-q8 | m4-c0 | 14768 / 18 | 8 | 118144 | 0 | 1 | 3786 → 3786 | legacy_selected |
| prefill-q8 | m4-c4 | 14768 / 18 | 8 | 118144 | 0 | 0 | 4146 → 0 | source_budget_exceeded |
| prefill-q8 | m4-c4-simplified | 14768 / 18 | 8 | 118144 | 0 | 1 | 4146 → 2425 | simplified_selected |
| batch-gqa-q4 | m0-c0 | 9120 / 10 | 10 | 72960 | 2 | 0 | 0 → 0 | ineligible |
| batch-gqa-q4 | m0-c4 | 9120 / 10 | 10 | 72960 | 2 | 0 | 0 → 0 | ineligible |
| batch-gqa-q4 | m4-c0 | 10400 / 13 | 5 | 83200 | 2 | 0 | 0 → 0 | ineligible |
| batch-gqa-q4 | m4-c4 | 10400 / 13 | 5 | 83200 | 2 | 0 | 0 → 0 | ineligible |
| batch-gqa-q4 | m4-c4-simplified | 10400 / 13 | 5 | 83200 | 2 | 0 | 0 → 0 | ineligible |

## 数据身份

- 主审计 JSON SHA-256：`e6f00bf3360b8cf85c2d79f9bff691cf1c9019a0ba9c4e43dbb09030c87cdbd4`。
- 主 plan SHA-256：`ad744b97220b1220e88e5b83c4af28af0f5202f65a6074b77ba5031bd3e05b01`。
- 补充审计 JSON SHA-256：`1fcbe6855cc6b90ffa0fda5a57f03b023e2a21cf220d759789b43dd6c2e1d10d`。
- 补充 plan SHA-256：`5f70ed9261b10350c8fc5dd6ba73e8f74693c42826bc47b534037eda6fba3acf`。
- 原始样本、六个配对比值、完整输出与对象身份均保留在证据中；本表仅为四舍五入展示。
