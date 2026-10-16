# Program-team：2026-09-14 原始证据归档

本目录只定义证据覆盖与复算边界；性能解释由主报告单独给出。未运行的实验不作为性能结果。

## 覆盖范围

- `raw-evidence.tar.gz` 保留 full1–6 build、full4/5/6 相关 host/SIMD/Metal 诊断与原始失败，不把后续通过覆盖为“历史全绿”；同时保留 freeze、toolchain、clone/configure、syntax/tidy、runner 和 inventory。
- CPU default full6：六个 case，各 local1/local8，共十二份真实 capture，含原始 stdout/metadata、LLVM、实际 ORC object、输入、FP64 oracle、输出、prepare/link/export/import 收据与独立 native replay。
- 同一 full6 binary 的 cohort-private off/on：六个 case；只改变已有选项，保留六份新 capture/prepared 与六组 ABBA。其 baseline 是 default local8，不是 local1。
- Inductor：四个 row case，实际 Torch 2.14 编译生成的 C++、wrapper、graph、native library、common timer，以及 local1/local8/Inductor 三臂 raw samples。沿用原始 `native_rows` 准入和计时常量；只有 adapter 指定 LLVM22 工具目录。v1–v4 的 ImportError、错误门禁假设、脚本转义、原地 summary 造成的 harness failures 原样保留；v5 才是通过版本，不代表 compiler/native numerical failures。
- `source-default-predication.tar.gz` 仅含 full6 freeze 指定的 17 个源码文件，从 **selected 冻结树**读取并逐项匹配 receipt hash，绝不使用正在修改的工作区版本。原 full4 `source-owner-slot.tar.gz` 和 full5 `source-composed-predicates.tar.gz` 原样包含在 raw archive，内部文件也逐项匹配各自 freeze。
  两个旧 tar 的 AppleDouble `._*` 元数据也原样保留并另列 hash，不把它们算作源码；每项均检查 AppleDouble magic 和对应的真实源码路径。
- `critical-binaries-full6.tar.gz` 只有 capture manifest 标识的四个 critical artifacts：benchmark executable、SIMD backend、Tile library、core library。**这不是历史完整 dylib/runtime 依赖闭包**；不包含 LLVM、Torch、系统库或 SDK 本体。
- `supporting-sources.tar.gz` 保留 selected 树中的两套 native timer/driver、FP64 reference、fixture 构造、ABI header，以及归档时读取的 CMakeCache/compile_commands。source clone receipt 保留基线来源，但这些 overlay 包 **不构成完整 Git/source 快照**。
- 独立 Metal copy-only-control v1/v2 的源码、offline prepare 和完整 v2 raw 目录也原样归档。v2 已终态失败（30 秒 cap、退出 -15，纯 copy 第五次提交等待；含两次采样）。这是独立 Runtime 诊断 Error，不作为 GPU 恢复证明或 Tile 回归结果。

本归档明确排除 Python bytecode、Inductor 编译缓存和大型机器缓存。被实际测量的 Inductor source/library 已单独保留；归档排除缓存不等于排除其被测产物。Metal copy-only-control 仅在主代理确认进程终态后纳入。

## 计时与验证边界

两套 native timer 分开解释，禁止跨 cohort 相除：

| 对照 | case 数 | 每 case visit | 每 visit samples | warmup / target |
|---|---:|---:|---:|---:|
| default local1/local8 | 6 | 12，ABBA×3 | 5 | 40 / 20 ms |
| cohort-private off/on | 6 | 12，ABBA×3 | 5 | 40 / 20 ms |
| local1/local8/Inductor | 4 | 18，三臂全部六种排列 | 7 | 100 / 30 ms |

共 216 visits / 1,224 raw samples。时间单位为 µs，测量单线程原生入口，包含入口内部 block 遍历、compiler-emitted libc/allocations；排除 Runtime dispatch、Python、JIT、调用者分配和验证。capture 内的 E2E 时间只是单次诊断，不构成平衡 Runtime 性能对照。

原生 runner 会做完整输出/FP64、guard 和输入不变检查。离线审计重新读取保留的完整 tensor、oracle 和哈希，重算原始样本与配对统计；但原始进程里的 guard bytes 没有另存，所以 guard 成功仍以 runner 收据为边界，不声称离线重执行过 guards。没有 MPS 对比，也没有由这些 CPU 数据推导 Metal 性能。

## 校验与复算

`manifest.json` 逐包、逐 member 给出 SHA-256、大小、原始路径，保留 build 状态和源码覆盖。`recompute.py` 只用 Python 标准库，直接读取 archive，不解压执行 native code：

```sh
python3 recompute.py
```

其 JSON 输出应等于 `summary.json`，所有配对 ratio 都用同一轮/相邻 ABBA pair 的样本中位数计算，不能用两组总中位数相除代替。`archive.py` 是本次机械归档程序，不负责重建机器环境。raw 内 `cpu-inductor.zGCsEf/audit.py` 和 `audit-results.json` 记录原始环境中的完整离线数值复核；前者含原始绝对路径，不是可迁移构建脚本。
