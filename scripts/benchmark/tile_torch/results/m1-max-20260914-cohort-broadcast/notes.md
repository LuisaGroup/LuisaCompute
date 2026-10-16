# Full9 cohort-private / uniform broadcast：CPU消融归档

本归档保存 full9（`cohort-recurrences` freeze）同一二进制的两个独立消融对照。结果见[完整22行对照表](results.md)和 `summary.json`，包含11个case、local1/local8及两个开关的全部44条边。

## 如何读结果

默认同时开启 cohort-private 与 uniform read-lane；另外两臂分别只关闭一个开关。每条边独立执行3轮ABBA、12 visits、每visit5 samples，warmup40ms/target20ms。倍率为**关闭/默认**的六个相邻配对比值的中位数，大于1表示默认更快。不能把不同边的默认时间混用，不能把两条收益相乘，也不能估计缺少“双关闭”臂的交互项。

固定W8、block32、一个CPU worker；全部66 captures、66 prepares、44 replays完成，共528 visits/2640原始样本。计时统一使用原始 `native_tile.py/native_tile_replay.cpp`，直接调用真实ORC原生入口，含入口内部block遍历和compiler-emitted libc/allocations，排除Runtime dispatch、Python、JIT、调用者分配、校验。capture内E2E时间仅用于诊断，不作为平衡对照。没有Torch/MPS对照，不能据此宣称已击败它们。

## 正确性边界

当前11个case的完整输出均通过FP64 reference、有限值、guard与输入不变检查；同一个case/local的三臂输出bitwise一致。离线脚本重新读取完整输入、FP64 oracle、全部输出、原始样本与所有归档字节。原进程的guard bytes未单独落盘，其成功断言保留为runner收据，不声称离线重执行过guards。

**后续审查发现跨epoch collective仍有潜在缺口。** 本轮只说明这些固定case在full9上通过，不能证明一般执行模型完备、所有时序/active-mask组合合法，或后续修正版性能必然相同。历史gate7失败和driver旧版本原样保留，放在`history/`或原始文件名中，不混入full9的成功门禁。

## 冻结与归档覆盖

- 在允许full10改写selected/build前，先从full9的SELECTED树保存23文件源码overlay、四个critical binaries、helpers/ABI和admission/gate身份为独立raw副本，逐项匹配full9 freeze。没有用正在变化的ROOT源码代替历史版本。
- 保留所有66 capture/prepare和44 replay的实际LLVM/ORC/library、输入/输出/oracle、command/stdout/stderr、RSS/timeout收据、driver/plan/binding/admission及旧版本。
- 保留full9完整build、unit SIMD、Tile SIMD、host-plan四道门禁和三项syntax/clang-tidy收据。host-plan成功不等于所有历史host测试全绿。
- 23源码文件是明确overlay，不是完整Git快照。四个critical binaries和逐入口产物也不是完整历史系统/SDK/LLVM/Runtime依赖闭包。原始绝对路径保留为证据，通过manifest aliases读取，不依赖当前机器对应路径。

原逻辑文件约657MiB，按内容SHA-256去重；每个逻辑路径都在`manifest.json.files`里有hash、大小和原始路径，`objects`映射至两个小于45MiB的分片。去重只合并相同字节，不丢弃重复的逻辑产物或visit输出。排除pycache、机器缓存及后来/正在进行的实验；不重复打包[full6性能归档](../m1-max-20260914-program-team/notes.md)或[full7/8 submission诊断归档](../m1-max-20260914-submission-diagnostics/notes.md)的完整内容。

## 复核

```sh
python3 evidence.py
python3 verify.py
```

`evidence.py`只需标准库，独立从raw visits重算44条边的六对ratio和中位数，并核对原44个job。`verify.py`另需NumPy，核对全部归档hash、full9身份、phase/metadata、完整FP64输入输出并与`summary.json`和`results.md`比对；不会加载native code，不读取当前ROOT/SELECTED。首次归档用了`verify.py --write`生成这三份派生文件；日后复核不加`--write`。`SHA256SUMS`给出所有顶层文件hash。
