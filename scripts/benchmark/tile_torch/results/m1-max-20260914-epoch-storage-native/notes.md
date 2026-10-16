# Full11 epoch-storage：新旧入口与 Torch 的新测量

这是独立的新归档，不修改 [full9 消融归档](../m1-max-20260914-cohort-broadcast/notes.md)。full11 的正确性修复见该历史目录中单独的 `correction_full11` 证据；本目录保存修复后的新性能实验。全文结果在 [results.md](results.md)，精确数值和六个相邻配对比率在 `summary.json`。

## 两个独立 timer cohort

- 11 个案例 × local1/local8：22 条 full11/default 对保留 full9/default 的新 ABBA 配对；相同的未修改 `native_tile` C++ timer。264 visits、1320 个 wall 样本。
- RMSNorm/masked-softmax 的 129×65、129×512 × local1/local8：8 条 full11 对已保留 Torch Inductor native entry 的新 ABBA 配对；双方用相同未修改 `native_rows` C++ timer。96 visits、480 个 wall 样本，另16次丢弃 timing 的有检查 preflight。

每条边 3 轮 ABBA，每 visit 5 samples、warmup40ms/target20ms。比率是相邻配对的中位数，不是两个总中位数相除。前一个 timer 不支持 Inductor ABI 1，后一个支持；因此两组分母不能混用，也不能拿旧 Torch timing 替代本轮测量。

full11/full9 比率 0.993765–1.005477：当前案例未观察到明显回退，不能宣称加速或正式统计等价。选择本轮较快 local mapping，四个 Torch 对照仍慢约20.5%、67.9%、38.0%、79.1%；**性能目标尚未完成**。

## 校验边界

全部360个正式 timed visits 通过完整 FP64 final-output、NaN writable 初始化、guards、输入位模式不变检查。Torch 的外露 scratch 也在原 native 进程核对（RMS平方和、softmax max/exp-sum）；共168次完整 writable-array 检查。compiler 内部临时分配位于timer内，但没有可外部复核的独立guard。

离线 verifier 独立读取保存的输入、FP64 oracle、全部360个 final outputs 和1800个 wall samples，重算全部30条边的中位数/配对比率。**guard bytes、actual scratch bytes 未单独持久化**，其通过状态来自保留的 native 收据，不声称离线再次执行了这些检查。保存的 scratch FP64 references 可以重新计算并比对。

## 身份与内容

- full11 来自 `H7yzJk/full11-immutable` 的26文件源码 overlay、四个关键 binaries、完整四门禁及admission；不用当前ROOT/SELECTED代替历史版本。
- full9 来自独立 `full9-immutable` 的23文件 overlay 和四个关键 binaries，22条默认capture/prepare绑定及本轮复制的prepared bundles。复制manifest只将name改为`full9-default`并记录父manifesthash；actual ORC/library没有重生成。
- 本轮22个实际capture ORC、LLVM文本、helper/ABI、输入输出、FP64 oracle、全部command/stdout/stderr、180秒/RSS收据和脚本原样保存。
- 四个 Inductor 实际生成源码/库、wrapper解析ABI、原prepared manifest及共同helper哈希保留。Torch helper源码取之前留存的runner副本，不读当前selected。不是完整历史Torch/runtime/LLVM/系统动态依赖闭包。

原始逻辑文件按SHA256内容去重，逻辑路径/原路径仍全部留在manifest。两个小于45MiB的压缩分片；没有解压到当前项目、没有执行归档的代码。排除pycache、机器cache和不相关历史消融payload。`full9_aliases`与full11 aliases分开，避免相同原始绝对路径在不同版本指向错误源码。

## 离线核验

```sh
python3 verify.py
```

需要NumPy，不需要Luisa/Torch/LLVM/native加载。只读本目录；第一次`--write`生成派生summary/verification/results/checksums，日后不加该参数。`SHA256SUMS`覆盖所有顶层文件。时间边界为单线程native entry wall：含launch resets、block traversal、compiler-emitted libc/allocation；排除Runtime dispatch、Python、JIT、调用者分配和验证，不是GPU时间或端到端吞吐。
