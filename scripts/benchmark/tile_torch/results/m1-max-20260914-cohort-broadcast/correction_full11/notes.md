# full11：跨循环 epoch 的 cohort 值修正证据

2026-09-14；对应本地提交 `e20d24d524908076e978b1752b05112262f925b1`。本目录独立封存，不修改 full9/full10 历史归档。

- full10 原生反例实际失败：W2/4/8/16、快路开关均出现退出 population `2 != 1`、后续循环 `121 != 111`；不是超时。归档保留原始日志及当时 23 文件 overlay。
- full11 使用 selected 的精确 26 文件 overlay，freeze SHA-256 为 `29ef908f51beadddedaac6cc85735f2c260cf86978c075bd0fb875707230099c`，不是读取当前 ROOT 来替代测量源码。
- full11 完整 build、13 个 unit_simd CTest、Tile SIMD runtime、两个 host-plan CTest 全部通过；含跨 epoch 标量及嵌套 uint4 快照反例。
- 六个 syntax/tidy TU 全部零错误。warning 数依次为 uniformity 3、XIR schedule 6、collectives 0、uniformity test 0、schedule test 0、codegen test 20；已核对为原有代码警告，未将 passed 等同于无警告。
- 修正是循环逃逸分类、依赖传播与 collective 结果形状，不改变 participant mask 或退出快照语义。此处没有性能样本，不能据此宣称加速、性能目标完成或 Metal 恢复。

`evidence.tar.gz` 包含源码 overlay、完整 command/result/stdout/stderr、freeze/owned/toolchain/source-clone、实际诊断脚本与编译数据库。它不是完整依赖树或二进制闭包；重建仍需 source-clone 中的基底和对应依赖。Metal trace 未纳入此 freeze。

在任何位置执行 `python3 evidence.py` 即可仅用标准库离线校验归档、成员哈希、红／绿回归及来源一致性，不运行原生代码或解压文件。`SHA256SUMS` 另覆盖本目录交付文件。首次生成使用 `python3 evidence.py --archive`，拒绝覆盖既有归档；归档前后会再次核对全部来源文件。
