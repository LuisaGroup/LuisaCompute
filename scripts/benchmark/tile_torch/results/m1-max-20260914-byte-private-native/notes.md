# Full12 byte-private：原始计时结果检查点

日期：2026-09-14。为及时提交而保存的最小结果包，不是完整可执行闭包。

`results.tar.gz` 保存85个原始文件：四阶段汇总、admission/self-check/plan/gates、两个驱动、30组正式测量结果，以及44份新旧 prepared 元数据。30组结果包含360次访问和全部1800个计时样本，未删除异常值；输出路径、输出SHA256与原执行的正确性/guards收据保留在相应结果中。prepare汇总保留byte IR观察记录。

Softmax的full12/full11配对耗时比为0.752–0.854，耗时降低14.6%–24.8%；RMS基本不变。独立Torch对照仍未达成性能目标。完整表格、时间边界、配对方式及限制见 [报告21.17](../../../../../src/tile/ATTENTION_MAPPING_REVIEW.md#2117-full12-byte-private纯-kernel-配对结果)。

**本包不含完整输出数组、LLVM/ORC对象、计时helper二进制、生产库或完整依赖源码。** 因此只能检查封存的计时记录和原执行收据，不能声称凭此离线复算完整数值正确性或重新执行kernel。驱动依赖其他历史路径，不能把保存驱动解释为自包含运行环境。源码与门禁另见 [full12 gates](../m1-max-20260914-byte-private-gates/notes.md)；full11父实验另见 [epoch-storage-native](../m1-max-20260914-epoch-storage-native/notes.md)。

未删除的完整本地原始目录：
`/tmp/luisa-metal-program-team.TGZv8f/cpu-full12-vs-full11.gFpqhZ`
（macOS解析后为`/private/tmp/...`）。它仍保存完整输出、对象与依赖材料；后续可扩充归档。临时目录不是永久保留保证。

归档SHA256：
`cec0d4fedf0411f1c88b7d946f0d25f00b0aee1c657540eacdb80b147effbe50`

当前代码又进行了独立的API清理（`413cefa3e`），不属于本包被测源码。该清理仅完成调用点审查与diff检查，尚未重新构建／运行测试；继续工作应先全量构建，再跑types/XIR/target-info/program-team回归。祖先Tile更新与新private-access成本特征仍未实现，不因本轮提交而宣称完成。
