# Provenance 补充说明（不修改已冻结记录）

原 `sources.tar.gz`、`provenance.json` 和其中的 source hashes 保持不变。实际归档的完整源码文件是本次 source snapshot 的权威内容；历史 producer/tool fingerprints 不构成完整 loader 闭包或独立二进制执行证明。

## 测试时间顺序更正

`provenance.json` 的 `preflight` 沿用了较早收到的测试状态；本补充说明此前所写“source snapshot → captures”也不准确。主线程确认，完整构建 retry1、四项完整 CTests 和七项短 host tests 均在 capture 与 source snapshot 之前完成并通过。测试日志单独留档，这里不补造其完成时间。

下列为原始 JSON 中的 Unix 时间戳原值：

| 原始字段 | Unix seconds |
| --- | ---: |
| `plan.json:frozen_unix` | 1789297276.716018 |
| `captures.json:started` | 1789297276.7162921 |
| `provenance.json:frozen_unix` | 1789297282.999222 |
| `captures.json:finished` | 1789297334.4378948 |
| `replays.json:started` | 1789297358.339317 |
| `replays.json:finished` | 1789297381.5977259 |

因此，source snapshot 完成时 captures 已在进行；competitive replay 在 snapshot 与 captures 都完成之后才开始。十个 owned 文件的 main/isolate hash 一致性已核验，归档过程也检查了源码未变。原冻结 `sources.tar.gz`／`provenance.json` 不回写，以上更正只存在于补充文件。

## 附带 diff 的表示限制

归档脚本对 `git diff` 的输出调用了 `.strip()`，移除了末尾 LF。这个附带 patch 只是对十个 owned 文件相对主 HEAD 的差异说明，不替代完整源码文件。Competitive replay 完成后，`git apply --numstat` 只读解析原附带 patch 返回 128，诊断为 `corrupt patch at line 639`。

因此另存 `owned-changes-exact.patch`：保留当前 `git diff --binary HEAD -- <十个 owned paths>` 的精确 stdout，并验证其 SHA256 与直接捕获 stdout 的 hash 相同。它比原附带 patch 多且仅多一个结尾 LF；`git apply --numstat` 只读解析返回 0，显示全部十个预期文件。没有真正 apply 补丁，没有修改源码或任何冻结快照。

精确补丁 SHA256：`ee7b25b7a47d96412835be14afed38cf67180b4f6569ff98afca0fb9c5e3e14d`。`provenance-correction-receipt.json` 保存两次只读解析结果、差异边界及冻结快照不变的 hash。原归档脚本也保持不改；后续重用时须移除对 patch 输出的 `.strip()`，但不得据此回写本轮历史归档。
