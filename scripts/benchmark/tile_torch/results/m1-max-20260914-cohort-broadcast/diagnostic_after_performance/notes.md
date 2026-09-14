# Full10：性能采样之后确认的 collective 反例

这是独立诊断增量，不是 full9 性能准入的一部分，也没有改写 full9 的分片、manifest、summary 或原始结论形成时间。它更新了主归档当时“潜在缺口”的表述：**跨 epoch collective 错误已由实际原生运行确认，不再只是风险推测。**

full10 完整构建通过；两个针对性测试 executable 都失败，CTest exit 8、非超时。宽度2/4/8/16，`enabled=0` 和 `enabled=1` 两种模式均观察到：active=2 时，lane1退出所保存的 population 实际为2、预期1；lane0随后循环结果实际121、预期111。详见保留的完整 stdout，重复输出按相同字段去重后有16条数值失败记录。

归档中的23文件 source overlay逐项匹配 full10 freeze及保存的 `owned-through-full10.json`。与不可变full9 freeze相比，只有两个反例测试文件变化，没有 production source 改动；因此不能用full9的11个数值样例全部通过来推导通用正确性，也不能把这次诊断失败归因于full10新增生产优化。修复候选及其门禁、性能不在此包中。

只保留已终态的 full10 freeze、原source tar、build与test完整收据/日志，加上owned快照和full9 freeze参考。没有读取当前ROOT/SELECTED源码、没有补跑原生测试、没有归档完整系统依赖闭包。主归档的11case计时仍是历史实测数据，但不代表模型正确性已经闭合，也不是发布就绪声明。

离线复核仅需标准库：`python3 evidence.py`。`manifest.json`包含逐文件hash、source核验及16条实际失败值；`SHA256SUMS`覆盖此独立增量。主归档原有校验文件保持不变。
