# Full12 byte-private：正确性与静态检查归档

本目录只保存源码与元数据，没有性能样本，也没有二进制闭包。对应的核心实现已提交为 `c23dc2f8b`；实际门禁绑定的是 **27 文件 selected overlay**，freeze SHA256 为 `71573837811263abe91d86a4c0b33bb425479e2bd5c2638e92d899a1afb56720`，不能用整个 ROOT commit 代替该实验源码身份。

full12 在保留的 full11 基础上仅覆盖 `llvm_schedule_emitter_memory.cpp` 和 `test_llvm_schedule_codegen.cpp`。25 个继承文件逐字节不变；新增 memory TU 是第27个显式冻结文件。两个文件的 baseline/candidate 副本、全部27个 frozen-source 文件、plan/owned/self-check/runner 及真实 command/result/stdout/stderr 都已保存。full11 父 freeze、四门禁、runner/toolchain/source-clone pins 和工具版本收据另存 `parent/`。

| 门禁 | 已完成结果 |
|---|---|
| 完整 build | passed，runner wall 17.80s；没有 `--target` 缩窄 |
| unit_simd | 13/13 CTest executables，CTest real 52.91s |
| Tile XIR runtime SIMD | 1 executable；31 tests / 5,207,697 asserts，52.64s |
| focused host plan | 2 executables；34 tests / 364,142 asserts，0.68s |
| memory TU syntax/tidy | 0 errors，0 warnings |
| codegen-test TU syntax/tidy | 0 errors，21 warnings；未称零警告 |

新增原生回归记录了 bool/int8/uint8 byte storage、active=0…W、非前缀 masks、私有槽完整 byte oracle 和循环退出快照，覆盖 W2/4/8/16 及 cohort-private 开关。bool 的计算类型仍为 i1，但内存保存是 i8；连续写保留未参与 lane 的原始 byte，不能用 `<W x i1>` store 覆盖 byte ABI。这是内部 typed private-array 实现，不意味着 FP16/FP8 自动获得该路径，也不改变 Tile DSL 的祖先读写约定。

test TU 的21条警告完整保留。其中新增 `cells` 的 `cppcoreguidelines-pro-type-member-init` 警告对应 `std::array<uint8_t,17> cells;`；下一语句即 `cells.fill(0xa5u)`，在读取前填充全数组，属于可能的初始化检查误报。没有为消除警告重写已冻结测试、隐藏 diagnostics 或放宽 oracle。

性能仍待新的 full12/full11 配对，不能把门禁通过或静态 i8 gather 消失直接写成已测收益。通用 planner 特征、RMW/byte 转换、mask 和 fallback 的计价边界见 [private-access cost 提案](../../../../../src/tile/PRIVATE_ACCESS_COST_MODEL.md)；该成本模型尚未实现／校准。

## 离线核验

运行 `python3 evidence.py`：只用标准库读取本目录，检查 archive/member 哈希、27文件身份、两文件差分继承、父 pins、全部终态及日志、精确 CTest 选择和零编译错误。不会 build、加载 binary、执行归档脚本、读取当前 ROOT/SELECTED 或再次运行测试。

首次 `--archive` 从原始不可变副本/收据收集，拒绝覆盖已有归档，前后复核来源；`SHA256SUMS` 覆盖全部顶层交付文件。原始路径保留在 manifest 以供检索，不是离线验证依赖。静态日志绑定被检查的源码，但本档没有完整 clangd 配置／编译数据库闭包；工具元数据是版本/path 收据，不是工具二进制哈希。source-clone 保存基底与更早 overlay 的来源信息，不宣称保存了全仓库及递归依赖源码。
