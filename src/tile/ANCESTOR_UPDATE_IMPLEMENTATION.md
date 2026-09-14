# 外层 Tile 局部更新与 parallel 汇合：实施草案

日期：2026-09-14。状态：**设计提案，未实现、未运行本文回归**。

本文只讨论如何补齐现有捕获与 TileIR 的表示能力，不修改执行模型的既定要求：**内层可以读写外层；`parallel` 已经声明实例之间互不干扰，不需要编译器再次证明这个前提才允许优化。** 当前写入限制是实现缺口，不是用户提出的约束。

本文独立于正在验证的 XIR program-team attention 调度改动。逐 phase 调度、逐值布局计划的存在，不代表这里的局部更新／并行汇合已经落地。

**范围澄清：本文后面的 patch 方案只覆盖独立实例的局部更新，不是外层写入的唯一形式。** 原设计还包含 ancestor anchor / child frontier：例如外层 `acc` 在内层执行层级上直接 `acc = mma(a, b, acc)`，可以是一项由子层级协作完成的逻辑更新，而不是每个参与者各写一整块 `acc`。编译器必须先区分逻辑操作实例与物理参与者，不能先按后者复制写入，再以冲突为由拒绝。

因此，下文 `t.update(...)` 和 `TILE_UPDATE` 都只是局部更新的候选表示，不是允许内层访问外层的前提，也不是已经决定新增的用户接口。完整修复还必须支持既定的直接赋值／集体更新；仅实现 patch API 不能宣称这一限制已经消除。

这里不保留旧 API 兼容性：如果新模型需要更合适的 API，就统一替换实现、测试、示例、benchmark 和文档，删除旧入口，不加别名或兼容包装。保留直接赋值的动机是它自然表达逻辑值更新，不是为了兼容现有代码；局部更新 API 同样按语义与使用体验选择。

## 1. 现状：可见性没有必要变成隔离墙

| 行为 | 当前捕获／验证器 | 代码依据 |
|---|---|---|
| 内层读取外层 Tile SSA | 支持；原有 Tile 不因读取而改变 | `value.h::Tile::at`、`dsl.cpp::extract_tile` |
| 内层读取已经初始化的外层手工 Memory | 支持；有明确正向测试 | `test_tile_memory.cpp::test_lexical_ownership` |
| 内层通过外层 TensorView／MemoryRef 显式 `.store()` | 支持捕获，仍受 `parallel` 非干扰契约约束 | `dsl.cpp::store_view`、`store_tile` |
| `serial`／`pipeline`／`reduce` 内更新外层变量 | 捕获为已有状态传递与同时赋值 | `dsl.cpp::enter_scope`、`exit_scope` |
| 子 `parallel` 内对外层命名 Tile 赋值 | 一律拒绝，尚无局部更新与汇合协议 | `dsl.cpp:216` |
| 子 `parallel` 内调用外层手工 `Memory.store()` | 也被拒绝：它更新外层 `_state` ValueSlot | `dsl.cpp::store_memory`、`verifier.cpp::_verify_memory_flow` |
| 子 `parallel` 返回一个组装后的 Tile | 当前验证器禁止 `parallel` 有任何 operands/results | `verifier.cpp:574` |

不能把 TensorView/MemoryRef 的 store 与手工 Memory 的 store 混为一谈。前者记录带地址的外部内存效果；后者目前消费／产生一条覆盖整个 allocation 的 MemoryState 链。

已有拒绝用例不能作为禁止祖先写入的依据：`test_parallel_cannot_capture_scalar_carry` 让四个实例累加同一个标量，`test_lexical_ownership` 的负例让三个实例写同一块完整 Memory。这两个具体程序存在跨实例冲突；它们没有覆盖合法的不相交更新，也不能证明当前一刀切的捕获 guard 合理。应保留冲突用例，同时补上合法祖先更新的正向回归。

当前 `Tile::at()` 只返回纯 Scalar；手工 Memory 只有整块 `load/store`，没有切片写接口。`IndexMap` 支持普通坐标表达式、组合、重排、reshape 和 strides，但没有引用运行时 TileIR `Value *` 的参数节点，也没有现成的动态切片更新指令。

相关实现入口：

- [Tile 与 MemoryRef](../../include/luisa/tile/value.h)、[Memory](../../include/luisa/tile/memory.h)。
- [捕获](dsl.cpp)、[TileIR 对象模型](../../include/luisa/tile/ir.h)、[验证器](verifier.cpp)。
- [布局表达式](../../include/luisa/tile/layout.h)、[分析／重写接口](../../include/luisa/tile/analysis.h)。

## 2. 正确语义：同一入口快照上的独立更新，不是 recurrence

设外层 Tile 的入口 SSA 值为 `T0 : D → Scalar`，子 parallel 的实例域为 `I`。实例 `i` 有自己的词法操作序列，产生若干更新：

```text
u(i,j) = (目标区域 R(i,j), payload V(i,j), 本实例序号 j)

                    同一个不可变 T0
                 ┌────────┼────────┐
             instance 0   1        2
             局部更新链  局部更新链  局部更新链
                 └────────┼────────┘
                 parallel completion
                          │
                          T1
```

`T1[d]` 的定义：如果没有更新覆盖 `d`，保留 `T0[d]`；如果有更新，取所属实例内部最后一次覆盖 `d` 的写入。不同实例的更新遵循已有 `parallel` 非干扰契约；这里不新增一个 `independent/proven_disjoint` 用户参数。

三个必须保留的区别：

1. **本实例顺序存在。** 同一实例可以多次写同一区域，后写覆盖前写；之后的读取能够看到本实例已产生的版本。
2. **入口 Tile 是 SSA 快照。** `auto old = t;` 保留原版本。一个实例读取 `old` 中后来由别的实例更新的位置，仍然读 `T0`，不是对可变共享存储的新旧值竞争。原位优化必须考虑这些旧值仍然活跃。
3. **汇合不是迭代携带状态。** `i+1` 不以上一个实例的结果为输入，不引入跨实例顺序，不把 `parallel` 改成 `serial`。

遇到确定违反非干扰契约的程序，可以给静态诊断；也可在调试模式记录地址／区域并报告冲突。**分析返回 UNKNOWN 不意味着程序不合法，更不能要求用户补一个证明。** 后端是否能实现某种索引映射，是另一件事。

整 Tile 赋值也不能被偷偷解释为“只写当前 lane 的部分”。在确定逻辑实例之后，整值替换的目标区域是整个逻辑 Tile；一次集体赋值可以由很多参与者共同实现，并不因此成为多次有冲突的整值替换。如果确实存在多个独立逻辑实例的局部更新，源程序或结构化映射应保留各自的区域。单实例、集体操作和分区更新不应被一条“外层不可变”规则一并拒绝。

## 3. 最小表示：一个值更新指令，加现有 parallel/yield 的结果协议

### 3.1 `TILE_UPDATE`：本实例的纯值版本

建议新增内部 TileIR 操作，名字暂定：

```text
next = tile.update(base, payload, origin_0, ..., origin_r-1)
```

初版采用同 rank 的矩形 patch；payload 的形状决定更新大小，结果类型与 base 相同。`origin` 是正常 SSA operands，不放在字符串 attribute，也不把运行时值偷偷编码进 `IndexMap`。

- 逻辑目标坐标：`destination[q] = origin[q] + local[q]`。
- 延用现有 bounds 语义：`ASSUME` 是调用前提；`ZERO` 在写侧表示越界部分不写，不是向目标边界外写零。
- 这不是地址引用或外部 memory effect；它构造新的 Tile 值。
- 旧 base 和 payload 的 SSA 快照不被改变。
- 后续可用类型化坐标映射描述 transpose、strided patch、稀疏／数据依赖 scatter；不应把当前矩形构造器写进语言公理。

它可以正规化为现有 `TILE_MAP + TILE_EXTRACT`，但先保留类型化操作有助于保留更新区域、避免生成完整 Tile 的重复复制，并使 passes 可以直接改写 operands。它不是新增执行 nest primitive。

### 3.2 `PARALLEL` 的输出表示 assembly，不沿实例传递

允许现有 `PARALLEL` 携带 Tile 初值与 Tile 结果，复用现有 region/block/Value/Use 基础设施：

```text
T1 = parallel(I, initial = T0) {
    body(i, entry_T0):
        payload0 = ...
        local1 = tile.update(entry_T0, payload0, origin0)
        payload1 = ... read(local1) ...
        local2 = tile.update(local1, payload1, origin1)
        yield.updates result[0] {
            (payload0, origin0, ordinal=0),
            (payload1, origin1, ordinal=1)
        }
}
```

这里的 `yield.updates` 是说明性打印形式：**复用 `YIELD` opcode，增加一个类型化的 parallel-update descriptor**，不是新增运行时 Patch 类型，也不是必须出现于用户 DSL 的语法。

描述符需要：目标 result slot、payload operand index、origin operand range、可选 predicate operand、实例内顺序。每个 payload／origin／predicate 都是 YIELD 的普通 SSA operands，进入 use-list；描述符只记录这些 operands 的结构，不能保存不可追踪的 C++ 裸闭包。

不能只 yield `local2`，再用数值差分或猜测 defining-op 来发现“写过哪里”。写入旧值、NaN、负零，以及多次覆盖都会使这种办法错误或脆弱。显式 patch operands 还保证：即使本实例不再读取 `local1/local2`，DCE 可以删除对应的整值构造，但不能丢掉 join 的 payload。

验证规则按 operation kind 区分：

- `SERIAL/PIPELINE/REDUCE` 保持现有 carry 协议。
- `PARALLEL` 的 Tile body argument 每个实例都绑定同一入口值，不是 PHI recurrence。
- parallel YIELD 的 operand 数量由 update descriptor 定义，不再错误地等同 result 数量。
- 检查 rank、类型、SSA dominance、operand 索引、结果归属、更新顺序和 bounds 选项；不添加“先证明实例互不干扰”的接受条件。
- 同一次完成事件产生多个结果时，保持所有 payload 的旧值依赖，再同时发布结果，不能把 `a/b` 的交换变成先写 a 再计算 b。

第一步只支持 Tile assembly 结果。MemoryState 区域汇合见第 6 节；任意 scalar 的“多个实例最终谁赢”也不能借此获得隐含语义。

### 3.3 分析与 transforms

增加一个普通分析结果 `ParallelUpdateSummary`，记录每个结果的 update sites、实际逻辑区域、入口快照读取、instance-local 顺序及是否完整覆盖。修改 region、origin、payload 或 layout 后，使用现有 `IRRewriter/AnalysisManager` 的失效机制，不复用陈旧结果。

该分析服务于表示／变换／存储规划，不是 parallel 独立性的证明门禁。输出坐标与 patch 坐标是否对应、正规化有没有漏写或重复执行，是编译器自己的正确性义务，不能拿 `parallel` 的契约掩盖编译器映射错误。

## 4. C++ 捕获：不改变既定 operator 约定

以下仅为**提案语法，当前不能编译**：

```cpp
auto t = A.tile(coord(base_row, 0), shape(rows, cols)).load();
auto old = t;

for (auto &row : parent.parallel(shape(row_count))) {
    auto patch = map<float>(shape(one, cols), [&](const Nest &element) {
        return old.at(coord(row.index(), element.index(cols))) + 1.0f;
    });
    t.update(coord(row.index(), 0), patch); // proposed API
}

O(coord(base_row, 0), shape(rows, cols)).store(t);
```

实施后可直接移入现有 Boost.UT 测试函数的 capture 与主机 oracle 主体如下。它使用当前参数／shape／显式 store 风格，唯一新增的用户 API 是标出的 `t.update`；实际执行仍须接入各 bridge 的既有 runtime harness，不能只凭 capture 通过宣称设备通过。

```cpp
using namespace luisa::compute::tile;
auto definition = tile_kernel(
    "ancestor_row_update", [](TensorView<const float, 2> A,
                              TensorView<float, 2> O,
                              TensorView<float, 2> OldO) {
        auto rows = axis("rows", 7), cols = axis("cols", 19);
        auto row_instances = axis("row_instances", 7), one = axis("one", 1);
        for (auto &parent : parallel(shape(2))) {
            auto base_row = parent.index() * 7;
            auto t = A.tile(coord(base_row, 0), shape(rows, cols)).load();
            auto old = t;
            for (auto &row : parent.parallel(shape(row_instances))) {
                auto patch = map<float>(shape(one, cols), [&](const Nest &element) {
                    return t.at(coord(row.index(), element.index(cols))) + 1.0f;
                });
                t.update(coord(row.index(), 0), patch); // proposed, not implemented
            }
            O(coord(base_row, 0), shape(rows, cols)).store(t);
            OldO(coord(base_row, 0), shape(rows, cols)).store(old);
        }
    });
auto kernel = definition.capture(tensor_shape(14, 19),
                                 tensor_shape(14, 19), tensor_shape(14, 19));
expect(kernel.valid());
luisa::vector<float> input(14 * 19), expected(14 * 19), expected_old(14 * 19);
for (size_t i = 0; i < input.size(); i++) {
    input[i] = static_cast<float>(i) * 0.125f;
    expected[i] = input[i] + 1.0f;
    expected_old[i] = input[i];
}
// Runtime harness: upload input, launch on each bridge, compare both entire
// outputs to expected/expected_old, and check guard canaries around buffers.
```

`t.update(...)` 可以只是 `t = insert(t, origin, patch)` 的命名变量糖；最终命名仍需统一审阅。**不把 `Tile::operator[]` 改成 writable reference**，不改变已有 `operator[] → loaded Tile`、`operator() → MemoryRef` 的约定；内存写仍必须显式 `.store()`。

这里的糖必须以 `t` 的**当前版本**为 base。独立的 `insert(old, ...)` 只是纯值表达式；例如先更新 `t`，再执行 `t = insert(old, ...)`，这次整值赋值会撤销先前更新中没有出现在 RHS 的部分，不能被误记成“追加一个 patch”。捕获器必须保留完整替换的语义，或明确尚不支持这种 assembly；不能根据 RHS 恰好是 `TILE_UPDATE` 就自动按局部更新处理。

捕获改动集中在 `ValueSlot/SlotSnapshot`：

1. 进入 child parallel 时保留每个命名变量的入口身份；必要时复用已有临时 forwarding definition 技巧，不能按相同 incoming Value 合并不同 C++ 变量。
2. 更新方法创建 `TILE_UPDATE`，更新当前 slot，另按 slot 记录 patch operands 与词法顺序。记录不能等到退出时通过整值相等性倒推。
3. 本实例后续 `t` 读取用当前版本；此前 `auto old = t` 的独立 slot 保持旧 SSA。
4. 退出 scope，建立 Tile 初值／结果／body argument，先形成带普通 operands 的 typed update YIELD，再 RAUW／移除临时 forwarders，将外层 slot 指向组装后的结果。这样直接引用 forwarding value 的 payload／origin 也会被正确重写。
5. 不变的捕获值仍是普通 lexical capture，不额外生成 outputs。

首个捕获实现可覆盖一个 child body 内静态数量的更新 sites，包括同实例重复覆盖。更新如果位于更深的动态 serial/pipeline 中，不能把内部 SSA 值直接塞进外层 YIELD 而违反 dominance；要保留嵌套更新 region，或构造该局部 region 的合法更新摘要／结果。尚未实现这种 region composition 时必须明确报告能力不足，不能悄悄把 child parallel 改成 serial。

## 5. 两条 bridge：先共用正规化，再补通用通信实现

```text
Tile DSL capture
       │  typed update sites + initial snapshots
       ▼
TileIR PARALLEL assembly / TILE_UPDATE
       │
       ├─ 可生成直接坐标对应：normalize_parallel_updates
       │                         │
       │                 TILE_MAP / EXTRACT
       │                   ┌─────┴─────┐
       │                 XIR         TIRx
       │
       └─ 其余：保留结构 → 逐 phase 分布／通信计划（后续）
```

### 5.1 最小可运行候选

第一候选选择 axis-aligned、可构造直接逆对应的 patch，且 region 无需跨 program 同步。对输出坐标 `d` 生成其所属 child `i` 与 patch 局部坐标 `q`，求 payload；不覆盖的位置读 `T0[d]`。例如三行一块：`i = d.row / 3`、`q.row = d.row % 3`。同实例多个更新按原有顺序选最后命中者。

这是**一个 lowering 候选的能力范围**，不是用户程序合法性的前提。不要求用户提供 injectivity proof，不因一般分析 UNKNOWN 就宣布源程序非法。构造逆对应本身、覆盖补集和 predicate 的保持仍要由变换验证；这些是验证变换有没有做对，不是在重新证明 parallel 的独立契约。

纯点式 payload 可被正规化到现有 `TILE_MAP`，不引入串行实例循环。对于包含复杂 collective/MMA、数据依赖 scatter 或无法廉价反解的区域，保留原结构，不把每个输出元素变成重新运行整个 collective 的巨大 map body。

### 5.2 XIR 接入

当前入口主要在 [lower.cpp](bridge/xir/lower.cpp) 的 `_loop/_region/_operation`、[resources.h](bridge/xir/resources.h) 与正在验证的 `bridge/xir/program_plan.h`。

- 先让共享正规化的简单输出进入现有 map lowering；明确保留所选执行分布，而不是把整个 child region 串行化后声称完成并行支持。
- 当前 program-team 对 direct extract 的接受范围有限。规范化生成的 `origin + local`／逆映射应带可分析的类型化坐标关系，由 access analysis 接入；不能把它随意归成“uniform broadcast”。
- 后续直接支持 assembly：安排目标 Tile 的 placement，按真正 owner 写入对应局部片段；不同布局之间生成 shuffle／shared staging／barrier，而不是要求用户改 source。
- 同一 program team 内的汇合可以是 SSA/寄存器连接，也可以需要局部同步；不同 threadgroup/program 的通信可能需要更宽资源和多个 launch。不能无条件插一个 subgroup barrier 冒充全局完成。
- resource 计数必须按真实片段和活跃快照，不为每个 child 分配一个完整目标 Tile。保留 `T0` 是否需要双缓冲由 liveness 与读取关系决定，不能因 `parallel` 就直接原位覆盖。

已有 `_loop` 对 root escaping results 的拒绝、program-team 对 nested parallel 的拒绝，都需要明确扩展；这份草案不宣称现有代码已经接受新 IR。

### 5.3 TIRx 接入

当前 [lower.cpp](bridge/tirx/lower.cpp) 的 `_lower_structured` 用同一 carry 缓冲协议处理结构化结果，`_lower_block` 的 YIELD 也按 carry 数量解释。新 parallel outputs **不能**直接走这个入口，否则会被错误实现成迭代 recurrence。

- 正规化的简单纯值更新复用 `_lower_tile_map`；测试生成的独立元素映射与最终 target schedule，不仅验证数值。
- 直接 assembly 路径建立一次结果 storage；必要时从 `T0` 填补未更新部分，再让各实例执行自己的 patch stores，保留 parallel／独立实例的 schedule 信息。
- 多个结果的 payload 与旧快照先稳定，再发布写入，延续当前 TIRx YIELD 已处理的 simultaneous assignment 原则。
- 如果要把 private 结果变为跨线程共享 storage，必须由 target planning 正确选择作用域并生成通信；不是把 `BufferStore` 放进 thread loop 就自动获得了共享结果。
- CPU、Metal 等可以共用逻辑更新 lowering，实际 worker binding、vector layout、局部 storage 和同步仍是 target policy 的职责。

两条 bridge 在接受新 IR 前都应 fail closed，并给出明确 capability diagnostic。不能让一条路径静默跳过更新，另一条把它当串行 carry。

## 6. 手工 Memory：区域效果与状态汇合是第二步

Memory 不是 Tile SSA；它有稳定 allocation identity，必须继续显式 `.store()`。不能为了绕过 guard，把 MemoryState 当普通 scalar carry，也不能将现有整块 store 自动理解成当前 worker 的切片。

建议方向：

- 扩展 addressable Memory 的 reference 构造，支持显式 origin/shape；仍返回 `.load/.store` 的地址对象。当前 MemoryRef 绑定的是 View，不能直接伪造 TypeKind::VIEW 来冒充 Memory，应共享内部 access descriptor 或增加正确的 backing variant。
- 对现有 MEMORY_LOAD/STORE 增加类型化区域参数；逻辑地址先由 origin/局部坐标确定，再与 allocation-local `IndexMap` 组合。动态 origin 保持普通 SSA operands。
- 给 parallel MemoryState 输出定义区域效果汇合；状态 token 记录一组区域写入，不把一个全 allocation 的线性 token 强行叉成互相矛盾的链。
- 入口 Memory 的可变读取与入口 Tile 快照不同。跨实例读写必须遵守 parallel 原有非干扰契约；若要读取固定旧内容，应在更新之前取得 Tile 快照。不能承诺普通 shared Memory 自动具有版本化存储。
- `_verify_memory_flow` 改为理解 region effects、到达状态和 join；同一实例仍检查 stale-state 使用与本地顺序。验证器不能再用“祖先 Memory 有变化”作为普遍拒绝条件。

这部分不作为 Tile assembly 第一里程碑的虚假完成项；但它必须进入计划，因为用户要求的是内层访问／使用外层资源，不仅是只读。

## 7. 回归与完成标准

以下是实现后的回归清单，**当前均未因本文而新增或运行**。尺寸要覆盖 tail，不能只用整 warp／整块形状。

| 回归 | 建议尺寸 | 必须验证 |
|---|---|---|
| 按行更新完整外层 Tile | `1×7, 4×19, 33×65` | 完整输出；patch 数量；parallel 仍是独立映射 |
| 每实例更新三行，最后一块截断 | `7×19, 65×33` | 尾部不越界；未覆盖区保留 T0；guard canaries |
| 稀疏更新偶数行 | `7×19, 33×65` | 奇数行逐 bit 保留，包括负零／NaN payload |
| 同实例重叠两次写 | `4×19` | 第二次写覆盖重叠区；第一次非重叠区保留 |
| 同实例 write→read→write | `4×19` | 后续读取看到本实例的新版本 |
| scope 前 `auto old = t` | `4×19` | old 仍逐 bit 等于原输入；禁止不安全原位优化 |
| 两个外层 Tile 同时交换分区 | `4×19, 33×7` | 不被降为先写 a 再从新 a 算 b |
| 多个 sibling child scopes | 第一个行分区，第二个列分区 | 第二个读取第一个 join 结果；布局转换与完成边清晰 |
| child parallel 位于 pipeline 内 | trip counts `0,1,3` | 初值／零次迭代；每步 join；没有跨实例 recurrence |
| 祖先 Tile 任意旧值只读访问 | 行更新读取另一行 old | 快照语义，不误套可变内存 R/W 冲突规则 |
| 未支持但合法的映射 | 数据依赖、非直接可逆分区 | 明确 capability diagnostic；不是“请证明 parallel 独立” |
| 确定重叠的不同实例写 | 两个实例同一区域 | debug/validation 可报告冲突；不定义 last-instance-wins |
| 手工 Memory 区域更新（后续） | 同上及不同 mem resource | reaching state、局部顺序、区域 join 与真实同步 |

验证分层：

1. Capture/IR：检查 descriptors、use-lists、lexical dominance、旧 slot 身份、join 结果与零次域，不只检查 `Kernel.valid()`。
2. Shared normalization：独立主机参考实现按上面的集合／顺序语义计算；枚举小尺寸核对坐标对应、覆盖补集和 tails。
3. XIR/TIRx 编译结构：检查不是 recurrence、没有每实例整 Tile 复制、没有静默序列化或漏掉通信。
4. 两条实际 bridge 执行：动态输入、完整 tensor、guards、只复制／插入时逐 bit；算术按该算子的既有精度契约。
5. 性能单独报告：同时保留 kernel timing 与 dispatch/端到端口径，不把 capture/编译通过当成性能结果。

可复用现有 `test_tile_dsl`、`test_tile_values_cpp*`、`test_tile_memory`、`test_tile_xir_program_team`、`test_tile_xir_runtime` 与 `test_tile_tirx_values/execution/memory` 的注册机制。实际执行前先完成所选 build tree 的全量 build；修改源后重复该门禁。本文没有运行任何测试命令。

第一里程碑的完成标准是：**合法的外层 Tile 分区更新能够 capture、经过可修改的 typed TileIR、在两条 bridge 执行并保持 parallel 语义，完整结果与快照回归通过。** 仅移除 guard、只写语法 PoC、只得到一份 IR dump，或者把实现改成 serial，都不算完成。
