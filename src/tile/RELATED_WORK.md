# Tile execution model：相关工作、设计取舍与研究机会

## 摘要

**我们的方向有价值，但 execution-first、执行与内存解耦、层级组合、自动 pipeline，以及基于约束的映射搜索，都已有直接先例。** 最接近的参照系不只是 TileLang、Triton 和 CuTe，还包括 Cypress、Hidet、Stripe、Prism/Bundl、Hexcute 与 Twill。

值得追求的不是一个“比所有工作更通用”的模型，而是一个更具体的结果：**从带有明确并行／归约契约的可变 Tile SSA 出发，以可组合的方式联合规划参与者、数据分布、存储版本和时间调度，并把方案可靠地实现为不同后端上的高性能代码。** 这是研究假设，不是当前实现已经具备的优势。

核心问题是：**我们能否建立一套更 rigorous（严谨、可验证）的模型？**

- 需要明确程序承诺了什么、变换保持了什么、哪些数值变化被允许，并给出相应证明或可检查的条件。
- 目前还不能声称比已有工作更严谨：Prism/Bundl、ATL 等已有实际形式证明，我们仍主要是语义草稿、受限实现与测试。
- 严谨性与实现自由度不冲突：约束应来自语义和硬件义务，而不是把语法层级、某个规范循环形态或默认 layout 固定为唯一实现。
- 当前最明显的问题不在 DSL 能不能写出算子，而在**从合法程序到高性能实现之间的候选覆盖面**。可表达、可验证、可高效 lowering、可被搜索器找到，是四件不同的事。

本报告采用中文，保留必要英文术语。文献核对截止 **2026 年 9 月 7 日**。研究范围是与本项目直接重叠的编程模型、IR、映射代数、异步调度和优化方法；这是一份有重点的相关工作调查，不是穷尽检索，也不是新颖性证明。文献陈述以所链接版本的相关章节为准，不代表对应项目所有后续版本的能力。本轮没有复现论文性能实验，也不把不同硬件上的论文数字拼成性能排名。

**文档归属与本次整理。** 本文件保留文献分析、审查依据和研究假设；正式语义以 [Tile 语言文档](../../docs/source/tile/design.md) 为准，变换义务以 [calculus](../../docs/source/internals/tile/calculus.md) 为准，候选组合以 [planner](../../docs/source/internals/tile/planner.md#compositional-search-contract) 为准，实施检查点见 [决策记录](../../docs/source/internals/tile/decisions.md)。本次把建议并入这些既有页面，不另建平行文档体系。按最新讨论，**普通 `reduce` 默认 `unordered_tree`；保序树与严格 fold 由用户显式选择**。§12 保留实现前的审查快照；逐操作 policy 现已落地，最新实现边界见 §13，不能再将快照中的“尚未实现”当作当前状态。

## 1. 先把比较对象拆清楚

### 1.1 不要把“有 hierarchy”当成一个完整模型

下面五个问题相互约束，但不能互相替代。图中箭头表示约束关系，不表示必须按这个顺序不可逆地做决定。

```text
                    Semantic program / region
                     values, effects, contracts
                              |
           +------------------+-------------------+
           |                  |                   |
     Work assignment    Value distribution    Event ordering
     who does what      who holds which v     when it may run
           |                  |                   |
           +---------- Resource placement --------+
                      versions and lifetimes
                              |
                    Target atoms / protocols
                              |
                        Executable code
```

例如，同一执行上下文中的 A、B、acc 可以使用不同 layout、不同资源和不同生命周期；同一 SSA value 的不同消费者也可以采用不同分布，只要显式实现必要的数据交换。反过来，两种实现访问同样的最终地址，不代表它们保持了相同的依赖、归约计数或 collective 参与者。

### 1.2 四个比较维度

| 维度 | 应当回答的问题 | 常见误判 |
| --- | --- | --- |
| 语义表达能力 | 能表示哪些值、依赖、并行、归约和动态行为？ | “接口更少，所以模型更完整” |
| 实现自由度 | 同一个合法程序允许哪些空间／时间／存储映射？ | “有嵌套语法，所以必须按硬件树逐层映射” |
| 形式保证 | 哪些变换有证明，哪些只有检查器或测试？ | “有数学符号／求解器，所以整个编译器已验证” |
| 有效自动化 | 编译器实际上能找到并生成多少好方案？ | “理论候选空间很大，所以性能一定更好” |

本报告后文的“可改进”均指具体可检验的方向，而不是默认我们在上述四个维度都优于现有工作。

### 1.3 直接重叠的研究地图

| 我们关心的部分 | 优先对照的工作 | 最重要的借鉴 |
| --- | --- | --- |
| 执行与资源独立映射 | Cypress、Stripe、Tiramisu | 独立 placement；保留 effect 和版本依赖 |
| 空间／时间任务组合 | Hidet、Graphene、LEGO | 可组合坐标映射；任务不等于物理线程 |
| 算法／调度分离与自由控制流 | Halide、FreeTensor | 源语言便利性与优化自由度要双向比较 |
| parallel／reduce 的语义 | Stripe、MDH、Lift | 原语直接携带可优化的契约 |
| 顺序／并行的组合与精化 | Concurrent Kleene Algebra | 明确事件偏序和 refinement，不把不等式当等式 |
| collective 的合法参与者 | Prism/Bundl | scope、uniformity 和 convergence 进入接口 |
| layout algebra 与硬件 atom | CuTe、Linear Layouts、Axe/TIRx、Hexcute | 复用已知代数，保留适用域与转换代价 |
| pipeline 与安全复用 | Cypress、Tawa、Twill、Pallas/Mosaic GPU | issue、completion、release 分开 |
| 组合式变换与搜索 | TensorIR、Exo 2、ATL、Mirage、符号 Prism | 区域摘要、可检查变换、符号候选族 |
| 不规则／跨 scope 执行 | Futhark、Event Tensor、Legion、DISTAL | 不要把一种 flattening 或静态调度写死 |

这张表是阅读路线，不是功能评分表。各系统的任务边界不同，不能用“是否有某个同名类／关键字”判断能力有无。

## 2. 执行层级与资源映射：已有非常接近的基础

### 2.1 Cypress：最直接的 execution/resource separation 先例

**已建立的机制。** Cypress 将逻辑 task、task variant、处理器绑定和逐 tensor 的内存放置分离。其论文模型区分 inner task 与 leaf task：前者分解、分区并启动子任务，后者访问 tensor 数据和执行底层计算。IR 使用带处理器维度的 completion-event 数组；消除 copy、展开层级和复用 buffer 时，同时维护必要依赖。评测中的 mapping 是作者手工调优的，不是一个自动联合求解器的输出。[Cypress，PLDI 2025，§3–§5.1](https://arxiv.org/html/2504.07004v1)。

**与我们的相同点。** 执行层级不决定唯一的 memory layout；一层可以访问多份独立映射的数据；通信和同步可以由编译器生成。这些不能再作为我们的独有贡献。Cypress 本身已有 SSA，不能把“我们使用 SSA”当成相对于它的区别。

**最值得借鉴的是依赖的粒度。** 不应只保存“阶段 A 在阶段 B 前面”，还应保存“哪一个资源版本的哪些生产者／消费者需要等待”。下面是我们需要支持的通用情形：

```text
                     +--> consumer A completes(v) --+
producer completes(v)+                             +--> reusable(v)
                     +--> consumer B completes(v) --+        |
                                                           v
                                                overwrite the same slot
```

图中的 join 只包含实际使用该版本的消费者。消掉某次 copy，不代表可以消掉 copy 原先承载的 publication/join；完成 A，也不能让仍被 B 读取的 slot 提前复用。

**我们可以尝试改善的地方。** 在普通可见代码中，从 Tile SSA、load/store 和词法上下文推导访问摘要，减少 task privilege 和 variant wiring 的常规样板；允许区域中直接混合 tile 运算与子 nest，不强制采用论文中的 inner/leaf 编程分工。opaque 调用仍必须有显式 effect 契约。更重要的是，把人工 mapping 的一部分变成可组合候选搜索，并证明不会因自动化而隐藏必要控制。

**尚未成立的优势。** 当前没有足够证据说我们的异步 lowering、资源复用或者映射自动化比 Cypress 更完整。C++ 嵌入和不依赖 MLIR 是工程选择，不自动构成语义优势。

### 2.2 Hidet：空间与时间因子组合不需要从零发明

**已建立的机制。** Hidet 将 task mapping 定义为 worker 到有序 task 列表的映射：

```text
f : Worker -> List<TaskCoordinate>
```

`spatial` 分配任务给不同 worker，`repeat` 分配多个有序任务给同一 worker；它们可组合，组合满足结合律，但一般不满足交换律。worker 可以是线程，也可以是 warp、block 或其他层级的抽象执行者。论文另有 post-scheduling fusion。[Hidet，ASPLOS 2023，§5.1–§5.2](https://arxiv.org/html/2210.09603)。

**借鉴。** 对矩形独立域，直接吸收这种空间／时间因子的组合意义：

```text
spatial(outer) * repeat(local) * spatial(inner)
```

它已经解释了“execution hierarchy 像 layout nest”的重要一半：工作坐标如何分解，如何分配，单个执行者内部如何排序。`repeat` 中的顺序是实现调度，不应误解为逻辑任务原本必须存在串行依赖。

**我们要补的部分。** task mapping 不是完整的依赖／资源语义。跨阶段 load/store、异步完成、资源复用和归约贡献的合法重组，需要附加契约。但这并不证明 Hidet 无法表达这些程序，只说明不能把一个 task-product 公式当作全部正确性条件。

**实现自由度的风险。** 如果我们只允许每一层 `parallel` 一对一绑定一个硬件层级，而不允许拆分、合并、时间复用，反而会比这种任务映射更僵硬。源程序的逻辑实例应与物理 worker 分离。

### 2.3 Stripe：parallel 的无冲突语义有直接先例

Stripe 的 parallel polyhedral block 允许一个实例内部有有序语句，但限制不同实例之间的普通读写依赖，并为指定的结合、交换 aggregation 保留例外。子 block 的访问通过祖先索引派生，refinement 表示 buffer 子区域。[Stripe，2019，Definition 2、§3.1–§3.2](https://arxiv.org/html/1903.06498)。

**借鉴。** 用户写 `parallel` 就是在提供语义契约，编译器应利用它，而不是每次都重新证明用户“确实想并行”。不过，这不等于所有参数全局 `noalias`：同一个实例里先读后写同一元素可以合法；不同实例的冲突是另一回事。

**可改善的契约划分。** 我们可以把 ordinary parallel、合法的 reduction、ordered recurrence 分开：普通写冲突不是隐式 reduction；有结合律但无交换律的合并允许保持顺序的树化，不允许任意条带重排。相对于论文中 aggregation 的定义，这是一个更细的区分；它不是“所有 Stripe 版本都不可能支持非交换归约”的断言。

### 2.4 MDH 与 Lift：归约方向和并行方向可以统一推导

MDH 用多维同态描述计算的分解与重组，每个维度带相应 combine 规则，并将其映射到 core 与 memory 层级；其输入／输出 views 与组合规则分别描述索引和计算。它直接覆盖了“部分维度拼接、部分维度归约”的思路。[MDH，TOPLAS 2024 的完整版本，§3–§5](https://arxiv.org/html/2405.05118)。

Lift 则把 map/reduce、布局重排和不同粒度的并行表示为可组合 pattern；低层 IR 还区分计算映射与结果存储位置，并利用这些语义生成访问、分配和同步。[Lift，CGO 2017，§3–§5](https://lift-project.github.io/publications/2017/steuwer17LiftIR.pdf)。

**借鉴。** 对 `y[m] = reduce_k f(m,k)`，不需要发明一个与其他 nest 完全异类的“tile 迭代器”。可以从贡献域到输出域的 grouping map 出发：

```text
g : M x K -> M
g(m, k) = m

different fibers: independent outputs
within one fiber: contributions + a merge contract
```

**需要保留的边界。** “沿 k 有依赖”不足以说明可以并行聚合；树化还需要兼容的状态／merge 契约，一般 recurrence 没有这个性质。我们的 `reduce` 默认允许无序树，显式严格 fold 则可以不提供并行 merge（见 §12.8）。scan 还暴露所有前缀结果，不能被只保留终值的 reduction 替代。浮点 add 的重结合来自选定的归约语义，不是精确实数代数自动赋予的权限。

我们更偏向 effectful、execution-first 的源语言；纯数据并行代数可以成为其中可强力优化的片段，而不是要求所有程序都先改写为一个全局同态。这是有意选择的分析边界，不是已经证明的整体表达能力优势。

### 2.5 Graphene 与 LEGO：计算布局和数据布局的组合也已有研究

Graphene 把多维数据和线程都表示为可层次化分块的 tensor，用它们之间的映射表示优化后的计算，定位是低层 tensor IR。这里直接核实的是作者的论文摘要；未取得完整正文来核对其所有变换限制，不能据此声称它缺少某项能力。[Graphene，ASPLOS 2023，作者页面](https://mgarland.org/papers/2023/graphene/)。

LEGO 使用可组合的 computation/data layouts 推导索引，讨论用户提供的置换、逆映射及部分 tile 等情形。它的双射核心使组合和反向索引清楚，但任意用户函数的双射性不能仅凭 API 接受了该函数就得到证明。[LEGO，CGO 2026，§III–§IV](https://users.cs.utah.edu/~tavak/assets/pdf/LEGO-CGO26.pdf)。

**借鉴。** 统一坐标空间的类型、组合方向、有效域和 inverse/preimage 契约。**不要混淆。** 索引组合解决“对应哪些元素”，并不独自解决“哪个参与者应执行一次”“有几个副本”“什么时候可读”和“何时能覆盖”。我们的改进应明确落在这些附加义务及其组合上，而不是宣称首次将线程看作 tensor。

## 3. Collective 视角：Prism/Bundl 比单纯的层级映射更接近语义问题

Prism 用 typed perspectives 表示代码控制哪一组线程，以及数据在什么粒度上一致；`group`、`split` 和 memory perspective 管理 collective 的合法调用。其核心 calculus Bundl 有 type-and-perspective safety 定理。它解决的是 modular collective programming，不只是把线性 thread ID 改写成多维索引。[Prism/Bundl，2025 年预印本，§3–§4](https://arxiv.org/html/2511.11939v1)。

**直接启示。** “有 32 个线程”不是一次 warp collective 的完整条件。还可能需要指定的线程分组、对齐、统一控制流、参与 mask、参数一致性和结果分布。`parallel` 的独立性不能代替这些条件：前者约束逻辑实例之间的 effect，后者约束实现某个逻辑操作时的物理协作。

**对我们的设计建议。** 把 participant/convergence 要求放进 atom 和区域接口，在普通 DSL 中尽量推导，而不是要求每个 kernel 都写 `owned_by`、`visible_to`、`accessed_by`。用户显式约束某个硬件层级时，仍由相同规则检查。学习它的语义并不要求现在恢复 intra-kernel Tile/SIMT DSL 混合。

**形式化保证到什么程度？** 目前 Prism/Bundl 在这部分有比我们草案更具体的形式结果。我们想加入自由 remapping 与自动 planning，意味着需要证明变换后仍满足 collective 契约，而不是因此自动拥有更强保证。另一方面，也不应把 perspective safety 误报为对所有内存错误和整个 GPU 编译链的完整证明。

## 4. Layout：复用成熟片段，而不是宣布一种代数无条件完备

### 4.1 CuTe、Linear Layouts、Axe/TIRx 各自擅长什么

| 基础 | 核心表达方式 | 值得直接借鉴 | 不应外推的结论 |
| --- | --- | --- | --- |
| CuTe | 层次 shape/stride，composition、product、division 等 | 混合进制分块及成熟组合语义 | 一个普通 strided layout 就能表示任意置换、依赖或生命周期 |
| Linear Layouts | GF(2) 上的位线性映射 | 常见 GPU 分布、转换、shuffle/swizzle 推导 | 对任意整数形状、任意算子和所有执行变换都完备 |
| Axe / TIRx | 命名物理轴上的 shard、replica、offset | 显式描述分布、副本、偏移及异构资源坐标 | 一份 storage layout 就唯一决定计算调度 |

CuTe 的运算及前提以官方代数文档为准。[CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)。Linear Layouts 的闭包与最小性结果针对论文指定的 Triton shape operators；其本体限制为二次幂形状，可用 padding/masking 处理部分其他域，不能把这个方法和直接表达所有形状混为一谈。[Linear Layouts，v5，§4、§8、Theorem 9.3](https://arxiv.org/html/2505.23819v5)。

Axe 将逻辑元素映射为命名物理轴上的位置集合，明确包含副本；其 DSL 也支持多粒度执行视角。TIRx 的官方设计把 layout 定位为 storage contract，而非 work-partitioning interface，并按执行 scope、operand layout 和 target 分派 tile primitive。[Axe，2026 年 v2 预印本，§2–§3](https://arxiv.org/html/2601.19092v2)；[TIRx，2026 年官方设计](https://tvm.apache.org/2026/06/22/tirx)。

**对我们最重要的结论。** 保持 execution 与 memory 解耦，与兼容 TIRx 并不矛盾。桥接的是规划后的实现关系，不是要求用户采用 TIRx 的源编程风格：

```text
logical element e
      |
      +--> (participant p0, local slot s0)
      +--> (participant p1, local slot s1)    replicated value
      +--> ...
```

这是一种关系／集合值映射，不能直接取“唯一反函数”。还需要携带有效域、资源实例和版本上下文，避免把 inactive slot 或属于不同 owner 的同名 slot 混在一起。

**采用建议。** 使用一个统一的、带类型的语义接口，在内部保留有明确适用域的整数／混合进制与位线性片段。只有落在某个片段内的表达式才调用其规范化和等价算法。并不是要求立即实现几套完整求解器，更不是把任意动态下标塞进“affine”标签。TIRx bridge 对其支持片段做关系保持检查，其余路径明确降级或诊断。

### 4.2 Hexcute：layout inference 必须从指令约束反向传播

Hexcute 从 GEMM／copy 等 anchor 建立 thread-value 与 shared-memory layout 约束，传播并搜索指令选择，必要时插入 layout conversion。成本模型区分 issue 与 completion。v3 的限制讨论明确提到：互联 GEMM 的一致 thread arrangement 仍可能需要用户标注，共享布局搜索的分解与剪枝也尚有空间。[Hexcute，2026 年 v3，§IV–§VI、§IX](https://arxiv.org/html/2504.16214v3)。

**借鉴。** 让硬件 atom 的要求驱动布局候选，而不是先选任意好看的 layout 再希望 tensorization 命中；将数据交换和 fallback 的真实工作计入成本。

**我们可以改进的假设。** 对 `mma -> reduce/exp -> mma` 这样的区域，不只从每个 MMA 局部推导最佳 layout，而是保留多个可连接的区域实现：

```text
MMA plan A ---- cheap conversion ---- reduction plan B ---- MMA plan C
MMA plan D ---- no conversion ------- reduction plan E ---- MMA plan F
```

第二行不一定更快：它可能需要更多寄存器或更少并行度。必须比较整段资源占用和执行时间。改进点应是“边界信息支持有效的联合选择”，不是“全局暴力枚举比局部推导更高级”。

### 4.3 TileLang、Gluon、Pallas：自动化边界必须可以被专家调节

TileLang 提供 tile 操作、布局推导、tensorization 与 pipeline，论文示例显式分配 shared/fragment 存储。[TileLang，2025 年 v2，§3–§4、附录 B](https://arxiv.org/html/2504.17577v2)。现代 Triton 的 Gluon 已直接暴露 layout、shared memory、warp specialization 和 target-specific 功能，不能沿用旧论文对早期 Triton 的描述来评价它。[Gluon 官方概述](https://triton-lang.org/main/gluon/index.html)。

**我们的区别是默认选择，不是它们“做不到”。** 普通 `.load()` 得到逻辑 Tile value，临时 materialization 默认留给编译器；手工 Memory 和硬件 binding 是优化约束。这可能减少样板，但也可能产生隐藏 copy 或性能突变。必须让自动方案可查看、可约束，并让手调和自动搜索使用同一套实现机制。

Pallas/Mosaic GPU 已暴露 pipeline 并发深度、warp-specialized 执行和延迟释放等控制；官方示例说明异步 MMA 未完成时不能提前复用输入 buffer。[Mosaic GPU Pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)。我们可以自动推导更多常规生命周期，但不能因此删掉专家处理困难情形所需的控制能力。

## 5. 时间、资源和调度：不能只写一个 II 和一个 latency

### 5.1 Tawa：ready 与 reusable 是两个状态

Tawa 使用 asynchronous reference（aref）表达跨 producer/consumer 的通信，操作区分发布、获取和消费完成，并用环形存储支持多迭代在途；论文给出相应操作语义及多粒度 pipelining 的 lowering。[Tawa，CGO 2026，§III-B–§III-E](https://www.csl.cornell.edu/~zhiruz/pdfs/tawa-cgo2026.pdf)。

**借鉴。** `issue(op)`、`complete(op)`、`release(resource-version)` 不应合并成同一时刻。最后一条文本上的 use 之后，硬件仍可能在异步读取该存储。

**适合我们的边界。** aref 可以是一种内部 protocol 实现，而不必成为每个用户必须操作的 DSL 实体。我们要先有可推导的版本／消费者关系，再根据后端选择 ring、barrier、shuffle 或串行实现。多消费者不能直接套用单消费者的释放状态机，需要实际完成条件的 join。

### 5.2 Twill：最直接的联合 solver 参照，而不是一句“可以用整数规划”

Twill 联合考虑 modulo schedule、warp assignment、存储存活和通信／阻塞同步，寻找模型内的最优 SWP+WS 方案。OSDI 2026 最终论文的实现仍限制为无额外控制流的单层循环，tile size 由外部选择，求解通常需要数十秒到数分钟；评测还手工将所求方案翻译为 CUDA，以处理剩余 lowering 问题。[Twill，OSDI 2026，§3–§6.1](https://www.usenix.org/system/files/osdi26-soi.pdf)。

**借鉴。** 求出的 pipeline 必须能由实际 issue warp 实现。若该 warp 阻塞在一次 wait 上，它就不能同时发出另一个操作；把 wait 仅当成 DAG 中一条“零代价边”会高估 overlap。存储容量、在途版本数、通信方式和 warp 分工必须共同约束。

**我们可以改善的三个具体目标。**

1. 从单一循环扩大到可组合的子区域／兄弟区域，而不是把全部标量事件一次性塞进一个巨大约束系统。
2. 在关键边界共同选择 atom、layout、参与者与 materialization，避免求解后靠人工修复实现。
3. 提供有预算的 JIT 路径；小规模精确求解器可作为离线 oracle，在线使用缓存、候选族、beam search 或测量反馈。

这些目标尚未完成。Twill 的窄片段换来了具体最优性保证，不能把“我们拟支持更多情况”当成比它更严谨。也不能把模型最优解释为真实硬件的全局最快。

### 5.3 Timeloop、CoSA、Halide autoscheduling：成本模型与搜索应分层

Timeloop 将硬件模型、mapping 与性能／能耗评价组合；CoSA 使用约束优化求空间加速器的调度；Halide autoscheduler 展示了学习式成本估计与树搜索的配合。这些都说明“有一个 cost model 加 solver”本身不是新贡献。[Timeloop，ISPASS 2019](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)；[CoSA，ISCA 2021](https://arxiv.org/abs/2105.01898)；[Halide autoscheduling，SIGGRAPH 2019](https://halide-lang.org/papers/autoscheduler2019.html)。

建议保持四层分工：

```text
hardware capabilities --> legal candidate construction
                               |
backend calibration ------> cost evaluation
                               |
                         search algorithm
                               |
                       measured executable
                               |
                    ranking/calibration feedback
```

硬件容量和数值契约是硬条件，policy 不应通过降低分数来绕过；延迟、带宽、并发程度的估计是模型参数，不应伪装成已测事实。后端重写 cost policy 很合适，但算法可复用还要求 candidate 的边界接口足够通用。

对有限候选集合，合理的目标是最小化给定测量场景下的预测时间，同时受布局、依赖、资源和 emitter 支持约束。不能无条件把各 op latency 相加，也不能无条件取 compute 与 memory 两者的最大值：真实依赖和 issue 限制可能禁止所假设的 overlap。

JIT 多次捕获／编译不同参数仍是合法、直接的 autotuning 实现；它与符号候选、共享分析和缓存并不矛盾。不应为了复用分析而强制“只能 capture 一次”。

## 6. 变换与搜索：薄 IR 不等于薄语义

### 6.1 Schedule trees 与 Tiramisu：多个 sibling scope 不是新的表达能力

Polyhedral schedule trees 已用 band、sequence、set、filter 等节点表达局部调度、先后关系和无序的子域，而不是只能表示一个完美循环嵌套。Tiramisu 进一步分离算法、执行调度、数据存储和通信四层表示。[Schedule trees，IMPACT 2014，§3–§4](https://acohen.gitlabpages.inria.fr/impact/impact2014/papers/impact2014-verdoolaege.pdf)；[Tiramisu，CGO 2019 的 v3 论文，§3](https://arxiv.org/html/1804.10694v3)。

**借鉴。** 函数体是带顺序和条件的 region tree，不必是一根只能向下延伸的 nest 链。源程序中的多个 scope 可以成为 fusion、fission、interchange 和 pipeline 的输入。

**关键限制。** `parallel A; parallel B` 只承诺 A 内部、B 内部各自无冲突，不承诺 A 与 B 之间没有依赖。例如：

```text
A(i): temporary[i] = f(input[i])
B(i): output[i] = g(temporary[i])

candidate: one worker runs A(i), then B(i)
```

若其余 effect 和 alias 条件允许，以上点对点关系适合融合。但若 B 读取 `temporary[i - 1]`，把 A/B 直接塞进独立 worker 的顺序体就不够了：需要 halo、重计算、可实现的同步，或保留分段执行。这个新增的跨 region 义务，不能被 A 自己的 `parallel` 契约免除。

**我们的潜在增量。** 把 collective 参与者、异步版本和 target atom 的要求纳入这种 region 组合；不是重新发明 sequence/set，也不是认为 affine 工具只能处理一维串行顺序。

### 6.2 TensorIR：保留 block 接口，减少从标量语句反推意图

TensorIR 的 block signature 记录迭代域、读写区域和归约初始化等信息，使 block 内部实现和外部调度可以分开操作，并利用 block 结构做合法性检查和 tensorization。[TensorIR，ASPLOS 2023，§3–§4](https://arxiv.org/html/2207.04296)。

**借鉴。** 我们的薄 Tile IR 应保留足以做变换的语义摘要，而不只是让 bridge 消费的一串 serialization opcode。MMA、reduction、copy 的语义在映射决策之前有价值，不能过早只剩它们碰巧展开出的循环形状。

**可增加的内容。** 在 block 式接口上，补充跨 phase 的 value distribution、参与者要求和资源版本的完成／释放条件。摘要由分析生成和随变换更新，不要求用户写第二套 ownership DSL。只有这些摘要确实支持更多合法变换或减少分析成本，才构成可验证的改进。

### 6.3 Exo 2 与 Fireiron：小而可信的变换动作，组合出大的优化策略

Exo 2 将细粒度 scheduling action、程序检查和 cursor 支持组合成可由用户库构建的调度机制。Fireiron 将计算规格逐步分解、细化，最终落到机器指令或 microkernel 实现。[Exo 2，ASPLOS 2025，§3–§5](https://arxiv.org/pdf/2411.07211)；[Fireiron，2020 年预印本，§3–§4](https://arxiv.org/pdf/2003.06324)。

**借鉴。** 搜索器的动作应该与手工调度、后端特化共用，而不是三份互不相通的实现：

```text
manual policy ------+
beam / ILP search --+--> checked refinement actions --> realizable plan
backend heuristic --+
```

**不要过度类比。** Exo 的 imperative scheduling 接口与我们 execution-first 源语言不相同；不必把用户的每个 kernel 都变成一长串 schedule script。真正可复用的是“小动作 + 检查 + 可组合策略”的编译器结构。

公共 DSL 的 primitive set 小，不等于内部只能有几个变换。library 可以组合复杂算子；内部也可以组合 factoring、repartition、materialize 和 atom selection。增加一个内部分析或 refinement，不应自动增加一个用户必须理解的新实体。

### 6.4 ATL：有形式化草稿，不等于已有形式证明

ATL 为纯 tensor 语言的关键重写提供了 Coq 机械化证明；所讨论的实现片段与带任意可变状态、异步资源和显式并行控制的系统不同。[ATL，POPL 2022，§3–§6](https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf)。

**借鉴。** 先选一个能真正验证的片段：定长 domain、明确数值语义、可见 read/write、有限 atom 和 protocol。证明少量基础规则，再证明它们的组合，而不是先宣称覆盖所有 kernel。

**我们的难点。** 纯表达式等价不能自动证明 buffer 复用安全，也不能证明 collective 不会 deadlock。若要扩展到 effectful execution refinement，应明确新增的观测语义和组合条件。目前的 calculus 文档是设计草稿，在机械化保证上并不比 ATL 更强。

### 6.5 Mirage 与符号 Prism：跨层搜索和参数族的验证值得复用

Mirage 用 kernel、thread-block、thread 层级的 μGraph 搜索代数和执行结构，并对候选进行功能验证；其随机化验证依赖指定计算片段，不能等同于所有浮点程序的逐 bit 等价证明。[Mirage，OSDI 2025，§3–§5](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf)。

2026 年的 *Prism: Symbolic Superoptimization of Tensor Programs* 将此类候选提升为带符号参数的 sGraph，分离结构／映射搜索和参数实例化调优；验证使用 e-graph 公理。论文明确不宣称公理集完备，也未给出整个 pipeline 的形式 soundness 证明，而是使用人工审查和随机测试。[符号 Prism，2026 年预印本，§2–§5](https://arxiv.org/html/2604.15272)。

**名称注意：它与第 3 节的 typed-perspective Prism/Bundl 是不同工作。** 一个重点是程序生成与符号等价，另一个重点是 collective 编程和类型安全。

**借鉴。** 对我们的一组 JIT 参数，可以复用区域结构、映射约束和已验证变换，而不为每个 block size 从头枚举同一证明问题。但动态值改变控制流、访问或数值语义时，不能无条件复用证明。缓存 key 必须覆盖相应前提。

**可能的差异。** 我们更希望在带 memory effect 和异步执行的 Tile IR 上做渐进 refinement，而不把全部优化都归结为纯 tensor graph 的代数搜索。这扩大了实现需求，也扩大了证明负担；目前不能据此宣称搜索能力更强。

### 6.6 Halide 与 FreeTensor：execution-first 的取舍要双向比较

Halide 分离函数式算法与 schedule；调度空间同时包含计算粒度、存储粒度、并行性与重计算，compiler 还做 sliding window 和 storage folding。它并不是“只调循环、不考虑 memory”的早期版本。[Halide，PLDI 2013，§3–§4](https://people.csail.mit.edu/jrk/halide-pldi13.pdf)。

**我们的差别在起点。** Halide 从算法定义加 schedule 构造执行；我们从带逻辑执行结构和 effect 的程序出发，细化参与者、资源与时间实现。后者便于直写有状态的多阶段计算，但也可能给编译器留下更多需要保持的观测。前者的纯语义则可能使重计算、融合、边界推导更容易。因此“更像 Halide、但 execution-first”是一种取舍，不是天然更大的优化空间。

**借鉴。** temporary storage 的自动规划和复用应成为常规能力；同时允许 compute 粒度与 storage 粒度不同。不能因为一个 memory 在某层创建，就强制所有消费它的 phase 使用同一执行分布；也不能把所有 load 都立即变成不可改变的物理 allocation。

FreeTensor 将细粒度控制流、partial evaluation、依赖感知变换和自动微分用于不规则 tensor 程序，直接提醒我们“自由形式的 imperative tensor DSL”也已有先例。这里核对了作者项目说明；作者页面所链论文 PDF 当前返回 404，因此不据此判断其具体变换集合或证明能力。[FreeTensor，PLDI 2022，作者项目页面](https://pacman.cs.tsinghua.edu.cn/~zjd/projects/freetensor/)。

**对我们有用的区分。** 让普通代码易写、允许动态访问，是表达能力；从显式 `parallel`／reduction 契约和硬件 atom 接口获得更稳定的优化前提，是另一层设计。二者可以结合，不能仅用是否采用 Python/C++，或者是否使用函数式语法，来判断谁更通用。

### 6.7 Concurrent Kleene Algebra：执行偏序的组合本身已有成熟理论

Concurrent Kleene Algebra（CKA）研究顺序、并行、选择与迭代的组合。其有界并行片段已有相对于指定 pomset language 语义的完备性结果；这里的 pomset 是带标签的事件偏序，不是 layout。[CKA: Free Model and Completeness，ESOP 2018，§1、§3–§4](https://arxiv.org/pdf/1710.02787)。

一个直接关联 scope fusion 的关系是：

```text
(A || B); (C || D)  <=  (A; C) || (B; D)
```

在该语义下，左边先做完 A/B 再执行 C/D，比右边只要求 A→C、B→D 增加了顺序；左边可作为右边规格的实现。**这不是可以任意双向改写的等式。** 把左边程序改为右边的执行，需要另行确认新跨越的 effect 允许去掉那些顺序。也不能只沿用符号，却省略它所依赖的语义解释。

**借鉴。** 为“强度关系”采用明确的行为／依赖偏序，区分规格和实现。它能支撑“独立工作可以时间复用”的推导，而不必发明一套仅靠 primitive enum 排序的规则。

**还需要补什么。** 纯 sequence/parallel 组合树不直接精确表示任意依赖 DAG。例如 A→C、B→C、B→D 而不要求 A→D 的四事件结构，不能只靠这两种组合精确写出；加更强的同步可以执行，却损失并行机会。我们的版本／事件关系应能携带这类边界依赖，不应为保持漂亮的 nest 树而全部变成整层 barrier。

该 CKA 完备性结果也不能直接移植为我们的完备性：资源实例、浮点归约、collective 协议和 target atom 都是新增语义。值得研究的是所需片段能否有一个清楚的嵌入或扩展，以及扩展后仍成立哪些组合规则，不是仅把现有理论换个名字。

## 7. 不必要的实现限制：真实反例与当前范围边界

### 7.1 Futhark：同一 nest 的不同 phase，可能需要不同并行粒度

Futhark 作者在 2026 年 full flattening 的工程记录中给出一个反例：为了利用非一致条件分支中很小的内部并行度，flattening 的管理成本反而造成退化；而同一外层 `map` 的兄弟代码确实需要并行，按该层统一选择的 incremental flattening 无法两全。手工 sequential 属性能帮助可见代码，但不容易修复编译器自动生成的代码。[Futhark 作者博客，2026-07-31](https://futhark-lang.org/blog/2026-07-31-full-flattening.html)。

**这不是“Futhark 不能处理嵌套并行”的证据。** 恰恰是更完整地利用并行，暴露了决策粒度的问题。

对我们的直接警告是：

```text
one logical region
  phase A: tiny pointwise work      -> few participants / local execution
  phase B: wide reduction           -> collective participants
  phase C: matrix operation         -> matrix-capable team
```

不能仅因为三个 phase 位于同一个词法 nest，就强制它们使用同一种物理分布；也不能只因为可并行，就默认展开到最多 worker。允许 phase 之间改变 mapping，并把转换成本计算在内，才是更实际的通用性目标。

### 7.2 Event Tensor：静态 tile algebra 不是全部动态执行语义

Event Tensor 用紧凑的事件表示描述细粒度任务依赖，并讨论静态／动态调度和数据依赖的任务路由。这一方向直接关联不规则负载和 persistent 执行，而不仅是给静态 thread/block 轴改名字。[Event Tensor，2026 年论文 v1，§2–§3](https://arxiv.org/html/2604.13327v1)。

**借鉴。** 静态编译器内部的 completion relation，应留下将来引入动态事件实例的空间。具体事件实现可以不同，不要在最初的语义中直接写死为一个 block barrier。

**范围边界。** shape specialization 不能消除运行时 routing、数据依赖的稀疏访问和不规则任务就绪关系。现阶段支持规则单 kernel 是合理的收敛目标；“以后可扩展”不等于当前已经能高效处理动态 MoE 或多机调度。

### 7.3 Legion 与 DISTAL：跨设备不是给 hierarchy 多加几个枚举值

Legion 将逻辑 region 的访问权限、依赖分析与 task／physical instance 的 mapper 分离；mapper 可以定制性能选择，而运行时负责保持数据和任务依赖。DISTAL 分离 tensor 的数据分布和计算调度，以生成分布式执行。[Legion，SC 2012，§II–§IV](https://elliottslaughter.com/pdfs/sc2012.pdf)；[DISTAL，PLDI 2022，作者论文页面](https://compilers.stanford.edu/publications/pldi22-distal/)。

**借鉴。** “后端 policy 可重写，但不改变正确性”是成熟的设计原则。memory 之间不是一个从小到大的全序，哪些执行者能访问哪些资源、是否需要迁移和一致性操作，才是有效模型。

**我们不应过早承诺。** 多 GPU／多机还涉及通信拓扑、collective 协议、运行时就绪和数据驻留等问题。相同的区域／事件接口可能帮助扩展，但单 GPU calculus 的成立不能替代这些额外义务。先把 Metal 与 SIMD 的独立 Tile kernel 做扎实，符合当前项目范围。

## 8. 综合建议：语义与推导更严谨，实现空间保持开放

本节是基于文献和仓库设计的**建议**，不是已经实施的新 DSL 规范。正式接口仍应统一维护在现有的 Tile 文档中，避免这份调查变成另一套互相矛盾的设计。

### 8.1 哪些限制必须严格，哪些限制不该来自模型本身

| 必须保留的约束 | 不该被误写成必要条件的限制 |
| --- | --- |
| 源程序中 observable 的值、effect、顺序和数值契约 | 每个纯语法 scope 都必须对应一个物理层级 |
| 同步／atom 的合法参与者和 uniformity | 同一 region 的每个 phase 必须使用同一参与者分布 |
| 每个逻辑贡献的计数与允许的合并次序 | 一个 value 只能有一种物理分布或一份 materialization |
| 每个资源版本发布、消费和复用的安全性 | 所有临时值都必须由用户手工分配 memory |
| 硬件容量、访问能力及 emitter 支持 | 所有后端必须具有同一棵硬件层级树 |
| 明确的 hard binding／precision 要求 | 默认 layout、pipeline 深度等启发式不能被搜索更改 |

所谓 observed scope，不只是“代码调用了 index”。索引派生、资源实例化、carry、collective 参与等都可能依赖上下文。变换要保持这些观测；但可以通过显式坐标恢复、资源重定位等方式保持它们，不必原样保留源代码的括号树。

### 8.2 一个区域的接口应包含什么

建议由编译器维护以下内部摘要，尽可能从普通 DSL 代码推导：

```text
Region contract
  domain and context observations
  input/output values + required distributions at the boundary
  read/write/reduction effects + resource versions
  ordering / carry / completion / release obligations
  collective and numerical contracts

Region implementation candidates
  participants, local work and target atoms
  layouts, transfers and materializations
  schedule, in-flight versions and resource usage
  exported boundary events + cost summary
```

语义层的输入输出不需要一开始就指定物理 distribution；上图中的具体边界 distribution 属于候选实现接口。两层必须区分，否则“推迟 mapping”会在创建 region 时被悄悄破坏。

区域组合也不能只返回一个 `estimated_us`。父区域需要知道子区域使用什么资源、何时释放、输入何时消费、输出何时可见，以及边界分布是否兼容。只按局部时间选一个 winner，可能丢掉总体最好的方案。

可保留一个有预算的 candidate frontier：即在**相同边界前提和可比较的资源／成本模型**下，去掉被支配方案。若资源 overlap 或调用上下文不同，不能仅凭两个标量分数就声称 dominance。这是对组合式 solver 真正需要形式化的部分。

### 8.3 Execution hierarchy 可以用 layout 描述坐标，但不等于完整 layout

一个实现见证可以沿用当前 calculus 草稿的分解：

```text
W = (tau, beta, {delta_v}, {mu_s, addr_s}, Theta)

tau      : active new logical occurrences -> original occurrences
beta     : logical prefixes -> target participants + virtual/temporal context
delta_v  : participant/local-slot occurrences -> logical elements of value v
mu_s     : logical owner/version -> resource instance
addr_s   : resource-local logical coordinate -> address
Theta    : issue, completion, communication and reuse protocol
```

这些项不是要求用户书写的五个对象，而是解释“这个 plan 为什么实现了这个程序”所需的信息。实际表示可以更紧凑，部分信息可以推导。

祖先坐标与 local access 的组合能够导出逻辑访问；`beta`、`delta_v` 和资源映射则决定由谁执行、数据在哪里以及如何访问。这里 `delta_v` 从存储 occurrence 指向逻辑元素，反方向使用其关系逆／preimage 表示副本，不假设存在唯一函数逆。但同一个坐标映射可以对应安全或不安全的 `Theta`，因此不能用 layout 等式替代全部执行正确性。

建议先支持可判定、可 lowering 的多个片段：JIT 固定参数后的整数域／混合进制映射，以及固定宽度的 bit-linear layout；在边界保留 typed map/relation 和显式转换。跨片段无法规范化时，保留未化简组合或退出该优化，而不是声称所有表达式都在某个高效 normal form 中闭合。任意用户函数需要弱化契约或额外证明，不能因为可调用就自动获得 inverse。

### 8.4 “强度关系”应分解，不能把 parallel、reduce、serial 排成一条链

固定同一组事件及其语义时，增加必要顺序会减少可用调度；但 `parallel` 同时带有独立性假设，reduction 同时带有贡献与合并法则，pipeline 可能带有不同的部分顺序。因此强度至少是“允许假设 × 必要顺序 × 数值权限”的乘积偏序，而不是一个 enum 的大小比较。

这回答了“parallel 最弱”中容易混淆的两点：

- **顺序上弱**：独立实例之间无需强制先后，可有更多实现。
- **承诺上强**：用户承诺普通实例不会冲突，编译器可以利用这个事实。

`reduce` 不应只是特殊的 serial。它描述向同一输出分组的贡献及其合并契约；可重结合、可交换、是否允许重计算、是否要求确定性应分别处理。source contract 的 validation 可以发现错误，但不能变成每次 lowering 都从零重证所有独立性的性能门槛。

### 8.5 先验证一组小的 refinement，再谈“完备”

可作为起点的内部规则包括：

| 规则族 | 必须检查或继承的条件 |
| --- | --- |
| domain reindex／split／merge | 有效域覆盖、贡献计数、坐标观测保持 |
| 空间／时间重新分配 | required order、参与者和 effect 契约保持 |
| repartition／replicate／materialize | 数据版本、转换和访问可实现；不重复非法 effect |
| reduction factoring | 分组及 identity／merge 契约、顺序和数值权限 |
| atom refinement | 操作语义、shape、precision、alignment 和 collective 要求 |
| fusion／fission／pipeline | 跨区域依赖、资源存活、同步可实现性及无死锁义务 |

后续至少应分开声明五种结论：

1. **表示闭包**：所声明片段的组合仍可表示，不保证仍然容易求解。
2. **Soundness**：每条被接受的变换，在前提下保持可观测行为。
3. **有界相对覆盖**：对明确的 domain、map、atom、protocol 候选词汇，能表示／生成哪些方案。
4. **模型内最优**：在给定候选集合和成本函数内的最优，不是所有等价算法中的最优。
5. **实测效果**：真实设备、形状和测量口径下的性能与泛化。

检查结果为 unknown 时，可以保留保守实现或拒绝该候选；不能将 unknown 当成证明成功。源程序违反契约、用户要求的 hard binding 无法满足，也不能靠静默换一个语义不同的方案来“fallback”。反过来，某个优化 matcher 不支持合法程序，也不应被报成 DSL 语义错误。

### 8.6 如何比较严谨性，而不是比较数学符号的数量

以下保证针对不同对象，**不能直接排成“谁全面更强”的名次**。对应来源和适用片段已在前文给出。

| 工作／方向 | 保证主要落在哪里 | 我们还需要补什么 |
| --- | --- | --- |
| Linear Layouts | 指定布局片段的代数性质与构造 | 执行 effect、贡献计数、生命周期不由布局闭包推出 |
| CKA 的有界并行片段 | 指定事件偏序语言中的公理完备性 | 新增资源、数值和硬件语义后重新建立适用条件 |
| Prism/Bundl | 核心 calculus 的类型与 perspective 安全 | 将参与者要求接入 Tile region／atom 的 refinement 检查 |
| ATL | 指定纯 tensor 重写的机械化证明 | 为可变资源和异步执行建立额外的观测与进度语义 |
| Twill | 指定调度模型与候选问题内的优化 | 覆盖 emitter 的实现义务；明确模型误差和在线预算 |
| 我们当前的 calculus | 定义、规则草稿、有限参考测试 | 尚缺所声明完整片段的证明及 production lowering 的验证链 |

**因此，合理目标不是“证明自己比每篇论文更严谨”，而是把目前断开的保证接起来。** 例如，合法 layout 不能接到不合法的 collective；局部正确的两个 stage 不能在共享 buffer 复用时破坏彼此；solver 的 plan 不能在 codegen 中变成另一套未经模型覆盖的行为。

### 8.7 一个值得落实的组合性命题

可以围绕以下命题建设模型和检查器，**这仍是待证明命题，不是本报告已经完成的定理**：

> 若每个区域实现都满足自己的语义接口，边界 value／版本／参与者契约兼容，新增通信保持 reaching value，联合资源与 issue/wait 协议可实现且不会引入死锁，则组合实现保持源区域组合的可观测行为。

它的重要性在于让优化器可以更换一个区域的实现，而不必把整个 kernel 重新展开为一个巨大的标量证明问题。难点也很具体：接口摘要必须足够表达跨区域的 hidden coupling，尤其是共享资源容量、异步占用和阻塞的 issue worker。仅分别证明每个区域“独立运行时没问题”是不够的。

可观测行为需要先定义：输出值、允许的内存 effect 和数值语义；不能把每个原始标量事件都强制一对一保留，否则 MMA atom 或合法归约树无法成为 refinement。浮点误差还需要可组合的契约，单次 `allclose` 通过不是具有传递性的等价关系。

优化层再单独定义：

```text
C(P, H, V) = admitted realizable plans
             for program P, hardware H, bounded vocabulary V

W* = argmin { predicted_time(W, H, scenario) | W in C(P, H, V) }
```

`C` 的合法性来自语义、资源／协议和 target 能力，不能被 cost policy 改写；`predicted_time` 则可以由后端校准。精确搜索最多证明这个有限问题的最优；有预算的启发式应报告 incumbent、搜索范围及可用的下界，不能伪称全局最优。真实时间是否接近预测，另由实验检验。

## 9. 对照当前仓库：已有基础与实际缺口

本节保留 **2026-09-07 原报告阶段的工作树静态检查**：分支 `codex/tile-programming-design`，HEAD 为 `8feb8fed4464b7baef367a349c0a83ad2d4824a5`。工作树包含已有的未提交优化，所以下述观察不是仅对该 commit 的声明，也不是新一轮性能复测结果。原报告阶段只新增此文件；本次继续整理正式文档，仍不修改那些编译器代码。源码位置以函数名和文件为准，不将随后移动的行号当成冻结版本证据。

### 9.1 已有可用基础，并不是从零开始

| 检查项 | 已有证据 | 不能由此推出的结论 |
| --- | --- | --- |
| 可变 IR 主干 | `Operation` 使用 managed intrusive node；SSA use list 支持替换 | 所有变换都已正确维护 analysis／semantic invariant |
| typed MMA 与数值权限 | 独立 `OperationKind::MMA`、`MmaPolicy::allow_reassociation` | 任意标量加乘均可替换为 matrix atom，或可随意降低输入精度 |
| 后端成本 policy | `ExecutionCostPolicy` 提供系数和 reduction cost hook | 已实现通用的 region/layout/pipeline 联合 solver |
| execution calculus 草稿 | 已区分 mapping witness、observed cuts、数值与资源义务 | 已完成任意域的 soundness 或 backend correctness 证明 |

源码依据：[Tile IR 与 MMA policy](../../include/luisa/tile/ir.h)、[ExecutionCostPolicy](../../include/luisa/tile/bridge/tirx/planner.h)。状态边界见现有 [IR 文档](../../docs/source/internals/tile/ir.md) 和 [calculus 文档](../../docs/source/internals/tile/calculus.md)。

### 9.2 一个具体的覆盖面风险：语义操作展开后再次匹配

当前 TIRx bridge 的 `_lower_mma` 会把 typed MMA 展开为初始化和 contraction 循环，同时保留 MMA permission annotation。Metal matrix 路径随后在循环形状中识别 matrix 候选：[MMA lowering](bridge/tirx/lower.cpp)、[matrix matcher](bridge/tirx/matrix.cpp)。

```text
typed TileIR MMA
       |
       v
annotated initialization + scalar contraction loops
       |
       v
recognize a supported loop / expression shape
       |
       v
matrix planning and target realization
```

该 matcher 有明确的结构边界：指定 annotations、可识别的静态矩阵域、初始化与 contraction 的语句组织、特定 accumulator-plus-product 形式，以及相关 predicate／alias 条件。它已支持投影额外的 unit axes；已有 `MatrixInitializerMaterializer` 也会为部分纯初始化表达式创建可识别的中间表示，不能忽略这些扩展而说它“只能处理最简单 GEMM”。

**但它仍揭示一类普适问题。** 一个合法 MMA 若经过其他正确变换，不再符合这个形状，就可能失去该快速候选。改 cost model 不能把未被生成的候选选回来。这里说的是静态结构风险；本轮没有做消融实验来量化它对某个性能差距的占比。

**建议的改进。** 在完成关键 mapping／atom 选择之前，保留 typed 语义操作或足够稳定的 region signature；通用 canonicalization 可继续服务 fallback 和已有 bridge，但不应让特定循环形状成为全部高性能路径的唯一入口。这是 TensorIR／Exo 式“保留可变换语义”的直接启示，并不要求引入 MLIR。

并非上述检查全都应删除：precision、访问安全和 atom capability 是真实条件；纯粹因不同语句组织而失败则可能是覆盖面不足。诊断应区分 `semantic violation`、`unsupported realization`、`unknown legality` 和 `unprofitable`，避免把所有 fallback 都叫作“证明失败”。

### 9.3 Policy 分离是对的，下一步是让搜索对象可组合

现有 `ExecutionCostPolicy` 明确把 hard limits 和数值权限留在 bridge 侧，policy 只影响性能评价。这正是应保留的边界。接下来更值得投入的是：

- 给候选保留 phase 边界、资源／distribution 接口和可实现性说明，而不只是一个算子名字与参数组。
- 让 native lowering 与 bridge 共用语义契约，但允许不同 target atom、资源能力和调度实现。
- 把 scalar fallback、subgroup collective、matrix atom 等看作有前提的实现族；是否选择由成本和上下文决定。
- 保留 source contract 与 derived fact 的来源，避免 lowering 后丢掉 `parallel` 已提供的独立性，又把它从地址公式重证一遍。

这里没有建议让 CPU、Metal、CUDA 共用相同 launch binding。应共用问题描述、变换和搜索接口，而不是强制不同硬件具有相同物理结构。

## 10. 如何证明“确实有改进”：实验与实施优先级

### 10.1 用模型反例驱动测试，而不只扩充算子名称

下表是**待实施的验证计划**，不是已通过的测试清单。每一项都应同时包含合法例和容易误优化的反例。

| 情形 | 验证重点 | 对应的实际算子 |
| --- | --- | --- |
| 同一执行上下文中 A、B、acc 的不同 layout／资源 | 不能从 hierarchy 推出唯一 memory mapping | GEMM、CNN、attention |
| 纯语法 cut 的消除，与有资源／carry 观测的 cut 重定位 | 不多保留无意义层级，也不合并不同实例 | elementwise、nested loops |
| pointwise sibling fusion 与邻域读取 | 不把 scope 内独立性误用为 scope 间独立性 | residual、stencil、传统 filter |
| 保序非交换 merge、浮点重结合和 prefix 输出 | reduction、recurrence、scan 的边界 | loss、softmax、scan |
| 一个 producer、两个异步 consumer、循环 buffer 复用 | ready 不等于 reusable，release 需要实际 join | multi-stage matrix pipeline |
| MMA → reduction／elementwise → MMA | phase 重映射收益与转换／资源代价 | attention、fused MLP |
| masked／ragged tail、不同 stride、索引置换 | 覆盖面不能只存在于整齐的方阵 | 大小／长宽比各异的矩阵 |
| compare-exchange、gather/scatter、多阶段合并 | 普通冲突不隐式变成 reduction；映射可服务非 MMA 算子 | sort、top-k |
| 同一语义程序的 SIMD 与 Metal 实现 | 复用语义不等于复用同一硬件绑定 | 上述可支持算子的交集 |

阶段不同就需要转换并不意味着必须 materialize 到 global memory；反过来，“fused”也不保证转换免费。测试应检查实际生成的交换、barrier 和存储位置。

### 10.2 把改进拆成三个可以被证伪的假设

**H1：稳定的区域语义扩大高性能候选覆盖面。** 对保持语义的变体——单位维度插入、坐标置换、等价初始化和合法 sibling 组合——比较候选覆盖与生成代码。若仍需逐算子添加名称／形状特判，通用性目标就没有达到。

**H2：phase 级联合 planning 比固定全区域 mapping 更有效。** 在相同 emitter、精度、调优预算下，对比统一 mapping、局部贪心和保留兼容候选的组合搜索。既测转换减少的情形，也测转换开销超过收益的反例。

**H3：policy 与搜索解耦能在未参与拟合的数据上泛化。** 在部分 shape／算子上校准，在保留的尺寸、长宽比、算子及另一后端上评价；后端可以有自己的校准参数，但不应靠测试集专用白名单获得结果。

最重要的消融包括：提前固定 layout、强制所有 phase 同一粒度、保守延长异步生命周期、只保留一个局部 winner、移除后端校准。每个消融只改变一个决策点，才能解释改善来自哪里。

### 10.3 测量不能把不同层次混成一个 speedup

建议同时记录：

- 正确性与能力：契约检查结果、有效候选数、失败原因、使用的 atom／collective，以及与参考实现的数值差异。
- 代码与资源：转换次数、barrier、版本数、register／shared-memory 占用；明确哪些来自编译器估计，哪些来自生成代码或 profiler。
- 搜索：捕获、分析、求解、编译和实测预算；小问题对有界 exhaustive／ILP oracle，大问题只报告已知 best-found。
- 性能：设备侧 kernel 时间与 host 到 completion 的 dispatch 时间分开。GPU command-buffer 总时间不能无条件称为单个 kernel 时间；SIMD 也应区分本体执行与队列／线程池开销。
- 泛化：held-out shapes／operators 的结果、退化案例、模型排序误差及相对于实际测过候选的 regret。

与 MPS、MPP、Torch、BLAS 比较时，应固定 dtype、精度许可、输入 layout、语义和计时范围，并注明 Torch 是 eager、compiled 还是调用 vendor library。它们回答工程性能目标；相关工作中的其他 DSL／solver 则回答机制与新颖性，不能互相替代。profile/capture 用来解释差距时，还应在不 capture 的情况下复测，检查观测扰动。

不要求在这台 Mac 上运行所有 NVIDIA 专用研究系统。可先比较本机支持的路径，并在同硬件实验条件具备后做对照；不能用论文里的另一代 GPU 数字来宣称胜负。

### 10.4 建议的工作顺序

| 优先级 | 交付物 | 为什么现在做 |
| --- | --- | --- |
| P0 | 保留 typed region 语义；记录候选准入／拒绝原因；增加语义等价变体测试 | 先让合法的好方案进入搜索，定位结构性缺口 |
| P1 | version／completion／release 接口；phase 边界候选；小规模联合选择 | 连接 execution、layout、memory 与 pipeline，而非各自优化后补救 |
| P2 | 参考解释器／检查器，关键规则证明，有界 oracle，跨形状／算子／后端消融 | 使严谨性、搜索质量和泛化成为可验证交付物 |
| 后续 | 动态事件、persistent／分布式执行，以及更广的不规则程序 | 在已有语义和实现保证上扩展，不干扰独立 Tile kernel 的当前目标 |

P0 不需要等完整论文定理；P2 也不能被永久替换为“又多跑了几个 GEMM”。实现与形式化可以互相反馈，但完成一个层次的工作时应只声明那个层次的成果。

## 11. 最终判断与阅读顺序

**最值得借鉴的组合是：Hidet 的任务组合、Cypress 的资源与事件、Prism/Bundl 的参与者契约、成熟 layout algebra 的明确片段、TensorIR／Exo 的可检查 refinement，以及 Twill 的联合调度约束。** 这些思路相互补充，但把它们放在一张图里还不是一个新定理或一个优秀编译器。

我们最有机会做出增量的地方，是**面向 effectful Tile SSA 的可组合 execution refinement**：在明确语义接口下，允许 phase 改变参与者、分布和资源实现，并把布局转换、异步复用和后端 atom 选择一起纳入有预算的搜索。需要证明的是组合边界足够、检查可执行、实现可落地；需要实验说明的是它确实覆盖更多好方案，而不是单个 kernel 的手工补丁。

目前不能说这是无人研究过的方向，也不能说我们已比上述系统更严谨或性能更好。更可信的论文主张应限定为：**在一个明确片段内，提供某组已有系统没有共同给出的组合保证，并用实际后端和保留测试集验证其收益。** 是否真的满足“已有系统没有共同给出”，仍需针对最终贡献补充更精确的文献对照，不能由本次调查直接判定。

若按实现价值安排下一轮精读：

1. Cypress §4：事件维度、资源 placement、copy 消除与复用；与我们的版本接口逐条对齐。
2. Hidet §5.1、Prism/Bundl 核心 calculus：分别检查坐标组合和合法参与者，两者不能混为一谈。
3. Tawa protocol 与 Twill §3–§6.1：从正确的异步生命周期，一直检查到 solver 方案怎样真的变成代码。
4. TensorIR block signature、Exo 2 scheduling action：设计我们内部可组合的分析／变换接口。
5. Linear Layouts、Axe/TIRx、Hexcute：列出可复用的布局片段、bridge 兼容条件及真正要联合求解的边界。
6. ATL 和符号 Prism：确定应证明的规则、允许的数值语义，以及哪些参数化分析能跨 JIT 实例复用。

所有文献链接位于相应分析段落。特别保留四个证据边界：Graphene 仅核对作者摘要；FreeTensor 的论文链接失效，此处仅据作者项目说明分析；Futhark 的案例来自作者工程记录而非理论定理；论文／官方接口的能力不代表仓库当前固定依赖版本已经集成。本报告不替代正式 DSL 规范，也不在 `docs` 下新增并行的说明体系。

## 12. 后续审查：把结构化语义与时空资源求解真正接起来

### 12.1 总体判断：保留模型方向，扩大实现的可选择空间

**我们要设计的是足够 flexible、同时保留有用 structure 的编程模型；编译器负责求解它到硬件的时空资源映射。** 按这个标准，当前前端和可变 TileIR 值得保留。主要缺口是 implementation freedom 尚未贯穿候选生成、资源规划、成本评价和 emitter，而不是需要再发明许多面向特定算子的 DSL 原语。

本节接续同日工作树审查，覆盖普通 reduction、Metal/TIRx cooperative mapping、独立 native MPP 入口以及 XIR/SIMD mapping。源码检查比 §9 更细；性能引用的是已有的冻结实验，不是本轮重新测量。所选 TIRx 构建树的全量构建通过；本轮没有修改编译器、没有新跑 kernel 性能测试，也没有验证全部后端正确性。

可以把目标明确成：

```text
P: structured regions + logical domains + values/effects + semantic contracts
H: hardware capabilities + resource limits + available atoms/protocols
W: work assignment + value distribution + placement + schedule + atom choices

Legal(P, H, W): W refines P and is realizable on H
W* = argmin predicted_makespan(W, H), for admitted W in a bounded vocabulary
```

这里的 structure 是分析和组合的依据，不是不可变的物理层级。**execution-first 不等于 execution-fixed。** 一个没有被资源身份、carry、effect 或 collective 观察到的 nest cut，可以被合并或重新分解；同一执行上下文中不同 value 仍可选择不同 distribution 和 storage。hard constraints 与 numerical permissions 先决定合法集合，cost policy 只能在其中排序，不能把“不合法但很快”的实现变成候选。

这与 §2 的 Cypress/Hidet、§3 的参与者契约以及 §5 的调度研究相接。潜在贡献不是“首次做硬件映射”，而是面向 **effectful Tile SSA 的可组合时空资源 refinement**：边界契约足够、组合可检查、候选确实可生成，并在未参与调优的程序上兑现收益。现在还不能把这个目标写成已完成的新颖性或完备性结论。

### 12.2 优先问题一：归约数值权限仍与后端优化开关混在一起

**这是语义建模缺口，优先级高于继续拟合 cost coefficients。** 当前 `Nest::reduce` 只接收 domain；`Operation` 为 MMA 保存了 typed numerical policy，但普通 REDUCE 尚无对应的逐操作字段。TIRx 的 `match_reduction_contract` 通过严格的 body 形态识别 FP32 add/max/min，生成一个整数 kind annotation；它不是完整的 reducer contract。

源码依据：[reduce 捕获](dsl.cpp)、[Operation 的 typed policy](../../include/luisa/tile/ir.h)、[归约识别](bridge/tirx/lower.cpp)。当前 verifier 的结构检查也主要验证 domain、region、carry/type 对应，而不是归约代数：[结构验证](verifier.cpp)。

更关键的是 `metal_subgroup_reductions` 的注释明确说，它同时是 **数值授权和候选开关**：[PlannerOptions](../../include/luisa/tile/bridge/tirx/planner.h)。默认关闭且明确 opt-in，不能据此指控当前实现擅自放宽了默认数值语义；但这种组织无法在同一 kernel 中自然表达“这一个 reduction 保序、另一个允许重结合”，也不利于把同一语义传给 SIMD、TIRx 与 native MPP。

建议拆开：逐 reduction 保存有效语义契约，backend option 只启用／禁用某个实现族。语言默认 `unordered_tree`，用户可用局部策略收紧；若提供 kernel 级默认策略，捕获后也必须解析到每个操作。显式更严格的局部约束不能被 target、autotuning 或更宽松的全局默认覆盖。识别到 `ADD` 只说明 update 的形态，不等于证明 FP32 加法满足精确结合律；默认树语义提供的是重结合／重排许可。

### 12.3 优先问题二：cooperative storage 仍由分发深度过早决定

`GroupWorkloadAnalysis` 将 `_lane_depth == 0` 的 allocation 累加进 shared-memory 预算；mapper 对这些临时值选择 shared，对分发内部的临时值保留 private。显式 memory 约束也只能匹配这两个预定选择：[资源分析](bridge/tirx/cooperative.cpp)、[资源实现](bridge/tirx/cooperative.cpp)。

这是一套保守、受限的实现，不是源模型必须具有的限制。实际 decode shader 中，坐标、布尔 mask、小标量和 score/probability 都有 threadgroup 中间数组，很多 phase 之间使用整组同步。因而“都 fuse 在一个 kernel 中”仍可能不断经过 shared-memory publication。

借鉴 Cypress 的资源版本与通信依赖、Hidet 的工作/数据分布以及 Tawa 的生命周期协议，下一步应逐 value、逐 phase 决定：保留寄存器中的同 worker 值、重算便宜 index/mask、实现必要的 subgroup exchange，还是 materialize 到 shared。容量评价应逐步从 allocation 总和改为具体方案的峰值 live storage，并区分 producer completion 与所有 consumer release。

**不能直接删除这些 barrier。** 当前分布下不少 barrier 确实承载跨 worker 的 reaching-value 依赖。应该先改变 placement/distribution，再用读写与版本依赖消除不再需要的同步；异步 slot 复用还要证明最后一个消费者已经释放。

### 12.4 优先问题三：组合程序的 reduction 成本尚未进入共同搜索

cooperative workload 对匹配到的矩阵建模，但对其他 element domain 主要累加输出数量。一个输出内部很长的 contraction/fold，不能由“一个 independent element”准确代表。若没有匹配到矩阵，`plan_group` 直接返回参考映射：[workload 摘要](bridge/tirx/cooperative.cpp)、[无 matrix 的分支](bridge/tirx/planner.cpp)。

组合程序中的 `try_metal_reduction_tile` 则在 group plan 已确定后的 mapper 中选择：[collective emission](bridge/tirx/cooperative.cpp)。独立 row-reduction 路径已有更细的 reduction cost hook、条带和 packing 搜索，不能说项目“没有 reduction planner”；问题在于这些能力还没成为组合区域的共同候选。

应让 region 摘要携带 fold 长度、输入访问、局部 stripe、collective 数量、参与者和边界 distribution，并把 reduction/scalar/MMA 实现放入同一受约束问题。phase 可以采用不同粒度，但转换不免费：必须比较交换、barrier、live storage 与少量参与者之间的串行关键路径。

现有 Pareto solver 对声明的可加目标和单个 shared-capacity 约束有清楚的适用边界：[frontier](bridge/tirx/planner.cpp)。不能把“目标模型覆盖不足”说成“这个有限求解算法本身错误”。扩展后，仅比较 score 和 released bytes 不再足够；不同边界分布／生命周期的候选不能过早互相支配。

### 12.5 优先问题四：表示变体有 first-success 选择，快速 atom 仍依赖形态

initializer materialization 是有实际价值的覆盖面扩展，但当前流程先尝试 materialized 程序，能成功就直接返回，只有异常时才尝试原程序：[候选入口](bridge/tirx/cooperative.cpp)。虽然会给 materialized 程序内部的工作和存储计分，却没有把原始表示与新表示一起比较。因此它还不是 representation-level 的联合寻优。

建议同时保留 original、materialized，以及 emitter 真正支持的 fragment-resident 变体，再比较同一目标。不需要立刻上大规模 ILP；先让相同语义、不同表示的候选都能进入搜索。§9.2 所述 typed MMA/region signature 的保留，也是为了避免等价语句组织使好候选消失，而不是取消必要的 dtype、predicate 和 alias 检查。

独立 native MPP 路径还要分开评价：它已经直接消费 typed MMA，但当前要求同 scope、sole-use、zero-init、完整 K，并明确不支持 K pipeline：[native MPP 准入](../../src/backends/metal/tile/metal_tile_codegen.cpp)。这说明“只保留 typed op”也不够，必须继续扩展可实现的组合接口。它与下节冻结数据中的 `TIRx -> Metal; mpp=false` 不是同一条路径。

### 12.6 优先问题五：SIMD 缺少 Tile 内部的映射候选

XIR planner 当前枚举 root axis permutation 和 block width，明确保留每个逻辑 worker 的完整 Tile 程序：[候选空间](../../include/luisa/tile/bridge/xir/planner.h)。lowering 将 root parallel 绑定 dispatch id，内部 nest 转成循环；MMA 在 bridge 中按输出和 contraction 展开标量 SSA：[root 映射](bridge/xir/lower.cpp)、[MMA 展开](bridge/xir/lower.cpp)。

不能因此说 SIMD 后端没有 vectorization：它可以 packetize root workers。缺少的是 **选择 Tile 内哪一维进入 packet、哪一维成为线程任务或寄存器局部循环** 的能力。单纯修改 arithmetic/gather 系数，无法选中根本不存在的 within-Tile 候选。多个 sibling root scope 目前也被显式拒绝，通用 fusion 仍是设计目标，不是这个 bridge 已有的能力。

优先增加独立维度的 split/fuse/reindex 与 packet-axis 候选，并保持源 `parallel` 的非冲突契约；需要额外检查的是变换新增的映射、别名和 collective 义务，不是把用户已承诺的整个并行语义重新证明一遍。然后再接 packed microkernel 或 provider，并单独标明 direct lowering 与调用 BLAS 的收益。

### 12.7 冻结实验支持“覆盖面问题”，但尚不能给出瓶颈百分比

数据来自 Apple M1 Max 的 FP32 attention，fast math 与 relaxed precision 均关闭，归约树显式开启。三条比较路径是当前 Metal/TIRx cooperative、冻结的上一版同路径、Torch/MPS；每个 shape 采用六种执行次序、每轮五个样本。记录中的 `native` 是当前 Luisa 测试路径标签，**这些记录明确为 `mpp=false`，不是独立 native MPP 的性能结果**。

以下时间是无 encoder instrumentation 的 **GPU command-buffer 区间之和，按 batch 重复次数归一化**，不是单个 isolated kernel 的时间；Q/K 表示 query/key 序列长度，括号内比值为逐轮配对后取中位数，不能用两个汇总中位数的商代替。

- B=1、Hq=4、Hkv=2、D=Dv=64，Q/K=64/128：当前 35.08 μs，旧版 79.46 μs，Torch 33.14 μs；当前/旧版为 0.441，当前/Torch 为 1.058。生成源码的 matrix intrinsic 调用点由 1 个变成 3 个，group width 也从 64 变为 128；这是表示候选扩展和后续重新规划的共同变化，不能把全部收益归因于某一个因素。
- 同一 B/head/D 设置，Q/K=37/193：当前 151.27 μs，旧版 151.41 μs，Torch 43.87 μs；当前/Torch 为 3.447。两边没有 matrix intrinsic，均为 512 threads/group。这是 ragged 形态没有命中快速候选的具体证据，不是仅仅“命中了但参数稍差”。
- 同一 B/head/D 设置，Q/K=512/2048：当前 1165.91 μs，旧版 2804.73 μs，Torch 337.62 μs；当前/旧版为 0.415，当前/Torch 为 3.455。候选扩展有明显收益，但离目标仍远。
- B=1、Hq=8、Hkv=2、D=Dv=64，decode Q/K=1/2048：当前 371.53 μs，旧版 378.17 μs，Torch 38.39 μs；当前/Torch 为 9.633。新旧生成源码相同且没有 matrix intrinsic，已有 `simd_sum`/`simd_max`，所以本次 MMA initializer 扩展没有覆盖到这个差距。

来源：[prefill 原始结果](../../scripts/benchmark/tile_torch/results/m1-max-20260907-composed-matrix/prefill/results.json)、[decode 原始结果](../../scripts/benchmark/tile_torch/results/m1-max-20260907-composed-matrix/decode/results.json)、[decode shader](../../scripts/benchmark/tile_torch/results/m1-max-20260907-composed-matrix/decode/attention-1x8x2x1x2048x64x64-r0-native.metal)。这些是既有实验记录，不构成当前整个 dirty tree 的新一轮性能认证，也不是 MPS 独立基准或 BLAS 结论。

原报告的数据质量检查发现，[原审计脚本](../../scripts/benchmark/tile_torch/results/m1-max-20260907-composed-matrix/audit.py) 错把每一个 native prefill 都预期为 materialized matrix 路径，因 ragged case 不满足而失败。**该历史失败必须保留，不是数值正确性失败。** 当时另行核对了 72 条唯一、标记有效的记录、GPU 原始样本／重复次数／计时范围、48 份非 Torch 源码哈希，以及 throughput/latency 的路径中位数与配对比值；这些核对一致。

后续同日检查点修正了 matrix-hit 的 case 分类；现有本地 `audit.json` 记录 `status=pass`、7 项负向审计。本次仅确认该记录，没有重跑性能或完整数值 oracle。临时完整输出未归档，无法仅凭哈希重做 FP64 数值验证。上述原始实验目录仍是本地未提交证据，不随本次文档检查点发布；路径用于本机追溯，不能当成已经归档到 Git 的复现材料。

此外，compute-counter instrumentation 相对于 control 的 command-buffer 时间扰动并不对称：小 prefill 中当前路径约 1.63 倍、Torch 约 2.02 倍；decode 中约 1.08 倍与 1.88 倍。因而插桩 compute-pass 时间只能作为带扰动的诊断，不能直接替换 control 宣称“纯 kernel 加速”。后续需在可控单 dispatch 或明确的 pass/dispatch 对应关系下 profile，并用未插桩复测核对。

**现在可以确认实现选择受限；尚不能量化 shared traffic、barrier、访存不合并、低并发各自占了多少时间。** 那需要 counter/capture 和单变量消融。不能从源码里看见 barrier 就把所有差距归因于 barrier。

### 12.8 reduce 的最小语义修订：分清 fold、重结合与重排

用户提出 fold L/R 或“随意”是有必要的补充。建议为已有 `reduce` 增加语义 policy，不增加四种 nest primitive。以下名字是**设计草案，当前头文件未实现**：

- `reduction::fold_left`：严格采用 `op(op(op(z, x0), x1), x2)` 的参考更新结构；不要求结合律或交换律。
- `reduction::fold_right`：严格采用 `op(x0, op(x1, op(x2, z)))`；也不要求结合律或交换律。它不仅是倒序执行 `acc = op(acc, elem)`，还涉及 reducer 的操作数方向。异构 state/element 时，左右 fold 的类型要求也可能不同。
- `reduction::ordered_tree`：允许改变括号，保留 logical contribution 的先后次序；精确变换需要可用的结合律，浮点则需要显式数值授权。
- `reduction::unordered_tree`（默认）：允许重结合和重排贡献；精确等价通常需要结合律与交换律，或其他足以保证置换不变的契约；浮点归约的变化则由该策略许可。它不是任意改变值、漏算或重复计算的许可，也不是“保证最快”。

fold 的左右方向以 [WG21 P2322R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2322r6.html) 的含义为参照。硬件库也确实需要区分归约代数：CUB 的 [BlockReduce 实现族](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_reduce.cuh) 同时提供只支持交换归约和支持非交换归约的算法，不能把“有结合律”直接等同于任意 lane striping。

一个保留现有 nest/赋值风格的提案是：

```cpp
// Proposed API, not yet implemented.
// x is an existing Tile<float> with domain shape(k).
auto acc = Scalar<float>{0.0f};
for (auto &step : nest.reduce(shape(k))) { // unordered_tree by default
    acc += x.at(step);
}
auto ordered = Scalar<float>{0.0f};
for (auto &step : nest.reduce(shape(k), reduction::fold_left)) {
    ordered += x.at(step); // preserve the specified update chain
}
```

`x.at(step)` 是已有 Tile 的纯投影，不是 MemoryRef 的隐式 load；赋值仍由 frontend 捕获 carry。默认调用的现有拼写不变，但其默认语义迁移和显式 policy overload 尚未实现。示例不另发明 accumulator proxy、`result()` 或 ownership 语法，落地时应提供 compile-checked 示例固定拼写。对保序策略，多维贡献次序采用 `IndexSpace` 轴顺序对应的词典序，不能让线程编号或内存 layout 暗中定义它。

右 fold 的 body 应按 element/state 的方向写更新，并逆序访问参考贡献。例如 `[1,2,3]`、seed 为 0 时，左 fold 的 `acc = acc - elem` 得到 -6，右 fold 的 `acc = elem - acc` 得到 2。policy 不会偷偷交换用户表达式的操作数；仅反向运行 `acc - elem` 不等于数学右 fold。

这几个预设背后，IR 应分开保存 **参考 fold／贡献顺序、允许的重结合与置换、state/contribution/merge 类型及身份值、数值与确定性约束**，而不是把它们压成“优化强度”的一个整数。左右 fold 不是谁比谁更强；固定 fold 是依赖链，保序树与无序树扩大的是另一维度的合法实现集合。

尤其要处理下面六条边界：

1. **结合律和交换律是事实／契约，重排许可是用户选定的语义。** 内置库可以注册有条件的法律，用户自定义 reducer 可以声明需要信任的 contract，验证器检查类型、纯度、结构和已知前提。有限随机测试只能找反例，不能证明任意 C++ 函数的结合律。FP32 add 的 tree policy 是数值放宽，不应伪装成精确结合律证明。
2. **seed 不等于 identity。** 非零 incoming state 只应进入聚合一次，不能在每个 worker 的 partial 中重复加入；其左右位置由参考契约决定。空域、mask、整数溢出、NaN、signed zero 和 argmax tie-breaking 都必须有定义。仅在 padding identity 满足该数值语义时才能填入空 lane，否则使用有有效位的 partial。
3. **保序树必须保持叶子次序。** 例如四个贡献 `a,b,c,d`，将偶数项交给一个 worker、奇数项交给另一个，再合并为 `(a op c) op (b op d)`，已经改变顺序。没有交换许可时，应生成连续分块／保序合并，或显式转换；不能沿用当前 striped emitter 就假定正确。
4. **一般 fold 不自动具有并行 merge。** `update : State × Elem -> State` 并不提供 `merge : State × State -> State`。要树化，需要可检查或声明的 `lift/merge` 契约及其同态关系。严格 fold-only 可保留，不应强行声称所有 recurrence 都是 monoid。带可见前缀或任意副作用的通用顺序程序仍应使用 `serial`／相应 scan 库。
5. **浮点确定性是独立要求。** 固定树可以 run-to-run 确定，但不同 JIT 参数／设备可能选择不同树；严格左右 fold 也不能单独保证跨设备所有浮点行为一致。若要求跨配置或跨后端复现，必须约束算法、算术语义及原子实现。CCCL 的 [determinism 要求](https://nvidia.github.io/cccl/unstable/cccl/determinism.html) 对保证范围作了类似区分。`fast_math=false` 不会撤销另一个显式给出的归约树授权。
6. **权限是局部的。** 多个独立 state 可以组成 product reducer，但不能因为 sum 允许重排就把另一个保序 state 的贡献重排；需分别实现或采用共同满足的约束。存在状态耦合时，必须用整体 reducer contract，不能根据两个 `+=` 分别猜测。

因此，正式文档现已修订“缺少代数律时 reduce 一律 ill-formed，只能 serial”的过强表述：**显式严格 fold 可以存在，但没有兼容 merge 就不能树化。** 按最新讨论，普通 `reduce` **默认 `unordered_tree`**，不要求用户为普通 sum 另写 fast-math 或树化授权。默认策略描述允许的计算集合，不强制使用某一种树或强制并行；串行实现也可以是合法候选。无法确定兼容 merge 的自定义更新应诊断，并提示提供契约或显式选择 fold，而不是静默猜测。

该默认值只放宽归约的重结合／重排，不隐含降精度、近似函数、FMA contraction 或忽略 NaN。单独的严格浮点操作模式不会撤销归约权限；反过来，用户标注的严格 fold 也不能被后端调参放宽。这是一次明确的语言设计决定，当前实现迁移仍待进行，不能把更新文档说成已改变所有 backend 的默认行为。

这些 policy 有直接的性能价值：严格 fold 可以并行处理独立输出组和提前准备贡献；保序树可以生成保序分块；无序树可以考虑 striped loads、多 accumulator、subgroup collective、跨组 partial。任何分块还须保持 seed/贡献计数和数值契约。decode 的 split-key 方案则额外需要 online-softmax 状态的合法 merge，以及 kernel/dispatch 边界代价，不能只把现有时间循环改标为 parallel。

### 12.9 实施和验证顺序

本次文档整理已将默认语义、IR 状态边界和候选接口分别纳入既有 language／internals 页面，并更新归约示意图；这不是 compiler 实现或形式证明的完成。接下来建议按下面顺序落地，而不是先替换求解器：

1. **语义准入可检查。** 为 REDUCE 增加逐操作 contract/policy，保留 MMA/region signature，记录每个候选为什么被拒绝。区分语义不允许、尚无 emitter、判定未知和预测不划算。
2. **补齐关键候选。** generic ragged/masked matrix realization、matrix-vector/小 M 的不同 work decomposition、SIMD within-Tile packetization，以及组合程序的 reduction 候选。手写 kernel 可以当已知可实现的比较对象，但最终不能只留下算子名／shape 白名单。
3. **联合决定 phase 边界。** 候选携带入口／出口分布、live versions、参与者和协议；比较寄存器保留、shared materialization、重算及转换成本。先做有界局部枚举和兼容 frontier；证实组合复杂度有需要时再接 ILP 或其他启发式。
4. **校准并测泛化。** 用独立尺寸/算子拟合 policy，在 held-out shapes/operators 上比较 candidate recall、实测排序错误和 best-measured regret。严格对齐 dtype、reduction 权限、输入、设备计时与 E2E，报告退化和拒绝，不把插桩时间当无扰动真值。

最小语义反例集应包含：非结合纯 fold 的左右结果不同、精确结合但不交换的合并、FP32 对重结合敏感的输入、非 identity seed、空域/部分 lane、NaN/signed zero/tie、同 kernel 的混合严格/宽松 reduction、复制 layout 不得重复计数，以及一个 producer 两个 consumer 的循环 buffer 复用。性能消融应单独关闭 representation candidate、phase remapping 和版本复用，避免同时改变多项后只留下一个总 speedup。

仍待回答的问题是：这些 region 边界摘要对我们声明的有界片段是否足够、低层 emitter 是否真正实现了被验证的 plan、以及预测目标能否在未见过的算子上正确排序。**把这三点连起来，才是“flexible 又 structured”从漂亮设计变成严谨、实用工作的路径。**

## 13. 实现跟进：逐操作归约策略与最终编译边界

这一轮落实了 §12.2 的首要缺口，不是新的性能排名，也没有完成整体 execution calculus 的形式证明。

| 层次 | 已落实 | 保留的边界 |
|---|---|---|
| C++ DSL / TileIR | 原有 `reduce` 增加可选 `ReductionPolicy`；默认 `unordered_tree`，显式保序树、左 fold、右 fold；rewriter 修改策略会使分析缓存失效 | 不增加 nest primitive、accumulator proxy 或 `result()`；任意自定义 lift/merge 契约检查器仍未实现 |
| TIRx | 独立传递数值策略与归约 body contract；CPU array provider、Metal striped/subgroup emitter 只接受无序树 | 保序树暂用保序串行实现；未知自定义 body 不凭空生成 merge |
| XIR / SIMD | 保留词典序或逆词典序贡献遍历；严格策略关闭 SIMD 优化与 LLVM 编译的全局 fast math | 尚无一般 within-Tile 树化／packet 分布 emitter |
| Metal Runtime | 有顺序要求时关闭最终编译 fast math；无显式 TIRx 配置且设备支持时自动启用 collective 候选族 | 候选开关不授予数值权限；每个操作仍分别准入 |
| 独立 TVM Runtime | 通过小型原生 C++ precise-math 扩展，将要求保留到模块载入和 `MTLCompileOptions` | 未安装扩展时严格归约明确编译失败；Luisa Runtime 的源 artifact 路径不依赖该扩展 |

右 fold 特别区分 **遍历方向** 与 **body 操作数方向**：nest 的策略只决定逆序访问，绝不自动改写用户 body；expression-level `reduce(x, axes, reducer, fold_right)` 则由库调用 `reducer(elem, state)`。多维贡献的顺序来自逻辑 domain，而非线程编号或物理 layout。测试覆盖非交换更新、空域、正负零 seed、非 identity seed、二维顺序指纹，以及对重结合敏感的 FP32 消去数据。

### 13.1 两个有推广意义的失败

第一，**合法的 IR 并不保证最终机器代码仍合法**。TVM Metal runtime 的原始实现将 `fastMathEnabled` 写死为 `YES`；即使 lowering 输出串行循环，`[2^24, 1, -2^24]` 这样的消去用例也会暴露重排。修复把需求一路传到最终编译：Luisa 的 `DeviceArtifact` 携带 `requires_precise_math`；独立 TVM 模块的精确模式同时有 compiler/runtime capability 检查并被序列化保存；LLVM target 的全局 fast-math flags 也不得覆盖严格策略。这不是放宽测试容差，不涉及生成 Python 或修改 shader 源字符串。

第二，**只数输出元素会漏掉内部 collective 的执行宽度**。同一 group 内一处无序 sum、一处严格 fold 的用例，原 planner 因两个结果都是标量而分配一个线程，导致合法的 subgroup sum 无法出现。现在 workload analysis 与 emitter 共用同一个完整 body/policy matcher，为已准入 collective 的每个独立输出预留一个完整 subgroup，并按硬件上限限制宽度。显式线程限制仍可选择串行 fallback。这个修改与算子名、特定矩阵尺寸无关。

这里必须区分**映射准入修正**和**成本模型完成**：上述 composed-group 选择仍是受限的参考绑定，不是已经校准的串行／subgroup／跨 subgroup 联合最优解。已有 whole-row reduction 的成本模型不能直接冒充 mixed matrix/reduction phases 的总成本。下一步应将每个 phase 的贡献工作、转换流量、峰值 live state、参与者占用及同步需求组合起来，再比较候选。

完整回归还揭示了 phase fence 的另一条独立义务：共享中间结果与写入 global view 的结果都可能跨 phase 被消费。没有 effect/participant 证明时，不能将同时覆盖 device/threadgroup 的 fence 缩成仅 threadgroup fence。数值许可不等于内存排序许可。

### 13.2 本轮不声称什么

这些实现消除了归约调优的语义障碍，并增加了 composed-group collective 的可达性，但本轮未重测性能。因此没有新的 MPS/Torch 加速比，既有 attention/decode、较大 GEMM 与部分 normalization 的差距仍然成立。通用 lift/merge checker、保序树 emitter、XIR 内部 Tile 分布、边界摘要组合与 held-out 成本校准仍待完成。特别是任意自定义无序 body 的诊断尚未达到 §12 的设计目标；当前实现仅对已识别的 body 树化，其他 body 保守保留串行更新。
