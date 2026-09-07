# Execution calculus: related work and research questions

**Execution-first tensor programming is a promising design direction, not an
unexplored one.** Prior work covers hierarchical execution, independent
compute/data placement, compositional scheduling, verified rewrites and
constraint-based mapping. Layout algebra is one part of that history.

This focused survey was checked on **September 7, 2026** using author papers,
project documentation and publisher/author records. It follows references
around task hierarchies, schedule composition and mapping optimization; it is
not an exhaustive review or a proof of novelty. Comparisons use the named
versions and sections, not assumptions about every current repository feature.
Paper-reported performance is not our measurement. In particular, an older
paper's description of Triton does not describe today's Gluon.

The proposed [calculus](calculus.md) and [optimization problem](planner.md#formal-finite-optimization-problem)
have separate owners. This page records intellectual overlap and the evidence
needed to establish a contribution, not another language specification.

```{contents} On this page
:local:
:depth: 2
```

## Closest precedents

```{table} Directly overlapping research
:class: design-table

| Relationship | Closest precedents | Consequence for this design |
|---|---|---|
| Hierarchical work/data maps | Hidet, Graphene, LEGO | Layout nests describing execution already exist |
| Independent execution and resource placement | Cypress, Stripe, Tiramisu | Two independently mapped structures are not a new idea |
| Layout and instruction synthesis | CuTe, Linear Layouts, Hexcute | Reuse established spatial algebras and instruction constraints |
| Asynchronous communication and reuse | Cypress, Tawa, Pallas/Mosaic GPU | Completion and safe reuse need more than SSA use-def edges |
| Joint scheduling and resource optimization | Twill, CC 2022 mapping, CoSA | A formal solver/cost formulation alone is not distinctive |
| Compositional, checked transformations | TensorIR, Exo 2, ATL, Mirage | Region interfaces, rewrite safety and search guarantees have precedents |
| Fine-grained dynamic task dependencies | Event Tensor, Legion, DISTAL | Multiple scopes and device coordinates do not establish novelty |
| Extensible native lowering | TIRx, Gluon, Fireiron | A small core plus target-specific realizations is established practice |
```

The comparisons below separate **established mechanisms**, **what to borrow**,
and **what an improvement would require**. Proposed extensions are not claims
that another system cannot express the same program.

## Hierarchy, mappings and effect boundaries

### Hidet: a direct foundation for spatial/temporal factors

Section 5.1 defines a task mapping from each worker to an ordered list of
multidimensional tasks. Its `spatial` and `repeat` maps compose into nested
assignments, including macro-workers at different hardware levels. The
composition is associative, but swapping factors can change the mapping.
Section 5.2 separately addresses fusion around scheduled operators.
[Published paper, Sections 5.1–5.2](https://cdn.amazon.science/9d/e4/97ec5cb54d8ab2a2b65abf15cb7b/hidet-task-mapping-programming-paradigm-for-deep-learning-tensor-programs.pdf).

**Borrow:** use this construction for rectangular independent work and
hardware-sized mapping factors; keep scheduling recipes reusable across
shapes. Do not invent a competing product with different corner cases.

**Difference to test:** our logical child identity is not necessarily a
physical worker. We propose to preserve observed ownership/effects while
changing worker and time factors. That could make remapping larger effectful
regions easier to check, but needs a comparison beyond post-scheduling
prologue/epilogue fusion. A task product alone is not a memory-lifetime proof.

### Cypress: the closest execution/resource separation

Cypress separates task variants/privileges from processor and per-tensor
memory mapping. Its SSA IR has processor-indexed completion-event arrays:
indexed dependencies and joins survive hierarchy flattening. Copy elimination
rewrites dependencies, while physical buffer reuse adds last-reader-to-next-writer
edges. Warp partitioning and versioned pipelines lower events to synchronization;
they do not require a dynamic task runtime. Evaluation mappings were manually
tuned. [PLDI 2025 paper, Sections 3, 4.1–4.2 and 5.1](https://rohany.github.io/publications/pldi2025-cypress.pdf).

**Borrow:** carry readiness relations through remaps; removing a materialization
must not accidentally remove its publication join. Plan storage sharing
together with the serialization it introduces. Keep these mechanisms internal,
not public ownership/event boilerplate.

**Difference to test:** lexical Tile SSA could infer routine effects and expose
mutable region bodies for compositional rewrites. Our current planner does not
yet deliver that generality. Execution/resource separation itself is shared
ground; C++ syntax and avoiding MLIR are engineering choices, not stronger
semantics or research contributions.

### Stripe: a particularly close precedent for primitive contracts

Stripe's parallel polyhedral blocks permit sequential statements inside each
instance and prohibit cross-instance read/write dependence, with designated
associative, commutative aggregation as the exception. Nested blocks derive
accesses from ancestor indices; refinements describe buffer subregions.
[2019 paper, Definitions 1–2 and Sections 3.1–3.3](https://arxiv.org/pdf/1903.06498).

**Borrow:** independence belongs to the representation's contract, and nested
access/refinement information should remain available to transforms. This is
very close to our ordinary parallel noninterference contract, not merely an
earlier memory-layout system.

**Difference to test:** separate ordinary parallelism from reduction algebra
and ordered recurrence. Our proposed reduction rules admit order-preserving
associative, noncommutative folds; asynchronous version lifetimes need a
separate protocol. This is a difference from the cited block definition,
not proof that every current Stripe extension lacks these possibilities.
Putting all computation in a no-dependence block would be too restrictive.

### Graphene and LEGO: execution layouts are not unexplored

Graphene represents both data and threads as hierarchically tiled tensors,
with computations expressed through their mappings. Its intended position is
a low-level tensor IR. The checked source here is the
[author's ASPLOS 2023 abstract](https://mgarland.org/papers/2023/graphene/);
the publisher full text was not retrievable in this audit. Do not infer absent
transformations or formal guarantees from that abstract.

**Borrow/test:** represent participants and data in compatible typed coordinate
spaces. A proposed TileIR refinement should survive comparison with Graphene's
actual mapping examples before being called more expressive.

LEGO derives indexing from composable computation/data layouts, including
user-provided permutations and inverses. Its main algebra is bijective, with
specified extensions for partial tiles and some non-bijective maps. It can
integrate with multiple code generators.
[CGO 2026 paper, Sections III–IV](https://users.cs.utah.edu/~tavak/assets/pdf/LEGO-CGO26.pdf).

**Borrow:** explicit composition domains, forward/reverse mapping contracts and
tail handling. **Boundary:** accepting an arbitrary user permutation does not
automatically prove its bijectivity. Neither a global bijection nor equivalent
indexing alone establishes preserved resource ownership or happens-before.
Our additional obligations should attach only where those observations exist.

## Layouts, tensor atoms and the control boundary

### Hexcute: constraints already connect layouts, atoms and time

Hexcute synthesizes CuTe-based thread-value and shared-memory layouts from
instruction constraints; it also ranks candidates with issue/completion costs.
Its interface leaves dataflow and placement explicit. The revised paper notes
consistent-thread-arrangement annotations for connected GEMMs and layout-search
scaling as limitations.
[v3, January 2026, Sections III–VI and IX](https://arxiv.org/html/2504.16214v3).

**Borrow:** derive constraints from supported instructions, propagate them
across value edges, and retain conversion costs. A scalar fallback is useful
only if its actual work enters the cost model.

**Difference to test:** search placement and participant alternatives along
with layouts across a whole region boundary. That is a hypothesis, not an
implemented advantage. We must beat independent local choices without
exponential compile-time growth; claiming a monolithic global solver does
not solve this problem. A different public syntax contributes little here.

### CuTe and Linear Layouts: complementary, bounded spatial algebras

[CuTe's official algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
defines hierarchical shape/stride layouts and operations such as composition,
product and division. It supplies a mature base for mixed-radix coordinate
manipulation. [Linear Layouts, v5, March 2026](https://arxiv.org/abs/2505.23819v5)
represents GPU distributions using linear maps over bits, enabling systematic
conversion and communication generation.

**Borrow:** preserve proven meanings inside their supported fragments, not
one universal layout representation with undocumented exceptions. Integer
shape/stride algebra and GF(2) algebra are not interchangeable.

**Our obligation:** type source/destination spaces and active domains; use a
relation for replication instead of inventing an inverse. Spatial layout
equality cannot certify reduction accounting, effect order or async completion.
Supporting a larger map vocabulary is useful only with efficient normalization,
legality checks and emitters. It is not automatically a more rigorous model.

### TensorIR and TileLang: structured regions and useful automation

TensorIR blocks expose iterator domains, reads/writes and reduction
initialization, separating outer scheduling from inner tensorized bodies.
Signatures support transformations without inspecting every scalar operation.
[ASPLOS 2023 paper, Sections 3.1–3.3](https://arxiv.org/pdf/2207.04296).

**Borrow:** summarize a region's boundary rather than scalarizing it for every
pass. Our proposed interface adds participant convergence, completion/version
requirements and numeric permissions; we must show why those additions enable
specific transformations. A region signature itself is not new.

TileLang combines explicit tile allocation/dataflow with layout inference,
thread binding, tensorization and configurable pipelines.
[April 2025 v2, Sections 3–4](https://arxiv.org/html/2504.17577v2).

**Borrow:** make common operations convenient while retaining expert controls.
**Different default:** our ordinary load produces Tile SSA and lets the
compiler plan temporaries; manual Memory remains available. To establish a
benefit, the automatic plan must match an expert's resource placement and
avoid hidden copies. Fewer allocation lines without comparable performance
are not an improvement. This versioned comparison does not assert limitations
of the latest TileLang compiler.

### TIRx: a compatible lower boundary, not our automatic planner

The June 2026 TIRx design keeps orchestration hardware-native and dispatches
tile primitives using execution scope, operand storage layout and target.
Its logical-to-physical layouts explicitly model shard, replica and offset on
named hardware axes. Higher-level scheduling and allocation are optional.
[Official design, programming model and lightweight backend](https://tvm.apache.org/2026/06/22/tirx).

**Borrow:** a direct, inspectable lowering path and extensible atom dispatch.
**Our position:** a logical execution-first frontend can produce these explicit
decisions. It need not make storage-first TIRx the public DSL, nor replace it.
The bridge must preserve the realized placement relation, including replicas
and participant context; it cannot reinterpret a many-to-one distribution
as an ordinary inverse. See the proposed
[adapter obligation](calculus.md#bridge-compatibility-is-a-relation-check).

This is a good abstraction boundary, not evidence that TIRx supplies our
whole-plan solver. The official design is also not proof that our pinned
bridge implements every currently documented primitive.

### Gluon and Pallas/Mosaic GPU: expert control already goes beyond tiles

[Gluon's official overview](https://triton-lang.org/main/gluon/index.html)
exposes layouts, shared memory, warp specialization and target features.
Its [layout tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)
describes explicit register/lane/warp distributions. Do not claim modern
Triton has no way to control these relationships.

**Borrow:** an expert path should be inspectable and should constrain the
same implementation choices as automation. Our proposed logical defaults may
reduce hardware-specific source, but must retain useful overrides; otherwise
they can be *more rigid* than Gluon in practice.

[Pallas/Mosaic GPU pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)
distinguishes parallel and sequential grid axes, supports warp-specialized
pipeline helpers, and exposes concurrency/release and resource-scope choices.

**Borrow:** producer completion is not permission to recycle a buffer; consumer
state must be allocated only for its actual participants. Our compiler should
infer these obligations from lexical values and accesses. This does not require
copying hardware-specific public roles into our DSL or adding intra-kernel
Tile/SIMT mixing. More automation is a benefit only if it preserves control
over the difficult cases these interfaces expose.

## Asynchrony, pipelines and optimization

### Tawa: compiler-internal channels with operational semantics

Tawa partitions tile programs into producer/consumer roles and uses asynchronous
references for communication. The aref protocol separates publishing, acquiring
and releasing a value; rings support multiple live iterations. Partitioning and
multi-granularity pipelining are compiler passes.
[CGO 2026 paper, Sections III-B–E](https://www.csl.cornell.edu/~zhiruz/pdfs/tawa-cgo2026.pdf).

**Borrow:** model safe reuse, not just readiness. A consumer's asynchronous
instruction can continue using storage after its issue point, so last textual
use is not necessarily the end of the physical lifetime.

**Difference to test:** treat such a channel as one supported implementation
of our resource/version protocol, not a mandatory public entity or universal
communication mechanism. Multiple consumers require their actual completion
join. Scope remapping must preserve the protocol's participants and progress
assumptions. We cannot call our sketches more rigorous than an operational
semantics already published for a narrower problem.

### Twill: the nearest joint cost/solver comparison

Twill jointly formulates modulo scheduling, warp assignment, memory liveness,
communication and blocking synchronization. Its evaluated fragment has a single
loop without inner control flow; tile sizes remain external. It searches
initiation intervals using constraint solvers and normalized operation costs.
Evaluation hand-translates derived schedules into CUDA because downstream
lowering remained problematic.
[December 2025 v1, Sections 3–6.1](https://arxiv.org/html/2512.18134v1).

**Borrow:** jointly price the ability to *realize* overlap; a blocked issuing
warp is not a zero-cost edge. Keep model optimality separate from actual timing.

**Difference to test:** compose plans across nested/sibling regions, including
alternative atoms/layouts, and realize them through our emitters without
manual repair. These are substantial unfinished obligations. Simply adding
more variables to the optimization problem is not a result. A practical
online planner also needs a compile budget; an exact offline solution can
serve as an oracle for bounded test cases rather than every JIT invocation.

### Lift and constraint-based parallelism mapping

Lift uses composable data-parallel patterns and rewrite-based lowering.
[CGO 2017 paper](https://lift-project.github.io/publications/2017/steuwer17LiftIR.pdf).
The CC 2022 work extracts hardware mapping and memory-scope constraints from
functional IR. Synchronizability and actual memory use still require late
checks in that implementation.
[Sections 4–5](https://www.pure.ed.ac.uk/ws/portalfiles/portal/274242906/Mapping_Parallisms_MOGERS_DOA25022022_AFV.pdf).

**Borrow:** propagate structural restrictions before compiling candidates.
**Lesson for us:** satisfying an incomplete constraint encoding is not a
certificate of executable code. Explicit participant convergence and
emitter-supported protocols should reduce late rejection; retain final
validation anyway. Report candidate rejection categories and compile cost,
not only the fastest valid sample. Our effectful SSA needs additional memory
and ordering contracts beyond a pure array-pattern fragment.

### Timeloop, CoSA and Halide: cost/search are separable tools

[Timeloop, ISPASS 2019](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)
couples accelerator models, mappings and performance/energy evaluation.
[CoSA, ISCA 2021](https://arxiv.org/abs/2105.01898) uses mixed-integer
optimization for spatial/temporal accelerator schedules.
[Halide autoscheduling, SIGGRAPH 2019](https://halide-lang.org/papers/autoscheduler2019.html)
combines learned cost prediction with tree search.

**Borrow:** separate architectural facts, candidate generation, cost estimation
and the search algorithm. A backend cost policy should supply service and
resource information rather than replace common legality with heuristics.

**Our proposed improvement criterion:** a policy calibrated on some operators
must rank unseen shapes/operators on the same target; another target should
reuse search without inheriting GPU-specific occupancy equations. Compare
regret and compilation budget. MILP, annealing, beam search and repeated JIT
measurement can coexist. None makes unknown hardware costs exact, or makes
an encoded optimum a universal performance optimum.

## Compositional transformations and formal assurance

### Schedule trees and Tiramisu: more than a flat affine schedule

[Schedule Trees, IMPACT 2014](https://acohen.gitlabpages.inria.fr/impact/impact2014/papers/impact2014-verdoolaege.pdf)
represents polyhedral scheduling with structured bands and ordered/unordered
composition. [Tiramisu, Sections 3–5](https://arxiv.org/html/1804.10694v3)
separates algorithm, execution management, storage and communication, targeting
CPU, GPU and distributed systems.

**Borrow:** use a region forest, dependence relations and explicit storage
decisions; preserve useful affine fragments and guards. A sequence of nests
is not an exotic new language feature.

**Different starting point:** our source declares logical concurrency and
aggregation contracts before physical scheduling. This can spare the compiler
from recovering promised independence. It also places a stronger obligation
on the user: a recurrence written as parallel is invalid, not something
the planner repairs. We must justify this trade-off through understandable
diagnostics and useful transforms, not call it automatically superior to
algorithm-first scheduling.

### Fireiron and Exo 2: small building blocks, extensible scheduling

Fireiron refines computation specifications through reusable decompositions
until they match an instruction or supplied microkernel.
[2020 preprint, Sections 2–3](https://arxiv.org/pdf/2003.06324).
**Borrow:** new hardware should extend supported realizations. Avoid demanding
that every operation be reduced to scalar code before instruction selection.
Our effectful region interface must specify a custom atom's effects and
convergence as well as its arithmetic.

Exo 2 composes trusted fine-grained actions, inspection and cursors into
scheduling libraries; the underlying program remains transformable.
[ASPLOS 2025 paper, Sections 3–5 and Appendix A](https://arxiv.org/pdf/2411.07211).
**Borrow:** build compound optimizations from a small checked rewrite basis,
with mutation-safe references and explicit invalidation of analyses.
**Difference to test:** compose asynchronous participant/resource refinements
along with ordinary scheduling. We do not need to expose a second scheduling
language or reproduce Exo's entire implementation to use this architectural
lesson. Conversely, an intrusive-list IR alone does not guarantee safe
references, effect correctness or compositional rewrites.

### ATL and concurrent algebra: rigor needs an actual theorem

ATL's POPL 2022 work defines a pure tensor language in Coq and proves
source-to-source scheduling rewrites. It identifies stateful sliding windows
and direct parallelization controls as outside that implementation.
[Sections 1, 3–6](https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf).

**Borrow:** derive rewrite laws from semantics rather than declare them axioms.
**Our status:** finite examples and proof sketches are substantially weaker
assurance. Effectful memory, asynchronous completion and numerical policy make
our intended domain different, not our current proof stronger.

[Concurrent Kleene Algebra, ESOP 2018](https://arxiv.org/abs/1710.02787)
provides a completeness result for bounded-parallelism event semantics.
**Borrow:** separate ordering refinement from arithmetic and effect
assumptions. **Boundary:** an abstract concurrent-algebra theorem is not a
proof of GPU barriers, publication, aliasing or progress. Our product order
must be instantiated with these contracts before it justifies fusion.

### Mirage: hierarchy plus search and verification already exists

Mirage's muGraphs span kernel, block and thread levels. Search combines
algebraic and scheduling changes with abstraction-based pruning. Its finite-field
probabilistic equivalence result applies to a stated expression class;
floating-point stability is checked separately. Layout/scheduling/memory
optimization follows candidate verification.
[OSDI 2025 paper, Sections 3–6](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf).

**Borrow:** keep search separate from equivalence checking and give pruning
guarantees precise premises. **Difference to test:** our proposed checker
accepts composed mapping/protocol witnesses for an effectful source instead
of validating only a functional tensor expression. That needs a real
implementation. It is not evidence that Mirage is unverified, nor that our
current allclose tests offer comparable guarantees. Model-optimal search,
probabilistic algebraic equivalence and bitwise IEEE equivalence must not be
reported as the same property.

## Dynamic tasks and distribution

### Event Tensor: a direct precedent for fine-grained sibling execution

Event Tensor represents task-set completion with symbolic-shaped event
collections and task/event coordinate relations. ETC lowers them to static
or dynamic persistent schedules, including data-dependent updates/triggers
and communication/computation overlap.
[MLSys 2026, Sections 2–3](https://arxiv.org/html/2604.13327v1).

**Borrow:** distinguish producer-instance readiness from whole-kernel
completion; preserve those relations through fusion. **Important difference:**
our current Tile Runtime lacks general multi-artifact or dynamic persistent
scheduling. Repeated shape-specialized JIT cannot determine routing values
computed inside the kernel. Dynamic task creation therefore needs an explicit
future execution/progress contract, not an assertion that static nests are
complete for all workloads.

Our near-term model should summarize static region dependencies compactly and
leave this extension possible. Do not add event counters to every public
kernel just to imitate a different runtime boundary. Conversely, we cannot
claim first-class event relations or fine-grained multi-device overlap as new.

### Sequoia, Legion and DISTAL: distinguish static compilation from runtime

[Sequoia, SC 2006](https://graphics.stanford.edu/papers/sequoia/) centers
hierarchical decomposition and locality.
[Legion, SC 2012](https://elliottslaughter.com/pdfs/sc2012.pdf) expresses
logical regions, privileges and partitions while separating placement.
[DISTAL, PLDI 2022](https://compilers.stanford.edu/publications/pldi22-distal/)
separates tensor distribution from computation scheduling for distributed
CPU/GPU programs.

**Borrow:** explicit data-access capabilities and region effects; topology
cannot be reduced to a numeric memory hierarchy. Our lexical inference can
avoid routine privilege annotations, but opaque effects still need contracts.

**Different scope:** the current plan is primarily static and single-device.
Extending participant coordinates to devices is only notation until allocation,
transfer, collective matching, visibility and progress have Runtime support.
A smaller static compiler can be useful; it is not thereby more general than
these systems.

## More rigorous, not unnecessarily rigid

These are independent axes. A model can be precise but restrictive, or
expressive but difficult to analyze. Our intended trade-off is **strict
observable semantics, open implementation choices**.

```{table} What to constrain and what to leave open
:class: design-table

| Decision | Necessary discipline | Avoidable rigidity |
|---|---|---|
| Parallel domain | Trust declared noninterference; preserve exact work coverage | Re-prove independence before every lowering, or forbid legal same-instance aliases |
| Nest boundary | Preserve observable owner/context and participant identity | Treat every lexical level as a fixed hardware level or mandatory barrier |
| Memory | Preserve reaching versions, access capability and safe reuse | One storage level per execution level; mandatory manual temporaries |
| Reduction | Distinguish ordered folds, commutative reductions and numerical permissions | Require every reduction to be commutative, or accept every recurrence as a reduction |
| Pipeline | Preserve readiness, completion, release and progress | Fixed producer/consumer role names, or one stage label per physical buffer |
| Formal fragment | State decidable/bounded subsets and fallback contracts | Pretend arbitrary dynamic indexing is affine, or ban it from the entire language |
| Search | Enforce legality; optimize a stated objective with a budget | Require an exact global solve before any program can compile |
```

An unobserved structural cut should be eliminable; an observed cut should be
preserved or explicitly re-homed. Likewise, an unsupported proof is
**unknown**, not a proof of invalidity. It can limit a transformation while
leaving the source program and a conservative lowering valid.

We should not describe the whole model as stronger than existing work.
Currently, ATL has mechanized rewrites where we have sketches; Twill has an
implemented joint solver where we have a general formulation; Cypress/Tawa
have asynchronous lowering beyond our current broad support; Event Tensor
has dynamic execution absent from our Runtime. The productive question is
whether one compact interface can connect these ideas with less manual work
and without losing performance.

## Design consequences and concrete comparison cases

The following are proposed compiler-internal requirements, **not new public
DSL entities or completed passes**:

1. **One region boundary interface.** Summarize coordinate domains and observed
   cuts, read/write effects and value versions, required participant
   convergence, completion/reuse relations, and numerical permissions. Infer
   routine information; require explicit contracts only at opaque boundaries.
2. **Typed mapping and completion relations.** Keep logical remapping, target
   binding, value distribution and per-resource addressing separate. A copy
   elimination or scope fusion transforms their relations together.
3. **Composable candidate plans.** A child's interface includes output layouts,
   live resources and readiness, not only shape and a scalar cost. Prune only
   candidates with compatible boundary conditions; otherwise a locally cheaper
   choice may make its parent slower or impossible.
4. **A small checker and reusable realization libraries.** Search strategies
   propose witnesses; the checker validates the admitted fragment. Atom and
   protocol emitters expose capabilities and costs. This can live in thin
   TileIR analyses and plan records without introducing MLIR or a second
   public language.
5. **An honest escape boundary.** Keep precise supported fragments and expert
   constraints. Opaque/data-dependent parts retain conservative effects;
   dynamic scheduling or unsupported protocols require deliberate extensions.

For example, one resource version with two consumers has this dependency
shape (a proposed internal model, not user-written synchronization):

```text
producer completes(v) --+--> consumer A completes(v) --+
                       |                             +--> reusable(v)
                       +--> consumer B completes(v) --+        |
                                                  producer may overwrite slot
```

Neither issuing A nor finishing only A makes the slot reusable while B still
reads it. Unrelated consumers of other resource versions need not join here.
The schedule determines when these events happen; storage assignment determines
which later version would overwrite this slot. Optimizing either in isolation
can change the other's legal choices.

To test whether this is an improvement, implement the same cases in the
closest systems. The cases below are a comparison plan, not completed ports:

```{table} Minimal distinguishing programs
:class: design-table

| Case | Required result | Most relevant comparisons |
|---|---|---|
| One nest, A/B/acc with unrelated layouts | Participants are shared without forcing a common memory map | Cypress, Graphene, Hexcute, TIRx |
| Split/merge a pure nest, then repeat with an observed owner | First is freely remappable; second preserves or re-homes identity | Hidet, LEGO, TensorIR |
| Pointwise sibling scopes versus neighbor reads | Pointwise zip is legal; neighbor zip needs remapping or communication | Schedule trees, Exo 2, Cypress |
| Ordered noncommutative fold versus commutative sum and scan | Retain contribution order where required and all observable prefixes | Stripe, Lift, ATL |
| Async producer with two consumers and a reused slot | Join the necessary completions before reuse, not every unrelated task | Cypress, Tawa, Event Tensor |
| Fused MMA/reduction/epilogue under shared-memory limits | Compare materialization, participant, atom and pipeline alternatives | Hexcute, Twill, CoSA |
| Ragged gather or runtime MoE routing | Separate static guards from data-dependent dependencies; retain a valid boundary | LEGO, TileLang, Event Tensor |
| Same logical region on SIMD and Metal | Different plans and costs, same source contracts and checker interface | Exo 2, Tiramisu, TIRx |
```

Measure output/numerical policy, inserted transfers and synchronization,
manual annotations, search coverage, late rejection, compilation budget and
held-out timing. Include ablations that remove boundary information or fix
placement early. A useful claim needs a concrete win that disappears in the
ablation, not just a longer list of supported concepts.

## A falsifiable contribution, not a novelty slogan

The working hypothesis is a **compositional refinement calculus for an
execution-first, effectful Tile SSA program, with jointly checked participant,
resource and temporal mappings**. The word *jointly* must mean more than
putting existing passes behind one API.

```{table} Claims to test before publication
:class: design-table

| Research question | Required evidence |
|---|---|
| Can remaps preserve observable ancestor identity? | Prefix/fiber laws; counterexamples where flattened bijections break sharing; comparison with Hidet, LEGO and Cypress |
| Can sibling scopes fuse without redefining primitives? | Effect/dependence interface, ordered and unordered fusion rules, reduction and pipeline counterexamples |
| Can layouts and resource/time plans compose? | A typed boundary witness tracking distributions, live values, completion and version ownership; compare Hexcute, Twill and Tawa |
| Is mapping complete in a useful sense? | A declared bounded vocabulary and constructive representation/search result; exclude unsupported dynamic indices and protocols |
| Can optimization generalize? | Held-out operator graphs and shapes, multiple targets, mapping/materialization/atom ablations and measured model regret |
| Does the theory check the implementation? | A small semantics, mechanized core or independent witness checker, and explicit trust boundaries for bridge/codegen/runtime |
```

The current compiler implements **bounded pieces**, not that research result.
The new automatic Metal cooperative mapping is a general candidate-admission
and resource-capacity repair; it is not a complete calculus, calibrated joint
solver or proof of superiority to these systems. Its measurements belong in
the [performance report](../../performance/tile/index.md).

The comparison cases above must establish which relationships are explicit,
which the compiler actually checks, and which plans it can emit. A credible
contribution would connect these boundaries with useful compositional
guarantees and measured benefits. A different vocabulary, more annotations,
or a superset of boxes in an architecture diagram is insufficient.
