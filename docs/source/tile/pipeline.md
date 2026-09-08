(pipeline-is-a-temporal-producer-consumer-nest)=
# Pipelines and stage boundaries

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

The [implemented software-prefetch path](../internals/tile/lowering.md#implemented-native-software-prefetch-path) is narrower than the scheduling model below; hardware async transfer and warp specialization remain future work.

```{contents} On this page
:local:
:depth: 2
```

Pipeline belongs to Execution because it organizes when one logical spatial
sub-hierarchy runs. It is not another memory hierarchy. Its natural parent is
therefore visible in the same syntax as any other nest:

~~~cpp
for (auto &nest : parallel(grid_shape)) {
    for (auto &subnest : nest.parallel(subnest_shape)) {
        for (auto &k : subnest.pipeline(iteration_space, policy)) {
            k.stage("produce");
            // Producer operations for k.index().

            k.stage("consume");
            // Consumer operations for k.index().
        }
    }
}
~~~

There is one logical pipeline instance per `subnest` coordinate. Moving the
pipeline outside that `parallel` changes the semantics to one pipeline shared
by the parent nest; it is not merely formatting. The pipeline loop adds a
temporal coordinate but does not deepen the spatial execution frontier.

Like `parallel` and `serial`, the C++ range executes its body exactly once
during capture. Dereference creates `PipelineOp`, pushes its body, and returns a
non-copyable iteration handle; `k.index()` is its staged coordinate. Increment
closes the region after discovering carried values. A multidimensional domain
returns a tuple-like `index()`. Generated code executes all logical iterations
through a prologue, steady state, and epilogue chosen by scheduling.

## Stage boundaries are lexical

A pipeline is a repeated producer/consumer graph, not a loop with only an `II`
annotation:

~~~text
Pipeline = (IterationSet I,
            StageSet S,
            StageOrder <S,
            Dependences D,
            Policy)

Dependence edge = (producer_stage, consumer_stage,
                   iteration_distance, value_or_effect)
~~~

`k.stage()` is a frontend cursor on the pipeline iteration handle, not an
executable marker operation. The first call begins stage zero and each
subsequent call ends the current source segment and begins the next.
`k.stage("load")` adds an optional compile-time name to the new segment. Capture
turns the segments into ordered child regions of the single pipeline operation;
internally each has identity `(PipelineId, ordinal)`. The name is a stable
diagnostic and scheduling label, not global identity. Consequently unrelated
pipelines cannot accidentally interleave their stage namespaces. A later fusion
pass may combine pipelines only by constructing a new graph and proving
dependence and resource equivalence.

Stage ordinals are not iteration coordinates, cycle numbers, memory versions,
execution levels, or hardware warp IDs. Source order supplies the default
same-iteration producer-before-consumer order; SSA, MemorySSA, and effect
analysis record the precise edges, including edges that skip phases and
loop-carried edges with positive iteration distance. Ordering a pair of stages
does not itself invent a whole-hierarchy barrier.

The compiler may infer all stage membership. `k.stage(optional_name)` is only
the explicit surface for pinning a cut when the programmer wants that
producer/consumer structure to be part of Candidate TileIR:

~~~cpp
for (auto &k : subnest.pipeline(k_tiles, policy)) {
    auto k0 = k.index();

    k.stage("load");
    auto a = A.tile(a_origin(k0), a_shape, bounds::zero).load();
    auto b = B.tile(b_origin(k0), b_shape, bounds::zero).load();

    k.stage("compute");
    acc = mma(a, b, acc);
}
~~~

`a` and `b` are virtual Tile SSA, not source-declared buffers. A GPU schedule
may map the stage-active operations to disjoint participant subsets and replace
the matched library expansion with copy/MMA atoms; a CPU schedule may use tasks
or serialize it. If an engine handoff requires addressable storage, the planner
inserts internal materialization, MemorySSA, versions, events, and barriers. No
target-specific role or mandatory staging buffer leaks into the kernel.

When `stage()` is absent, all stage membership is open for inference. Once a
pipeline contains an explicit cut, every effectful or tile operation in its
immediate body must follow the first cut. Cursor calls themselves must occur
unconditionally at that body level, so Candidate TileIR has a static stage
graph; stage contents may contain ordinary structured control flow and child
execution regions. Pure iteration-index expressions may precede the first cut
and be used by several stages.

Because a cursor call does not open a C++ block, `a` and `b` remain naturally
visible after the cut. Their cross-stage SSA edges are explicit in TileIR. This
is why a cursor cut is preferable to a stage-specific C++ brace scope.

```{figure} ../../_static/tile/pipeline-stage-flow.svg
:alt: C++ stage cursor cuts become pipeline subregions and a dependence graph whose participant and engine bindings are chosen later.
:width: 100%

The source fixes producer/consumer cuts while leaving communication,
participant subsets, engines, and physical realization open.
```

The pipeline iteration coordinate and stage identity remain orthogonal:
`k.index() == 0` means the first logical iteration. It is useful for a true
prologue special case, but it does not select an execution role.

## Scheduling and versioning

For a dependence `e = (sp, sc, distance, payload)`, a legal schedule satisfies:

~~~text
II >= 1
Issue(stage s, iteration i) = i * II + theta(s)
Issue(op in s, i) = Issue(s, i) + delta(op)

Issue(sc, i + distance) >= Issue(sp, i) + latency(e)

Schedule(op, i)
  = (Issue(op, i), anchor(op), frontier(op), Active(op), engine(op))

Version(materialized edge or MemoryState, i) -> VersionCoord
~~~

`theta(s)` places a logical stage in the modulo schedule and `delta(op)` orders
operations within it. `max_in_flight` bounds the scheduling window; it is not
the number of logical stage segments and is not blindly copied to every
buffer depth. `initiation_interval` belongs in `pipeline_policy` because it is
a primary temporal constraint.

The compiler derives:

- prologue, steady state, and epilogue;
- async-copy and compute issue points;
- per-edge live version count;
- storage ring indices for materialized edges;
- barriers, events, waits, and fence scopes;
- resource pressure and legality.

Only a materialized edge's `VersionCoord` enters its `AddressMap`. Pure SSA
edges do not acquire fictitious memory versions.

```{figure} ../../_static/tile/pipeline-timeline.svg
:alt: Three pipeline iterations overlap across load, compute, and store engines while their memory versions remain live.
:width: 100%

The scheduler overlaps stage instances subject to dependence latency; storage
version count follows liveness rather than source stage count.
```
