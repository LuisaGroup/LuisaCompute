# Shared suspending callables (experimental)

A regular DSL `Callable` may contain `$suspend`. Suspension is inferred through
its static call graph; it suspends the enclosing `Coroutine`. A synchronous
kernel cannot call such a function. No additional callable type is required.

A suspending definition is lowered once per coroutine compilation, including
when it is marked `noinline`. Each static call site stores arguments and a
return selector, then branches to the shared function region. Returns write
the callee result and branch to the selected caller continuation. These
transfers do not yield to the scheduler. Only `$suspend` yields; root return
completes the task.

Because the DSL disallows recursion and calls are synchronous, a coroutine
instance needs at most one active invocation of each function. Parameter,
return-selector, result, and local storage can therefore live in a statically
allocated root frame. No runtime allocation or dynamic stack is used.

`CoroGraph::call_graph()` preserves the static function/call-site relations.
Function zero denotes the root; each edge pairs a call with its return-site
selector. Function resume tokens identify shared suspension continuations.
This is source-level, conservative metadata: subsequent dead-code elimination
may remove an unreachable call or suspension. The existing graph nodes and
queue edges remain the scheduler view, and do not turn calls into suspension
points. Function and return-site IDs are compilation-local.

The number of distinct suspend tokens is determined by the source suspensions,
not by call sites, call depth, or ordinary callable inlining. A loop or several
calls may execute the same suspension many times while retaining the same token.
Call arguments, return selectors, and internal control-flow selectors do not
create suspend tokens. The inliner treats suspension instructions as barriers;
transitively suspending calls are lowered into shared regions before ordinary
source optimization, including when the user has not requested `noinline`.

Shared regions lower to a single raw CFG before distillation and frame
coloring. Return selectors retain their activation identity through optimization
and are cleared on return. Liveness first traverses an analysis-only product of
the semantic CFG and active return selectors, including suspend/resume edges.
Only the selected return edge is traversed. Its scope and transition relations
are then projected onto the shared graph; no analysis state becomes a generated
callable or scheduling node. Frame coloring conservatively joins the legitimate
contexts sharing a continuation. `call_graph().analysis_state_count` reports the
number of reachable analysis states. It measures compile-time analysis work,
not frame size or generated scheduling nodes.

After frame layout and continuation splitting, each shared-call continuation
uses a local block dispatcher to express its raw CFG as a structured loop and
switch. Each raw block becomes one arm; its terminator selects the next arm.
Conditional and indexed branches retain their conditions and target payloads.
This makes control-flow structure linear in the number of raw blocks without
cloning their instruction bodies. Different suspend scopes may still need the
same synchronous instructions during continuation splitting.

The dispatcher's block selector is an ordinary local variable introduced after
frame layout. It is neither a frame field nor a suspend token. Dispatch remains
inside the current continuation until an original suspension or return exits
it; changing a block selector does not return control to a scheduler.

Reference and resource parameters use a finite alias domain. Each call captures
an alias selector and any dynamic access-chain indices; reads, writes, and
ordinary calls dispatch over that domain. Two references to the same storage
therefore remain aliases, including through nested calls and across suspension.
Different call sites may pass different resources. Resource handles and local
pointers are never spilled into the coroutine frame.

Alias families canonicalize paths by their original storage, access-chain shape,
constant indices, and dynamic index types. Equivalent incoming paths reuse the
function's index-capture slots, which are refreshed on every call. Forwarding
one reference through several call sites at each level therefore does not
multiply alias alternatives or duplicate captured indices at the next level.

Scheduler-visible writable reference bindings use guarded `CoroSlotAccess`
projections of the original storage. Each finite alias/access-chain alternative
has a Boolean guard captured at suspension and a static frame projection. A
handler reads and writes the selected original slots directly, so repeated or
overlapping bindings observe each other's updates. Unselected carrier values
are preserved. The normalized analysis describes this as a read/modify/write
of the candidate set, including for logically write-only bindings. Compiler
carrier bindings stay outside the plugin's logical schema. Ordered extension
stages, partial frame reconstruction/writeback, packed Boolean fields, and
AoS/SoA compaction use the same existing slot plans. XIR cloning, verification,
analysis certificates, and text/bitcode interchange retain the projections.

Compiler boundaries are checked explicitly:

- Suspending synchronous callbacks and recursion are rejected.
- The lowering accepts AST-derived, PHI-free definitions. It is an internal
  coroutine compilation stage, not a general-purpose inliner.

The on-demand texture example calls one outlined sampler at two coordinates.
It checks that both sites share one resume node and validates the weighted
sample result against the host texture. `test_coro_shared_callable` additionally
covers nested calls, conditional suspension, early returns, live caller state,
state-machine and persistent scheduling, compacting wavefront AoS/SoA, and
scheduler writes through nested, dynamically indexed aliases. A CFG regression
checks that state defined between two calls is live on the second suspension
edge only, and contrasts this with the context-free CFG result.

`test_coro_callable_rendering` compares three small deterministic images against
ordinary, non-suspending callable shaders with the same arithmetic:

| Scene | Shared call depth | Static call sites | State carried across suspension |
| --- | ---: | ---: | --- |
| Sphere reflections and textured environment | 2 | 4 | Ray, throughput, radiance, filtered sample |
| Front-to-back volume integration | 4 | 7 | Variable step count, opacity, nested return state |
| Layered materials | 8 | 17 | Early returns, shared leaf calls, dynamically indexed local aggregate |

Each scene runs through state-machine, compacting wavefront AoS and SoA, and
persistent scheduling. The deepest chain leaves callable inlining enabled.
Every case asserts one distinct source resume token and two scheduling nodes,
including the entry node. Inputs cover both conditional suspension and calls
that finish without suspending. Output is filled with NaNs before each run;
comparison checks finite RGBA values, nonconstant reference output, and maximum
absolute error below `2e-4`. Shader caching is disabled for these comparisons.

Frame-size limits are 192, 256, and 448 bytes for the three fixed scenes. The
Metal measurements are 144, 208, and 388 bytes. Persistent
scheduling uses 64-thread blocks so the largest frame and scheduler bookkeeping
fit within 32 KiB of threadgroup memory. A separate scaling case keeps a
128-byte callee array live across suspension and compares 1, 4, and 16 call
sites against a host-computed result. It requires one shared resume token and
allows at most 16 bytes of frame growth over the single-site case; measured
frames are 160, 164, and 164 bytes. Increasing static call sites therefore
cannot silently replicate the callee's live array in the frame.

A separate reference-forwarding regression uses depths 1, 4, and 8, with two
calls to the next level at every layer. Four root sites change the dynamic
array index, switch to a different array, and select a constant element. All
four scheduler configurations compare against a host execution oracle. It
checks one source token, two scheduling nodes, and at most 32 bytes of extra
frame storage per added call level. The measured frames are 100, 172, and
268 bytes, respectively: 24 bytes for each additional shared call level.

The rendering executable requires an explicit backend, for example
`build-coro/bin/test_coro_callable_rendering metal4`, after a full build of the
selected build tree. Setting `LUISA_CORO_RENDER_OUTPUT` to an output directory
optionally writes the actual images as linear RGB PNGs for inspection. The
numeric comparison always runs and does not depend on these diagnostic images.

The shared-region boundary demotes cross-block rvalues after source optimization.
A resumed scope may re-enter the same callee and recompute those definitions;
replacing all uses with one entry frame reload would incorrectly reuse the
previous invocation's value. Explicit definition stores preserve this distinction.
Cross-block GEP chains are reconstructed at dispatcher consumers before scalar
transport, so dynamic indices keep their captured values and each address
dominates its use. This happens after frame layout and adds no frame fields.

Generic CFG restructuring is validated independently of the continuation
dispatcher. Its regression suite interprets both the original and transformed
graphs, comparing ordered observable stores as well as returned values in both
transactional and in-place modes. Deterministic generated graphs include
conditional and indexed branches, bounded loops, and varied block allocation
order. Dedicated cases cover nested loops with early exits, conditional latch
payloads, and cloned regions with escaping scalar values, local storage, and
dynamically indexed addresses.

Re-entry splitting includes the entry's complete executable strongly connected
component within the current construct activation. A dominance-only boundary
can cut such a cycle into sibling regions; repeatedly copying either half then
creates another entry into the other half and never converges. Copying the
complete cycle preserves its backedges and stops this artificial unfolding.
Main selection recovery and the final audit use the same source-relative
enclosing-boundary predicate. A conditional that already exits its parent
construct does not acquire an unnecessary selection and exit selector on a
subsequent pass.

Construct contraction preserves each child's complete physical exit cut,
including legal non-local exits, rather than substituting only the declared
merge. Parent exit repair retargets those literal edges together. Conditional
latches retain their exit work and gain a separate unconditional update edge.
Loop-internal reachability applies the header's dominance boundary to both the
search seed and later edges. Exiting to an enclosing loop and returning in a
new activation therefore does not become a false internal exit.
Cloned regions share escaping ordinary storage and transport scalar definitions
on both paths; copied headers with unowned structural boundaries are recovered
from their executable CFG. These repairs keep the strict repeated-obligation
and structural verification checks enabled.

Affine ray-query state needs a different clone boundary: copying an initializer
also copies the executable paths to every consumer of that query. The closure
includes any further query lifetimes initialized on those paths. If this crosses
the parent's old merge, its structural annotation is recovered from the updated
CFG. The lifetime traversal continues through already-owned consumers to include
external paths that return to them. Arm-owned terminal payloads remain inside the
arm. This preserves a single
initializer dominating each query object's consumers, including consumers after
the original selection merge and across multiple branches.

The rendering cases also exposed an Apple GPU compilation error in dynamic
threadgroup atomics following divergent loop exits. A separate ordinary-kernel
regression records every lane's selected bucket and compares the histogram with
a host oracle in 64- and 128-thread blocks. Both the MSL and direct AIR paths
materialize the atomic address through a thread-private volatile pointer before
issuing the atomic. This preserves per-lane addresses; it does not mark shared
frames volatile, add a barrier, change scheduling, or add coroutine frame fields.

Packed Boolean stores seed a fresh physical word with zero. They read the old
word only when the transition's live mask contains bits outside its store mask.
This avoids undefined first-write RMW while retaining dormant neighboring bits;
the existing three-scheduler packed-Boolean regression runs on Metal and Metal4.
