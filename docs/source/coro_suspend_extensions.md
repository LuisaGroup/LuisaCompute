# Coroutine suspend extensions

## Status

This document defines the host and IR contract for extensible behavior at a
coroutine suspension boundary. It is a design contract; implementation should
land in independently testable slices and must preserve the existing
`$suspend(name)` and `coro_frame_export(name, value)` spellings during
migration.

## Motivation

A coroutine boundary is useful to more than the coroutine state machine. A
scheduler may want to sort waiting work, group it into dispatches, or select a
specialized kernel. A graph transform may want to insert an external stage,
such as a neural-network evaluation. A debugger or profiler may want to
observe selected values.

These uses must not be encoded as magic frame-field names or scheduler-specific
configuration lists. The boundary declaration must survive cloning, hashing,
AST/XIR interchange, coroutine splitting, and graph materialization, while the
consumer remains optional and scheduler-specific where appropriate.

## Source API

The source-side object is move-only and polymorphic:

```cpp
enum class CoroSuspendFallback : uint8_t {
    ignore,
    warn,
    reject,
};

class CoroSuspendExtension {
public:
    virtual ~CoroSuspendExtension() noexcept = default;

    [[nodiscard]] virtual luisa::string_view schema() const noexcept = 0;
    [[nodiscard]] virtual uint32_t version() const noexcept = 0;
    [[nodiscard]] virtual bool is_annotation() const noexcept {
        return false;
    }
    [[nodiscard]] virtual CoroSuspendFallback fallback() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const CoroSuspendBinding>
    bindings() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const CoroSuspendAttribute>
    attributes() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<CoroSuspendExtension>
    clone() const noexcept = 0;
};

class CoroSuspendAnnotation : public CoroSuspendExtension {
public:
    [[nodiscard]] bool is_annotation() const noexcept final { return true; }
};

using CoroSuspendExtensionPtr =
    luisa::unique_ptr<CoroSuspendExtension>;
```

Factories return `CoroSuspendExtensionPtr`, so a call site can remain uniform:

```cpp
$suspend("shade_surface",
         coro_sort_by(material_id, material_count),
         coro_dispatch_by(ray_kind),
         coro_stage("com.example.nn-shade", nn_inputs, nn_outputs),
         coro_breakpoint("after-surface"));
```

The plugin-created virtual object exists only while the DSL records the suspend
statement. Recording freezes it into a normalized, data-backed implementation
of the same interface. AST `SuspendStmt`, XIR `CoroSuspendInst`, and compiled
graph boundary records each own a `unique_ptr<CoroSuspendExtension>` containing
that normalized representation. They may query the virtual interface directly,
but they never retain a plugin-owned behavior object or depend on C++ RTTI
across a shared-library boundary.

Deserialization reconstructs a generic data-backed annotation or semantic
extension from schema data. Consumers dispatch by schema and version, so the
original factory's C++ derived type is not required to clone, cache, or load an
XIR module.

Existing frame exports remain accepted and may be mixed with extensions during
migration:

```cpp
$suspend("shade_surface",
         coro_frame_export("legacy_value", value),
         coro_sort_by(material_id, material_count));
```

## Annotation and fallback are independent

`is_annotation()` states whether dropping the extension preserves the
observable result of the coroutine. It does not state whether dropping it is
allowed for the current invocation.

Examples:

| Extension | Annotation | Typical fallback | Meaning |
| --- | --- | --- | --- |
| sort key | yes | ignore | Unsupported scheduling loses performance only. |
| profiling marker | yes | ignore or warn | Collection may be optional. |
| requested debugger breakpoint | yes | reject | Observation is the requested operation. |
| neural shade stage | no | reject | Dropping it changes rendered values. |

An unhandled extension follows its declared fallback. A consumer must not infer
fallback from `is_annotation()`.

## Normalized descriptor

The normalized implementation contains schema-based data:

```cpp
struct CoroSuspendExtensionData {
    luisa::string schema;
    uint32_t version;
    bool annotation;
    CoroSuspendFallback fallback;
    luisa::vector<CoroSuspendBinding> bindings;
    luisa::vector<CoroSuspendAttribute> attributes;
};
```

Requirements:

- `schema` is a stable, globally namespaced identifier. Built-ins use the
  `luisa.coro.*` namespace; third parties use a reverse-DNS namespace.
- `version` versions the descriptor schema, not a handler implementation.
- binding and attribute names are unique within an extension.
- attributes are deterministic typed scalars or strings. Opaque native byte
  blobs, raw pointers, RTTI names, and process-local callback addresses are not
  permitted.
- attributes are stored in canonical name order before hashing or interchange.
- duplicate schemas at one boundary are allowed only when that schema declares
  a deterministic merge rule. Otherwise validation rejects them.

`CoroSuspendAttribute` supports `bool`, signed and unsigned 64-bit integers,
`double`, and UTF-8 strings. A future aggregate value must be added as an
explicit interchange type instead of smuggling bytes through a string.

`CoroSuspendBinding` values contain owner-local indices, not frontend pointers.
The AST owner maps a binding index to an `Expression` or writable access path;
AST-to-XIR remapping makes the same binding index address an XIR value or
lvalue identity; graph materialization maps it to a physical frame field and
optional access path. This lets AST and XIR passes query one virtual extension
interface without putting `Expression *` and `xir::Value *` in a shared ABI.

## Operand lifetime

Every typed operand declares both how it accesses the coroutine data flow and
when a consumer needs it:

```cpp
enum class CoroSuspendBindingAccess : uint8_t {
    read,
    write,
    read_write,
};

enum class CoroSuspendBindingLifetime : uint8_t {
    boundary, // consumed while lowering or immediately at the suspend edge
    queued,   // must remain readable while the continuation waits
    resumed,  // consumed by an inserted stage immediately before resume
};
```

`read` accepts an rvalue expression and snapshots its value at the boundary.
`write` and `read_write` require a writable AST lvalue. A writable operand may
name a local variable or a statically representable aggregate access path; a
temporary expression is rejected. Recording marks the corresponding AST usage
as write or read-write instead of manufacturing a read of an uninitialized
value.

`queued` and `resumed` operands become semantic coroutine frame exports during
normalization. The compiler generates a reserved alias from the suspend token,
extension ordinal, and operand name. Users and schedulers do not communicate
through that alias; the graph resolves it to a physical frame-field index and
stores the index in the materialized descriptor.

This keeps liveness, frame coloring, relocation, and SoA/AoS storage under the
existing coroutine ABI analysis. A scheduler must never recover an extension
operand by looking up a hard-coded source name such as `coro_hint`.

### Partial-frame reconstruction protocol

A binary read/write classification is insufficient for an external stage. It
would conflate values consumed by the stage, values merely kept alive across
it, values defined by it, and physical carrier words needed by a partial
write. The graph therefore exposes binding-local effects separately from a
stage-wide data-flow plan.

Let `A` be the unique logical frame-atom domain after aggregate decomposition
and before interference coloring. Let `phi(a) = (slot, slice)` map an atom to
its colored physical slot and, for packed Booleans, its bit slice. For the
extensions on one static boundary in declaration order, define:

```text
U_i = atoms read by extension i      (read or read_write bindings)
D_i = atoms must-defined by i        (write or read_write bindings)
L_m = complete resident live set required by the target continuation

L_i = U_i union (L_(i+1) - D_i)      for i = m-1 ... 0
```

Extensions execute in owner order. Reusing one logical atom in adjacent
extensions denotes successive versions of that atom: a definition by stage
`i` satisfies a use by stage `i + 1` through the same colored frame slot. It
does not create a second atom or a second extension-only slot.

Thus extension `i` has:

```text
live_in_i      = L_i
live_out_i     = L_(i+1)
preserve_i     = live_out_i - D_i
required_def_i = D_i intersect live_out_i
```

`live_out` is the complete token-indexed resident state, including values
first used by later continuations. It is not the smaller set loaded into the
immediately resumed kernel. `D_i` is a must-def set: every successful handler
path must produce those values. A `read_write` binding is in both `U_i` and
`D_i`, so its old value is live-in and its new value can be live-out.

`CoroSlotAccess` records the binding-local projections of `use`, `def`, and
read-modify-write carrier slots. The boundary's stage plan records the logical
and physical projections of `live_in`, `live_out`, `preserve`, and
`required_def`. A user executes an external pass as follows. Let `A_in` and
`A_out` be additional physical fields required by that user's scheduler or
application (for example, a debugger's coroutine id or an application-owned
residency tag). These are an explicit policy layered on the compiler
certificate; they are not inferred extension storage.

1. The user reconstructs a partial `CoroFrame` from the slot information and
   its scheduling/application policy. The minimum load set is
   `reconstruct_i = phi(U_i) union RMW_i`; the actual set is
   `reconstruct_i union A_in`. The user is responsible for ensuring every
   application-added field was initialized by the source transport.
2. Read and write bindings through their typed `CoroSlotAccess` callables.
   Those callables may access only the slots declared by the plan.
3. The user commits frame changes according to the slot information. Every
   member of `phi(required_def_i)` must be written back; explicitly requested
   `A_out` fields may also be committed. If scheduling relocates a frame to a
   different queue or storage allocation, transport `phi(live_in_i)` into the
   stage and `phi(live_out_i)` out of it. An in-place stage leaves
   `preserve_i` resident and performs no traffic for it.

The API deliberately does not own a hidden load/store kernel. The user chooses
AoS, SoA, shared memory, an out-of-core cache, or another storage policy and
uses the slot spans to build those operations. `CoroSlotAccess` callables only
project bindings into the user-created local `CoroFrame`; they neither fetch
nor commit backing storage themselves.

The source continuation uses the same transfer. If `K_e` is the set
must-defined by source code before edge `e`, `T_e` is the set touched on that
source path, and `G_e` is the reverse extension transfer above, then the
inter-scope fixed point propagates

```text
source_requirement_e = G_e(live_begin(target_e)) - K_e
source_store_e       = (G_e(live_begin(target_e)) intersect T_e)
                       union G_e(empty)
                       union designated_e
```

Consequently a value produced by an early external stage and consumed by a
later one is never loaded or spilled as an old source value. A value already
resident and untouched by the source is left in backing storage unless an
extension consumes it at this boundary. `G_e(empty)` is exactly the set of
extension inputs not satisfied by an earlier external definition; it also
captures source-callable arguments and replayable expressions that ordinary
`touched` analysis intentionally omits. The boundary exposes this exact spill
certificate as `source_store`, and exposes the terminal resident certificate
as `target_live`.

A write-only packed Boolean has no logical use, but its shared physical `uint`
carrier is an RMW slot because neighboring live bits must survive. Physical
slot sets may therefore overlap even when their logical atom sets are
disjoint.

The recurrence gives a direct completeness argument. Assume the backing frame
represents `live_in_i`. Every stage use is reconstructable, every definition
is produced, and every member of `preserve_i` remains unchanged. Writing back
`required_def_i` therefore makes the backing frame represent `live_out_i`.
Induction over the extension sequence yields `L_m`, exactly the state required
by the target continuation. Omitting any logical use, RMW carrier, or required
definition admits a program that observes a missing or corrupted value.

The source fixed point supplies the induction base: every member of `L_0` is
either freshly stored because the source touched it, already resident because
the source did not touch it, or designated explicitly. Source must-defs remove
the need to load older versions. Thus the three user operations--reconstruct,
invoke typed binding access, and commit modified slots--are sufficient for
both in-place and relocating schedulers.

Loading or storing the complete frame is not a valid shortcut. Coloring makes
slots outside the current live domain intentionally undefined, and a dormant
pass-through value need not be loaded merely because it shares the frame
allocation.

A write operand is a data-flow definition made by the extension between the
source scope and the resumed scope. AST-to-XIR must preserve the logical lvalue
and access chain rather than load it as an ordinary input. CFG distillation
models the extension write in the transition kill/store sets, frame coloring
allocates its destination even when it had no value before suspension, and
materialization replaces the logical binding with a physical frame field and
optional bit/access path. Subsequent continuation loads consume that new
definition.

## Boundary identity and graph promotion

An extension is declared on a suspend boundary, not intrinsically on a target
continuation. The distinction matters when several source scopes suspend to
the same token.

The XIR representation therefore retains a stable boundary identity and its
extension descriptors. `CoroGraph` initially records them on boundary records
owned by transition edges.

Some extensions describe a destination queue. A queue consumer may promote
such an extension to a target node only when all reachable incoming boundaries
declare compatible schema versions, binding types, attribute values, and merge
semantics. An advisory extension may be dropped according to fallback when
promotion is impossible. A required extension makes incompatibility an error.

Graph construction must not silently coalesce two differently annotated
suspend instructions merely because their source and destination scope indices
match.

## Compilation lifecycle

Extensions pass through the following representations:

1. **DSL recording** consumes a plugin-created
   `unique_ptr<CoroSuspendExtension>` and freezes it to a normalized,
   data-backed implementation plus typed owner bindings.
2. **AST** owns normalized extension pointers and expression/lvalue bindings in
   `SuspendStmt`. Statement hashing, callable serialization, duplication,
   visitors, and JSON output query the virtual interface and include them.
3. **AST to XIR** clones normalized extension records onto `CoroSuspendInst`
   and remaps owner binding indices to XIR operands and logical lvalue paths.
4. **XIR transforms** preserve descriptors one-to-one unless a registered
   extension pass explicitly consumes or rewrites them. Clone, verifier,
   textual printing, and interchange cover the complete descriptor.
5. **Coroutine analysis** treats queued/resumed operands as designated frame
   values and retains per-boundary descriptor provenance.
6. **CoroGraph materialization** replaces queued operand references with
   physical frame-field indices and records boundary and promoted queue
   descriptors.
7. **Scheduler planning, external stages, and observers** negotiate supported
   schemas and mark descriptors as consumed or intentionally ignored.
8. **Final validation** applies the fallback policy to every unconsumed
   descriptor.

No pass may delete or combine an instruction carrying an unconsumed semantic
extension unless it supplies a documented transfer rule. Diagnostic-only
metadata follows the existing diagnostic metadata policy; suspend extensions
are semantic IR data, not comments.

## Consumers

Data objects expose virtual queries to AST, XIR, and graph passes. Behavior is a
separate interface: consumers register against a stable schema and supported
version interval, not the source factory's C++ derived type:

```cpp
class CoroSuspendExtensionHandler {
public:
    virtual ~CoroSuspendExtensionHandler() noexcept = default;
    [[nodiscard]] virtual luisa::string_view schema() const noexcept = 0;
    [[nodiscard]] virtual CoroSuspendVersionRange versions() const noexcept = 0;
    [[nodiscard]] virtual CoroSuspendPhase phase() const noexcept = 0;
    virtual void consume(CoroSuspendExtensionContext &) noexcept = 0;
};
```

A phase-aware implementation may expose `on_ast`, `on_xir`, `on_graph`, or
`on_scheduler` through the phase context, but those behavior objects are owned
by an explicit pass/compile/scheduler context and are never serialized inside
the extension data object.

Handlers are owned by an explicit compile, graph, scheduler, or debugger
context. A process-global mutable registry is not the primary API because it
would make plugin lifetime, tests, and concurrent compilation nondeterministic.

The supported phases are:

- XIR analysis or transform;
- graph transform;
- scheduler queue planning;
- runtime stage dispatch;
- debugger/profiler observation.

A handler reports whether it consumed, transformed, promoted, or deliberately
ignored a descriptor. This report drives final fallback validation and is
available to diagnostics.

## Built-in scheduling annotations

The first built-in schemas are scheduling annotations:

- `luisa.coro.schedule.sort`, version 1;
- `luisa.coro.schedule.dispatch`, version 1.

A sort annotation contains one or more unsigned key operands, each with an
exclusive range and priority. Multiple keys define lexicographic order.
Partitioning by stable frame or path state is expressed as another key, not as
an out-of-band shader specialization constant. Runtime key ranges and partition
sizes are shader arguments when they do not change the selected sorting
algorithm or scratch topology.

A scheduler may fuse compatible keys into one integer, perform a stable
multi-pass sort, or ignore advisory keys. It must report the selected plan.
This allows a surface queue to request shader coherence while a later
intersection queue requests spatial or state coherence without imposing one
global order.

## External stages

`luisa.coro.stage` is semantic and defaults to `reject`. Its descriptor names a
stage provider schema and typed read, write, and read-write bindings. The
recommended builder spelling makes every binding explicit:

```cpp
$suspend("nn_shade",
         coro_stage("com.example.nn-shade")
             .read("features", features)
             .read_write("throughput", throughput)
             .write("closure", closure));
```

A handler may:

- insert an XIR transformation before coroutine splitting;
- add a graph node and frame fields during graph transformation; or
- schedule an external kernel between queue production and continuation
  resume.

The handler must declare ownership of output initialization, synchronization,
failure propagation, and frame relocation. A stage that writes resumed values
participates in frame liveness and graph edge store sets. A write-only binding
must be definitely written on every successful stage path before the target
continuation can resume. A scheduler must not model a semantic stage as a mere
profiling callback or resume with the pre-stage value after failure.

## Debugger and profiler annotations

Debugging annotations may carry source labels, predicates, and watched queued
operands. They are semantic IR records even when they do not change kernel
results. Device-side printing is not implied. A debugger handler may capture to
a dedicated buffer, stop queue consumption, or emit host events.

Profiling labels that do not affect code generation are kept out of shader
identity and attached through the runtime shader map. A watched expression or
instrumentation kernel does affect the relevant AST/XIR hash.

## Hashing and shader cache identity

The following participate in the coroutine/program hash:

- schema and version;
- annotation flag and fallback policy;
- binding types, access modes, lifetimes, and expressions;
- canonical typed attributes;
- transformations or inserted stage code selected by a handler.

Runtime scheduler values do not enter a continuation shader AST merely because
they influence host scheduling. They are passed as runtime arguments. A value
enters a sort helper's identity only when it changes algorithm topology, such
as choosing a bucket implementation instead of a radix implementation.

Diagnostic labels that neither alter code nor request captured values use the
existing runtime shader-map path and do not perturb AOT/cache identity.

## Compatibility and migration

The migration order is:

1. implement descriptor, AST ownership, hashing, duplication, serialization,
   JSON, and DSL tests;
2. implement XIR instruction storage, clone, verifier, text and interchange
   round trips, and immutable-analysis certificate coverage;
3. retain descriptors through CFG distillation and materialization into
   `CoroGraph` boundary records;
4. implement the built-in sort handler in wavefront schedulers;
5. migrate `coro_hint` plus `hint_fields` callers to
   `luisa.coro.schedule.sort`;
6. deprecate the magic frame alias and configuration list only after both APIs
   have overlapping validation coverage.

The old API and new annotation must not be active for the same queue. Scheduler
construction rejects that ambiguous configuration.

## Required regression coverage

Each implementation slice includes tests for:

- move-only extension ownership and destruction exactly once;
- annotation and semantic-extension fallback independently;
- AST hash changes for schema, version, attribute, binding, and fallback;
- callable serialization and duplication preserving descriptors;
- AST JSON preserving canonical descriptors;
- AST-to-XIR operand and descriptor mapping;
- XIR clone, verifier, text, and interchange round trips;
- XIR malformed binding ranges, duplicate attributes, and unsupported versions;
- CFG analysis retaining queued values even when otherwise replayable or dead;
- write and read-write lvalues surviving AST-to-XIR without accidental loads,
  including scalar and aggregate access-path outputs;
- definite stage output initialization and continuation reads observing the
  stage definition rather than the pre-suspend value;
- immutable CFG certificate hashing all extension fields;
- multiple incoming boundaries with compatible and incompatible promotion;
- graph relocation preserving extension bindings in AoS and SoA layouts;
- scheduler support, intentional ignore, warning, and required rejection;
- sort correctness, multi-key ordering, cache-key stability for runtime values,
  and image equivalence after the Psycles migration;
- external-stage ordering and output liveness;
- debugger observation without implicit device printing.
