#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>

// CoroGraph analysis (XIR port of the Rust reference impl's coro_graph.rs).
//
// Two-stage analysis:
//
//   1) PRELIMINARY GRAPH: walk the structured XIR CFG once and produce a flat
//      vector of CoroInstruction nodes, one per source IR node, plus indexed
//      references for nested control flow (Loop/If/Switch). Suspend markers
//      become Suspend nodes; returns/unreachables become Terminate nodes.
//
//   2) SPLIT SCOPES: traverse the preliminary graph from the entry, collect
//      reachable nodes per scope (one scope per suspend + entry), insert the
//      first-flag and condition-replay constructs the materializer needs to
//      reconstruct control flow on resume.
//
// The materializer (coroutine_split) consumes the result to emit one
// CallableFunction per scope.

namespace luisa::compute::xir {

class BasicBlock;
class Function;
class Instruction;
class Module;
class Value;

}// namespace luisa::compute::xir

namespace luisa::compute::xir::coro {

struct CoroInstrRef {
    size_t index{static_cast<size_t>(-1)};
    [[nodiscard]] bool valid() const noexcept { return index != static_cast<size_t>(-1); }
    [[nodiscard]] friend bool operator==(CoroInstrRef a, CoroInstrRef b) noexcept { return a.index == b.index; }
};

struct CoroScopeRef {
    size_t index{static_cast<size_t>(-1)};
    [[nodiscard]] bool valid() const noexcept { return index != static_cast<size_t>(-1); }
    [[nodiscard]] friend bool operator==(CoroScopeRef a, CoroScopeRef b) noexcept { return a.index == b.index; }
};

// One frame of the condition stack: which control-flow node we're inside, and
// which branch the path-to-suspend took. `value` is 1/0 for if, the case value
// for switch, and unused (loops are not recorded — the first-flag mechanism
// handles them).
struct ConditionStackItem {
    Instruction *control_flow_inst{};
    Value *condition_value{};// for replay: the value that drove this branch
    int32_t value{};
};

struct CoroSwitchCase {
    int32_t value{};
    luisa::vector<CoroInstrRef> body;
};

// Tagged-union node in the preliminary graph. Variants mirror the Rust ref
// (`enum CoroInstruction`). Indexing is by CoroInstrRef into the owning
// CoroPreliminaryGraph::instructions vector.
struct CoroInstruction {
    enum struct Tag {
        ENTRY,
        ENTRY_SCOPE,
        SIMPLE,// any non-control-flow IR instruction
        CONDITION_STACK_REPLAY,
        MAKE_FIRST_FLAG,
        SKIP_IF_FIRST_FLAG,
        CLEAR_FIRST_FLAG,
        LOOP,
        IF,
        SWITCH,
        SUSPEND,
        TERMINATE,
    };

    Tag tag{Tag::ENTRY};
    Instruction *source_inst{};// for SIMPLE / IF / SWITCH / LOOP / SUSPEND nodes

    // ENTRY_SCOPE / LOOP body
    luisa::vector<CoroInstrRef> body;

    // IF
    CoroInstrRef cond;
    luisa::vector<CoroInstrRef> true_branch;
    luisa::vector<CoroInstrRef> false_branch;

    // SWITCH
    luisa::vector<CoroSwitchCase> cases;
    luisa::vector<CoroInstrRef> default_branch;

    // SUSPEND
    uint32_t suspend_token{};

    // CONDITION_STACK_REPLAY
    luisa::vector<ConditionStackItem> replay_items;

    // SKIP_IF_FIRST_FLAG / CLEAR_FIRST_FLAG / MAKE_FIRST_FLAG references the
    // owning first-flag node by CoroInstrRef.
    CoroInstrRef first_flag;
};

struct CoroScope {
    luisa::vector<CoroInstrRef> instructions;
    luisa::unordered_map<luisa::string, Value *> designated_values;
};

// Result of stage (1).
struct CoroPreliminaryGraph {
    luisa::vector<CoroInstruction> instructions;
    luisa::unordered_map<Instruction *, CoroInstrRef> source_to_instr;
    luisa::unordered_set<size_t> terminators;// indices that terminate (suspend, return, unreachable, recursive)
    CoroInstrRef entry_scope;
    luisa::vector<luisa::string> diagnostics;
};

// Result of stage (2).
struct CoroGraphInfo {
    bool ok{};
    CoroPreliminaryGraph preliminary;
    luisa::vector<CoroScope> scopes;     // scopes[0] is the entry scope
    luisa::unordered_map<uint32_t, CoroScopeRef> token_to_scope;
    luisa::unordered_map<luisa::string, Value *> designated_values;
    luisa::vector<luisa::string> diagnostics;
};

// Stage 1: build the preliminary graph from a function definition.
[[nodiscard]] LUISA_XIR_API CoroPreliminaryGraph
coro_preliminary_graph_build(Function *function) noexcept;

// Stage 2: split into scopes. Consumes the preliminary graph (moves out).
[[nodiscard]] LUISA_XIR_API CoroGraphInfo
coro_graph_split(CoroPreliminaryGraph preliminary) noexcept;

// Convenience: full pipeline.
[[nodiscard]] LUISA_XIR_API CoroGraphInfo
coro_graph_run_on_function(Function *function) noexcept;

}// namespace luisa::compute::xir::coro
