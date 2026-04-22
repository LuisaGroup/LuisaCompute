#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/map.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class CallableFunction;
class Instruction;
class Value;

using CoroInstructionRef = uint32_t;
using CoroScopeRef = uint32_t;

inline constexpr CoroInstructionRef invalid_coro_instruction_ref = ~0u;
inline constexpr CoroScopeRef invalid_coro_scope_ref = ~0u;

struct CoroConditionStackItem {
    Value *value{nullptr};
    int32_t selected_value{0};
};

struct CoroSwitchCase {
    int32_t value{0};
    luisa::vector<CoroInstructionRef> body;
};

enum class CoroInstructionTag : uint32_t {
    ENTRY,
    ENTRY_SCOPE,
    SIMPLE,
    CONDITION_STACK_REPLAY,
    MAKE_FIRST_FLAG,
    SKIP_IF_FIRST_FLAG,
    CLEAR_FIRST_FLAG,
    IF,
    SWITCH,
    SIMPLE_LOOP,
    LOOP_CONTINUE,
    LOOP_BREAK,
    SUSPEND,
    TERMINATE
};

struct CoroInstruction {
    CoroInstructionTag tag{CoroInstructionTag::SIMPLE};
    Instruction *simple{nullptr};
    Value *condition{nullptr};
    luisa::vector<CoroConditionStackItem> condition_stack;
    CoroInstructionRef related_instruction{invalid_coro_instruction_ref};
    luisa::vector<CoroInstructionRef> body;
    luisa::vector<CoroInstructionRef> true_branch;
    luisa::vector<CoroInstructionRef> false_branch;
    luisa::vector<CoroSwitchCase> cases;
    luisa::vector<CoroInstructionRef> default_body;
    uint32_t token{0u};
};

struct CoroScope {
    luisa::vector<CoroInstructionRef> instructions;
    luisa::unordered_map<luisa::string, Value *> designated_values;
};

struct CoroGraph {
    luisa::vector<CoroScope> scopes;
    CoroScopeRef entry{invalid_coro_scope_ref};
    luisa::map<uint32_t, CoroScopeRef> tokens;
    luisa::vector<CoroInstruction> instructions;
    luisa::unordered_map<luisa::string, Value *> designated_values;
};

struct CoroAccessPath {
    Value *root{nullptr};
    luisa::vector<int32_t> chain;
};

struct CoroScopeUseDef {
    luisa::vector<CoroAccessPath> external_uses;
    luisa::vector<CoroAccessPath> internal_touches;
    luisa::unordered_map<CoroScopeRef, luisa::vector<CoroAccessPath>> internal_kills;
};

struct CoroGraphUseDef {
    luisa::unordered_map<CoroScopeRef, CoroScopeUseDef> scopes;
    luisa::vector<CoroAccessPath> union_uses;
};

struct CoroTransitionEdge {
    CoroScopeRef target{invalid_coro_scope_ref};
    luisa::vector<CoroAccessPath> live_states;
    luisa::vector<CoroAccessPath> states_to_load;
    luisa::vector<CoroAccessPath> states_to_save;
};

struct CoroTransitionState {
    CoroScopeRef scope{invalid_coro_scope_ref};
    luisa::map<uint32_t, CoroTransitionEdge> outlets;
    luisa::vector<CoroAccessPath> union_live_states;
    luisa::vector<CoroAccessPath> union_states_to_load;
    luisa::vector<CoroAccessPath> union_states_to_save;
};

struct CoroTransitionGraph {
    luisa::vector<CoroAccessPath> union_states;
    luisa::unordered_map<CoroScopeRef, CoroTransitionState> nodes;
};

[[nodiscard]] LUISA_XIR_API CoroGraph compute_coro_graph(CallableFunction *function) noexcept;
[[nodiscard]] LUISA_XIR_API CoroGraphUseDef compute_coro_use_def(const CoroGraph &graph) noexcept;
[[nodiscard]] LUISA_XIR_API CoroTransitionGraph compute_coro_transition_graph(const CoroGraph &graph,
                                                                              const CoroGraphUseDef &use_def) noexcept;

}// namespace luisa::compute::xir
