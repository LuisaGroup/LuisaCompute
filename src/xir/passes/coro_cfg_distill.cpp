#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/deque.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/verifier.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

class DistillCertificateHasher {

private:
    uint64_t _state{luisa::hash64_default_seed};

public:
    template<typename T>
        requires(std::is_integral_v<T> || std::is_enum_v<T>)
    void add(T value) noexcept {
        _state = luisa::hash64(&value, sizeof(value), _state);
    }

    void add_pointer(const void *value) noexcept {
        auto bits = reinterpret_cast<uintptr_t>(value);
        add(bits);
    }

    void add_string(luisa::string_view value) noexcept {
        add(value.size());
        if (!value.empty()) {
            _state = luisa::hash64(value.data(), value.size(), _state);
        }
    }

    [[nodiscard]] uint64_t finish() const noexcept { return _state; }
};

static void hash_optional_string(DistillCertificateHasher &h,
                                 const luisa::optional<luisa::string> &value) noexcept {
    h.add(value.has_value());
    if (value.has_value()) { h.add_string(*value); }
}

static void hash_optional_token(DistillCertificateHasher &h,
                                const luisa::optional<uint32_t> &value) noexcept {
    h.add(value.has_value());
    if (value.has_value()) { h.add(*value); }
}

[[nodiscard]] static uint64_t compute_distill_validation_hash(
    const CoroCfgDistillResult &result,
    const FunctionDefinition *definition) noexcept {
    DistillCertificateHasher h;
    // Version the schema so adding a semantic field cannot silently retain a
    // certificate computed by an older layout.
    h.add(uint64_t{1u});
    h.add_pointer(definition);
    if (definition != nullptr) {
        h.add_pointer(definition->body_block());
        h.add(definition->arguments().count_size());
        for (auto *argument : definition->arguments()) {
            h.add_pointer(argument);
            h.add_pointer(argument->type());
            h.add(argument->derived_argument_tag());
        }
        h.add(definition->basic_blocks().count_size());
        for (auto *block : definition->basic_blocks()) {
            h.add_pointer(block);
            h.add(block->instructions().count_size());
            for (auto *instruction : block->instructions()) {
                h.add_pointer(instruction);
                h.add(instruction->derived_instruction_tag());
                h.add_pointer(instruction->type());
                h.add(instruction->is_lvalue());
                h.add(instruction->is_terminator());
                h.add_string(instruction->intrinsic_identifier());
                if (auto name = instruction->name()) {
                    h.add(true);
                    h.add_string(*name);
                } else {
                    h.add(false);
                }
                h.add(instruction->operand_uses().size());
                for (auto *operand : instruction->operand_uses()) {
                    h.add_pointer(operand->value());
                }
                switch (instruction->derived_instruction_tag()) {
                    case DerivedInstructionTag::CORO_SUSPEND: {
                        auto *suspend =
                            static_cast<const CoroSuspendInst *>(instruction);
                        h.add(suspend->token());
                        h.add_string(suspend->name());
                        break;
                    }
                    case DerivedInstructionTag::CORO_RESUME:
                        h.add(static_cast<const CoroResumeInst *>(instruction)->token());
                        break;
                    default: break;
                }
            }
        }
    }

    h.add(result.scopes.size());
    for (auto &scope : result.scopes) {
        h.add(scope.blocks.size());
        for (auto *block : scope.blocks) { h.add_pointer(block); }
        h.add(scope.suspend_points.size());
        for (auto &point : scope.suspend_points) {
            h.add_pointer(point.block);
            h.add(point.token);
            h.add_string(point.name);
        }
        h.add(scope.scope_id);
        hash_optional_token(h, scope.suspend_token);
        hash_optional_string(h, scope.suspend_name);
        h.add(scope.trigger_token);
        hash_optional_string(h, scope.trigger_name);
        auto hash_values = [&](auto &values) noexcept {
            h.add(values.size());
            for (auto *value : values) { h.add_pointer(value); }
        };
        auto hash_names = [&](auto &names) noexcept {
            h.add(names.size());
            for (auto &name : names) { h.add_string(name); }
        };
        hash_values(scope.external_values);
        hash_values(scope.touched_values);
        hash_values(scope.live_in_values);
        hash_values(scope.live_out_values);
        hash_names(scope.external_variables);
        hash_names(scope.touched_variables);
        hash_names(scope.live_in_variables);
        hash_names(scope.live_out_variables);
        h.add(scope.is_terminal);
    }

    h.add(result.edges.size());
    for (auto &targets : result.edges) {
        h.add(targets.size());
        for (auto target : targets) { h.add(target); }
    }

    h.add(result.transition_edges.size());
    for (auto &edge : result.transition_edges) {
        h.add(edge.from_scope);
        h.add(edge.to_scope);
        h.add(edge.token);
        h.add_pointer(edge.exit_block);
        h.add(edge.is_suspend);
        auto hash_values = [&](auto &values) noexcept {
            h.add(values.size());
            for (auto *value : values) { h.add_pointer(value); }
        };
        auto hash_names = [&](auto &names) noexcept {
            h.add(names.size());
            for (auto &name : names) { h.add_string(name); }
        };
        hash_values(edge.killed_values);
        hash_values(edge.touched_values);
        hash_values(edge.live_values);
        hash_values(edge.store_values);
        hash_names(edge.killed_variables);
        hash_names(edge.touched_variables);
        hash_names(edge.live_variables);
        hash_names(edge.store_variables);
    }

    h.add(result.frame_values.size());
    for (auto &frame_value : result.frame_values) {
        h.add_pointer(frame_value.value);
        h.add_string(frame_value.name);
        h.add_pointer(frame_value.type);
    }
    h.add(result.structured_cfg_error_count);
    h.add(result.invalid_input_error_count);
    h.add(result.invalid_cfg_error_count);
    return h.finish();
}

// Coroutine scopes are cloned into void continuation callables. Phi nodes need
// edge-specific values, but suspend/resume transitions are not ordinary CFG
// predecessors in XIR; treating a Phi as a block-local definition therefore
// loses the value selected on the continuation edge. Require reg2mem at this
// boundary instead of silently manufacturing a non-equivalent Phi.
[[nodiscard]] static bool validate_coroutine_input_language(
    FunctionDefinition *def) noexcept {
    if (def == nullptr || def->body_block() == nullptr ||
        def->type() != nullptr ||
        def->find_metadata<SignatureConstraintMD>() != nullptr) {
        return false;
    }
    for (auto *block : def->basic_blocks()) {
        auto resume_count = 0u;
        for (auto *inst : block->instructions()) {
            if (inst->isa<PhiInst>()) { return false; }
            if (inst->isa<CoroResumeInst>()) { ++resume_count; }
        }
        // A block has one continuation identity. Two resume tokens in one
        // block create two roots with the same block set; a resume in the
        // ordinary entry block aliases scope zero for the same reason.
        if (resume_count > 1u ||
            (block == def->body_block() && resume_count != 0u)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static luisa::optional<uint32_t> resume_token_of_block(BasicBlock *bb) noexcept {
    if (bb == nullptr) { return luisa::nullopt; }
    for (auto *inst : bb->instructions()) {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
            return static_cast<CoroResumeInst *>(inst)->token();
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] static bool coro_cfg_distill_validate_coroutine_tokens(
    FunctionDefinition *def) noexcept {
    if (def == nullptr) { return false; }
    luisa::unordered_set<uint32_t> suspend_tokens;
    luisa::unordered_set<uint32_t> resume_tokens;
    auto valid = true;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || block->parent_function() != def ||
            !block->is_terminated()) {
            valid = false;
            continue;
        }
        for (auto *inst : block->instructions()) {
            if (inst->derived_instruction_tag() ==
                DerivedInstructionTag::CORO_SUSPEND) {
                auto token = static_cast<CoroSuspendInst *>(inst)->token();
                valid &= token != 0u && token != TERMINAL_TOKEN &&
                         suspend_tokens.emplace(token).second;
            } else if (inst->derived_instruction_tag() ==
                       DerivedInstructionTag::CORO_RESUME) {
                auto token = static_cast<CoroResumeInst *>(inst)->token();
                valid &= token != 0u && token != TERMINAL_TOKEN &&
                         resume_tokens.emplace(token).second;
            }
        }
    }
    if (suspend_tokens.size() != resume_tokens.size()) { return false; }
    for (auto token : suspend_tokens) {
        if (!resume_tokens.contains(token)) { return false; }
    }
    return valid;
}

[[nodiscard]] static luisa::unordered_set<BasicBlock *> collect_coro_reachable_blocks(
    FunctionDefinition *def,
    const luisa::unordered_map<uint32_t, BasicBlock *> &token_to_resume) noexcept {
    luisa::unordered_set<BasicBlock *> reachable;
    luisa::deque<BasicBlock *> worklist;
    worklist.emplace_back(def->body_block());
    while (!worklist.empty()) {
        auto *bb = worklist.front();
        worklist.pop_front();
        if (bb == nullptr) { continue; }
        if (!reachable.emplace(bb).second) { continue; }
        if (!bb->is_terminated()) { continue; }
        auto *term = bb->terminator();
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::CORO_SUSPEND: {
                auto *suspend = static_cast<CoroSuspendInst *>(term);
                if (auto it = token_to_resume.find(suspend->token());
                    it != token_to_resume.end()) {
                    worklist.emplace_back(it->second);
                }
                break;
            }
            case DerivedInstructionTag::CORO_TERMINATE:
                break;
            default:
                bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                    worklist.emplace_back(succ);
                });
                break;
        }
    }
    return reachable;
}

[[nodiscard]] static AllocaInst *trace_local_alloca(Value *value) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ALLOCA: {
                auto *alloca = static_cast<AllocaInst *>(inst);
                return alloca->is_local() ? alloca : nullptr;
            }
            case DerivedInstructionTag::GEP: {
                value = static_cast<GEPInst *>(inst)->base();
                break;
            }
            default:
                return nullptr;
        }
    }
    return nullptr;
}

[[nodiscard]] static luisa::optional<luisa::string> local_alloca_name(Value *value) noexcept {
    if (auto *alloca = trace_local_alloca(value)) {
        if (auto name = alloca->name()) {
            return luisa::string{name.value()};
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] static bool is_always_available(Value *value) noexcept {
    if (value == nullptr) { return true; }
    switch (value->derived_value_tag()) {
        case DerivedValueTag::UNDEFINED:
        case DerivedValueTag::FUNCTION:
        case DerivedValueTag::BASIC_BLOCK:
        case DerivedValueTag::CONSTANT:
        case DerivedValueTag::ARGUMENT:
        case DerivedValueTag::SPECIAL_REGISTER:
            return true;
        case DerivedValueTag::INSTRUCTION:
            return false;
    }
    return true;
}

[[nodiscard]] static bool is_frameable_ssa_value(Value *value) noexcept {
    if (value == nullptr || value->type() == nullptr || value->is_lvalue()) { return false; }
    if (value->derived_value_tag() != DerivedValueTag::INSTRUCTION) { return false; }
    auto *inst = static_cast<Instruction *>(value);
    return !inst->is_terminator();
}

[[nodiscard]] static Value *frame_value_for_operand(Value *value) noexcept {
    if (auto *alloca = trace_local_alloca(value)) { return alloca; }
    return is_frameable_ssa_value(value) ? value : nullptr;
}

[[nodiscard]] static luisa::string frame_value_name(Value *value, size_t index) noexcept {
    if (auto *alloca = trace_local_alloca(value)) {
        if (auto name = alloca->name()) {
            return luisa::string{name.value()};
        }
    }
    return luisa::format("_coro_frame_{}", index);
}

static void append_sorted_names(luisa::vector<luisa::string> &dst,
                                const luisa::unordered_set<Value *> &src,
                                const luisa::unordered_map<Value *, luisa::string> &names) noexcept {
    dst.clear();
    dst.reserve(src.size());
    for (auto *value : src) {
        if (auto it = names.find(value); it != names.end()) {
            dst.emplace_back(it->second);
        }
    }
    std::sort(dst.begin(), dst.end());
}

static void append_ordered_values(luisa::vector<Value *> &dst,
                                  const luisa::unordered_set<Value *> &src,
                                  const luisa::unordered_map<Value *, size_t> &order) noexcept {
    dst.clear();
    dst.reserve(src.size());
    for (auto *value : src) { dst.emplace_back(value); }
    std::sort(dst.begin(), dst.end(), [&](auto *lhs, auto *rhs) noexcept {
        auto li = order.find(lhs);
        auto ri = order.find(rhs);
        auto lo = li == order.end() ? static_cast<size_t>(-1) : li->second;
        auto ro = ri == order.end() ? static_cast<size_t>(-1) : ri->second;
        return lo < ro;
    });
}

static void sort_frame_values_by_layout(luisa::vector<Value *> &values) noexcept {
    std::stable_sort(values.begin(), values.end(), [](auto *lhs, auto *rhs) noexcept {
        auto *lt = lhs->type();
        auto *rt = rhs->type();
        if (lt->alignment() != rt->alignment()) {
            return lt->alignment() > rt->alignment();
        }
        return lt->size() > rt->size();
    });
}

[[nodiscard]] static bool same_set(const luisa::unordered_set<Value *> &a,
                                   const luisa::unordered_set<Value *> &b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (auto &v : a) {
        if (!b.contains(v)) { return false; }
    }
    return true;
}

struct ScopeDataflowState {
    luisa::unordered_set<Value *> killed;
    luisa::unordered_set<Value *> external;
    luisa::unordered_set<Value *> touched;
};

class DenseValueSet {

private:
    luisa::vector<uint64_t> _words;

public:
    explicit DenseValueSet(size_t bit_count = 0u) noexcept
        : _words((bit_count + 63u) / 64u, 0u) {}

    [[nodiscard]] static DenseValueSet full(size_t bit_count) noexcept {
        DenseValueSet result{bit_count};
        std::fill(result._words.begin(), result._words.end(), ~uint64_t{0u});
        if (auto tail_bit_count = bit_count % 64u;
            tail_bit_count != 0u) {
            result._words.back() &=
                (uint64_t{1u} << tail_bit_count) - uint64_t{1u};
        }
        return result;
    }

    void set(size_t index) noexcept {
        _words[index / 64u] |= uint64_t{1u} << (index % 64u);
    }

    [[nodiscard]] bool test(size_t index) const noexcept {
        return (_words[index / 64u] &
                (uint64_t{1u} << (index % 64u))) != 0u;
    }

    void union_with(const DenseValueSet &other) noexcept {
        for (size_t i = 0u; i < _words.size(); ++i) {
            _words[i] |= other._words[i];
        }
    }

    void intersect_with(const DenseValueSet &other) noexcept {
        for (size_t i = 0u; i < _words.size(); ++i) {
            _words[i] &= other._words[i];
        }
    }

    void subtract(const DenseValueSet &other) noexcept {
        for (size_t i = 0u; i < _words.size(); ++i) {
            _words[i] &= ~other._words[i];
        }
    }

    [[nodiscard]] bool operator==(const DenseValueSet &other) const noexcept {
        return _words == other._words;
    }
};

static void touch_value(Value *value, ScopeDataflowState &state) noexcept {
    if (value == nullptr) { return; }
    state.killed.emplace(value);
    state.touched.emplace(value);
}

static void may_touch_value(Value *value, ScopeDataflowState &state) noexcept {
    if (value == nullptr) { return; }
    state.touched.emplace(value);
}

static void use_value(Value *value, ScopeDataflowState &state) noexcept {
    if (is_always_available(value)) { return; }
    if (auto *frame_value = frame_value_for_operand(value)) {
        if (!state.killed.contains(frame_value)) {
            state.external.emplace(frame_value);
        }
    }
}

static void use_pointer_indices(Value *value, ScopeDataflowState &state) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        if (inst->derived_instruction_tag() != DerivedInstructionTag::GEP) { break; }
        auto *gep = static_cast<GEPInst *>(inst);
        for (size_t i = 0u; i < gep->index_count(); ++i) {
            use_value(gep->index(i), state);
        }
        value = gep->base();
    }
}

static void transfer_call_instruction(CallInst *call, ScopeDataflowState &state) noexcept {
    auto arg_iter = call->callee()->arguments().begin();
    for (auto *arg_use : call->argument_uses()) {
        auto *argument = arg_use->value();
        if (arg_iter != call->callee()->arguments().end() &&
            (*arg_iter)->is_reference()) {
            use_pointer_indices(argument, state);
            if (auto *alloca = trace_local_alloca(argument)) {
                use_value(alloca, state);
                may_touch_value(alloca, state);
            } else {
                use_value(argument, state);
            }
        } else {
            use_value(argument, state);
        }
        if (arg_iter != call->callee()->arguments().end()) { ++arg_iter; }
    }
    if (call->type() != nullptr && !call->is_lvalue()) {
        touch_value(call, state);
    }
}

static void transfer_instruction(Instruction *inst, ScopeDataflowState &state) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA:
            break;
        case DerivedInstructionTag::CALL: {
            transfer_call_instruction(static_cast<CallInst *>(inst), state);
            break;
        }
        case DerivedInstructionTag::LOAD: {
            auto *load = static_cast<LoadInst *>(inst);
            use_pointer_indices(load->variable(), state);
            if (auto *alloca = trace_local_alloca(load->variable())) {
                use_value(alloca, state);
                touch_value(inst, state);
            } else {
                use_value(load->variable(), state);
                touch_value(inst, state);
            }
            break;
        }
        case DerivedInstructionTag::STORE: {
            auto *store = static_cast<StoreInst *>(inst);
            use_value(store->value(), state);
            use_pointer_indices(store->variable(), state);
            if (auto *alloca = trace_local_alloca(store->variable())) {
                if (store->variable() == alloca) {
                    touch_value(alloca, state);
                } else {
                    use_value(alloca, state);
                    may_touch_value(alloca, state);
                }
            } else {
                use_value(store->variable(), state);
            }
            break;
        }
        case DerivedInstructionTag::ATOMIC: {
            auto *atomic = static_cast<AtomicInst *>(inst);
            for (auto *index : atomic->index_uses()) { use_value(index->value(), state); }
            for (auto *value : atomic->value_uses()) { use_value(value->value(), state); }
            if (auto *alloca = trace_local_alloca(atomic->base())) {
                use_value(alloca, state);
                may_touch_value(alloca, state);
            } else {
                use_value(atomic->base(), state);
            }
            if (atomic->type() != nullptr && !atomic->is_lvalue()) { touch_value(atomic, state); }
            break;
        }
        case DerivedInstructionTag::CORO_SUSPEND:
        case DerivedInstructionTag::CORO_TERMINATE:
            break;
        default: {
            for (auto *op_use : inst->operand_uses()) {
                use_value(op_use->value(), state);
            }
            if (inst->type() != nullptr && !inst->is_lvalue() && !inst->is_terminator()) {
                touch_value(inst, state);
            }
            break;
        }
    }
}

struct ScopeDataflowResult {
    luisa::unordered_set<Value *> external;
    luisa::unordered_set<Value *> touched;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<Value *>> killed_at_exit;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<Value *>> touched_at_exit;
};

[[nodiscard]] static bool same_pointer_state(
    const ScopeDataflowState &a,
    const ScopeDataflowState &b) noexcept {
    return same_set(a.killed, b.killed) &&
           same_set(a.external, b.external) &&
           same_set(a.touched, b.touched);
}

static void merge_pointer_state_into_entry(
    ScopeDataflowState &dst,
    const ScopeDataflowState &src,
    bool first_predecessor) noexcept {
    for (auto *value : src.external) { dst.external.emplace(value); }
    for (auto *value : src.touched) { dst.touched.emplace(value); }
    if (first_predecessor) {
        dst.killed = src.killed;
    } else {
        luisa::unordered_set<Value *> killed;
        for (auto *value : dst.killed) {
            if (src.killed.contains(value)) { killed.emplace(value); }
        }
        dst.killed = std::move(killed);
    }
}

[[nodiscard]] static ScopeDataflowResult
analyze_scope_use_def_pointer_oracle(
    const CoroCfgDistillResult::Scope &scope) noexcept {
    ScopeDataflowResult result;
    if (scope.blocks.empty()) { return result; }
    luisa::unordered_set<BasicBlock *> scope_blocks;
    for (auto *block : scope.blocks) { scope_blocks.emplace(block); }
    luisa::unordered_map<BasicBlock *, ScopeDataflowState> in_states;
    luisa::unordered_map<BasicBlock *, ScopeDataflowState> out_states;
    for (;;) {
        auto changed = false;
        for (auto *block : scope.blocks) {
            ScopeDataflowState next_in;
            auto first_predecessor = block != scope.blocks.front();
            block->traverse_predecessors(
                false, [&](BasicBlock *predecessor) noexcept {
                    if (!scope_blocks.contains(predecessor)) { return; }
                    auto iter = out_states.find(predecessor);
                    if (iter == out_states.end()) { return; }
                    merge_pointer_state_into_entry(
                        next_in, iter->second, first_predecessor);
                    first_predecessor = false;
                });
            auto old_in = in_states.find(block);
            if (old_in == in_states.end() ||
                !same_pointer_state(old_in->second, next_in)) {
                in_states[block] = next_in;
                changed = true;
            }
            auto state = next_in;
            for (auto *instruction : block->instructions()) {
                if (instruction->derived_instruction_tag() ==
                        DerivedInstructionTag::CORO_SUSPEND ||
                    (instruction->is_terminator() &&
                     instruction->derived_instruction_tag() !=
                         DerivedInstructionTag::CORO_SUSPEND)) {
                    result.killed_at_exit[block] = state.killed;
                    result.touched_at_exit[block] = state.touched;
                }
                transfer_instruction(instruction, state);
            }
            auto old_out = out_states.find(block);
            if (old_out == out_states.end() ||
                !same_pointer_state(old_out->second, state)) {
                out_states[block] = std::move(state);
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    for (auto &[block, state] : out_states) {
        static_cast<void>(block);
        for (auto *value : state.external) { result.external.emplace(value); }
        for (auto *value : state.touched) { result.touched.emplace(value); }
    }
    return result;
}

[[nodiscard]] static ScopeDataflowResult analyze_scope_use_def(
    const CoroCfgDistillResult::Scope &scope) noexcept {
    ScopeDataflowResult result;
    if (scope.blocks.empty()) { return result; }

    auto block_count = scope.blocks.size();
    luisa::unordered_map<BasicBlock *, size_t> block_indices;
    block_indices.reserve(block_count);
    for (size_t i = 0u; i < block_count; ++i) {
        block_indices.emplace(scope.blocks[i], i);
    }

    // Summarize each transfer exactly once. For a block B:
    //
    //   K_out = K_in union K_B
    //   T_out = T_in union T_B
    //   E_out = E_in union (E_B - K_in)
    //
    // K is a must-definition fact (intersection at joins); T and E are may
    // facts (union at joins). This is the same finite dataflow problem as the
    // former pointer-set fixed point, but instruction transfer is no longer
    // re-executed on every iteration.
    luisa::vector<ScopeDataflowState> local_states;
    local_states.reserve(block_count);
    luisa::unordered_map<Value *, size_t> value_indices;
    luisa::vector<Value *> values;
    for (auto *block : scope.blocks) {
        auto &local = local_states.emplace_back();
        for (auto *instruction : block->instructions()) {
            transfer_instruction(instruction, local);
        }
        auto number_values = [&](auto &set) noexcept {
            for (auto *value : set) {
                if (!value_indices.contains(value)) {
                    value_indices.emplace(value, values.size());
                    values.emplace_back(value);
                }
            }
        };
        number_values(local.killed);
        number_values(local.external);
        number_values(local.touched);
    }

    auto value_count = values.size();
    struct DenseBlockTransfer {
        DenseValueSet killed;
        DenseValueSet external;
        DenseValueSet touched;
        explicit DenseBlockTransfer(size_t n) noexcept
            : killed{n}, external{n}, touched{n} {}
    };
    luisa::vector<DenseBlockTransfer> local_transfers;
    local_transfers.reserve(block_count);
    for (auto &local : local_states) {
        auto &dense = local_transfers.emplace_back(value_count);
        for (auto *value : local.killed) {
            dense.killed.set(value_indices.at(value));
        }
        for (auto *value : local.external) {
            dense.external.set(value_indices.at(value));
        }
        for (auto *value : local.touched) {
            dense.touched.set(value_indices.at(value));
        }
    }
    local_states.clear();

    // Number the induced scope CFG once. Successor construction also gives us
    // sparse predecessor lists without repeatedly walking intrusive use lists.
    luisa::vector<luisa::vector<size_t>> successors(block_count);
    luisa::vector<luisa::vector<size_t>> predecessors(block_count);
    for (size_t i = 0u; i < block_count; ++i) {
        scope.blocks[i]->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                auto iter = block_indices.find(successor);
                if (iter == block_indices.end()) { return; }
                auto j = iter->second;
                auto &out = successors[i];
                if (std::find(out.begin(), out.end(), j) == out.end()) {
                    out.emplace_back(j);
                    predecessors[j].emplace_back(i);
                }
            });
    }

    auto solve_worklist = [&](auto &&update) noexcept {
        luisa::deque<size_t> worklist;
        luisa::vector<uint8_t> queued(block_count, 1u);
        for (size_t i = 0u; i < block_count; ++i) {
            worklist.emplace_back(i);
        }
        while (!worklist.empty()) {
            auto block = worklist.front();
            worklist.pop_front();
            queued[block] = 0u;
            if (update(block)) {
                for (auto successor : successors[block]) {
                    if (queued[successor] == 0u) {
                        queued[successor] = 1u;
                        worklist.emplace_back(successor);
                    }
                }
            }
        }
    };

    // Solve must-kill first. This is a forward must analysis, so non-entry
    // states start at TOP and monotonically decrease to the greatest fixed
    // point. Starting at the empty set instead computes the least fixed point:
    // in a loop, a value killed on every path from the entry can then be lost
    // merely because the unvisited back-edge initially contributes BOTTOM.
    // The entry boundary remains the empty set even when a loop targets it.
    // Keeping the final must solution fixed makes the subsequent may-use
    // equations monotone and avoids transient external-value overestimates.
    auto killed_top = DenseValueSet::full(value_count);
    luisa::vector<DenseValueSet> killed_in(block_count, killed_top);
    luisa::vector<DenseValueSet> killed_out(block_count, killed_top);
    killed_in.front() = DenseValueSet{value_count};
    killed_out.front() = local_transfers.front().killed;
    solve_worklist([&](size_t block) noexcept {
        DenseValueSet next_in{value_count};
        if (block != 0u && !predecessors[block].empty()) {
            next_in = killed_out[predecessors[block].front()];
            for (size_t i = 1u; i < predecessors[block].size(); ++i) {
                next_in.intersect_with(
                    killed_out[predecessors[block][i]]);
            }
        }
        auto next_out = next_in;
        next_out.union_with(local_transfers[block].killed);
        if (next_in == killed_in[block] &&
            next_out == killed_out[block]) {
            return false;
        }
        killed_in[block] = std::move(next_in);
        killed_out[block] = std::move(next_out);
        return true;
    });

    luisa::vector<DenseValueSet> external_out(
        block_count, DenseValueSet{value_count});
    luisa::vector<DenseValueSet> touched_out(
        block_count, DenseValueSet{value_count});
    solve_worklist([&](size_t block) noexcept {
        DenseValueSet next_external{value_count};
        DenseValueSet next_touched{value_count};
        for (auto predecessor : predecessors[block]) {
            next_external.union_with(external_out[predecessor]);
            next_touched.union_with(touched_out[predecessor]);
        }
        auto local_external = local_transfers[block].external;
        local_external.subtract(killed_in[block]);
        next_external.union_with(local_external);
        next_touched.union_with(local_transfers[block].touched);
        if (next_external == external_out[block] &&
            next_touched == touched_out[block]) {
            return false;
        }
        external_out[block] = std::move(next_external);
        touched_out[block] = std::move(next_touched);
        return true;
    });

    auto append_dense_values = [&](auto &destination,
                                   const DenseValueSet &source) noexcept {
        for (size_t i = 0u; i < value_count; ++i) {
            if (source.test(i)) { destination.emplace(values[i]); }
        }
    };
    for (size_t i = 0u; i < block_count; ++i) {
        append_dense_values(result.external, external_out[i]);
        append_dense_values(result.touched, touched_out[i]);
        auto *block = scope.blocks[i];
        if (block->is_terminated()) {
            append_dense_values(result.killed_at_exit[block], killed_out[i]);
            append_dense_values(result.touched_at_exit[block], touched_out[i]);
        }
    }
    if (auto *flag = std::getenv("LUISA_CORO_VERIFY_DENSE_DATAFLOW");
        flag != nullptr && luisa::string_view{flag} == "1") {
        auto oracle = analyze_scope_use_def_pointer_oracle(scope);
        auto pointer_set_difference = [](auto &a, auto &b) noexcept {
            luisa::unordered_set<Value *> difference;
            for (auto *value : a) {
                if (!b.contains(value)) { difference.emplace(value); }
            }
            return difference;
        };
        auto dense_only_external = pointer_set_difference(
            result.external, oracle.external);
        auto oracle_only_external = pointer_set_difference(
            oracle.external, result.external);
        auto dense_only_touched = pointer_set_difference(
            result.touched, oracle.touched);
        auto oracle_only_touched = pointer_set_difference(
            oracle.touched, result.touched);
        LUISA_ASSERT(
            dense_only_external.empty() &&
                oracle_only_external.empty() &&
                dense_only_touched.empty() &&
                oracle_only_touched.empty(),
            "Dense coroutine dataflow differs from pointer oracle for scope "
            "token {} (external dense-only={}, oracle-only={}; touched "
            "dense-only={}, oracle-only={}).",
            scope.trigger_token,
            dense_only_external.size(), oracle_only_external.size(),
            dense_only_touched.size(), oracle_only_touched.size());
        for (auto *block : scope.blocks) {
            auto dense_killed = result.killed_at_exit.find(block);
            auto oracle_killed = oracle.killed_at_exit.find(block);
            auto dense_touched = result.touched_at_exit.find(block);
            auto oracle_touched = oracle.touched_at_exit.find(block);
            auto empty = luisa::unordered_set<Value *>{};
            auto &dense_killed_set = dense_killed == result.killed_at_exit.end() ?
                                         empty :
                                         dense_killed->second;
            auto &oracle_killed_set = oracle_killed == oracle.killed_at_exit.end() ?
                                          empty :
                                          oracle_killed->second;
            auto &dense_touched_set = dense_touched == result.touched_at_exit.end() ?
                                          empty :
                                          dense_touched->second;
            auto &oracle_touched_set = oracle_touched == oracle.touched_at_exit.end() ?
                                           empty :
                                           oracle_touched->second;
            LUISA_ASSERT(
                same_set(dense_killed_set, oracle_killed_set) &&
                    same_set(dense_touched_set, oracle_touched_set),
                "Dense coroutine exit dataflow differs from pointer oracle "
                "for scope token {}.",
                scope.trigger_token);
        }
    }
    return result;
}

[[nodiscard]] static luisa::unordered_set<Value *> set_difference(
    const luisa::unordered_set<Value *> &a,
    const luisa::unordered_set<Value *> &b) noexcept {
    luisa::unordered_set<Value *> r;
    for (auto *value : a) {
        if (!b.contains(value)) { r.emplace(value); }
    }
    return r;
}

[[nodiscard]] static luisa::unordered_set<Value *> set_intersection(
    const luisa::unordered_set<Value *> &a,
    const luisa::unordered_set<Value *> &b) noexcept {
    luisa::unordered_set<Value *> r;
    for (auto *value : a) {
        if (b.contains(value)) { r.emplace(value); }
    }
    return r;
}

static void append_set(luisa::unordered_set<Value *> &dst,
                       const luisa::unordered_set<Value *> &src) noexcept {
    for (auto *value : src) { dst.emplace(value); }
}

static void append_names_from_values(luisa::vector<luisa::string> &dst,
                                     const luisa::vector<Value *> &values,
                                     const luisa::unordered_map<Value *, luisa::string> &names) noexcept {
    dst.clear();
    dst.reserve(values.size());
    for (auto *value : values) {
        if (auto it = names.find(value); it != names.end()) {
            dst.emplace_back(it->second);
        }
    }
}

static void analyze_live_variables(CoroCfgDistillResult &result, FunctionDefinition *def) noexcept {
    auto n = result.scopes.size();
    luisa::vector<ScopeDataflowResult> scope_data;
    scope_data.reserve(n);
    for (auto &scope : result.scopes) {
        scope_data.emplace_back(analyze_scope_use_def(scope));
    }

    luisa::unordered_map<uint32_t, size_t> trigger_to_scope;
    for (size_t i = 0u; i < n; ++i) {
        trigger_to_scope.emplace(result.scopes[i].trigger_token, i);
    }

    luisa::vector<luisa::unordered_set<BasicBlock *>> scope_blocks;
    scope_blocks.reserve(n);
    for (auto &scope : result.scopes) {
        auto &set = scope_blocks.emplace_back();
        for (auto *bb : scope.blocks) { set.emplace(bb); }
    }

    luisa::unordered_map<BasicBlock *, size_t> block_to_scope;
    for (size_t i = 0u; i < n; ++i) {
        for (auto *bb : result.scopes[i].blocks) {
            block_to_scope.emplace(bb, i);
        }
    }

    auto append_cross_scope_successor_edges = [&](size_t from, BasicBlock *exit_block, auto visit) noexcept {
        if (exit_block == nullptr || !exit_block->is_terminated()) { return; }
        luisa::unordered_set<size_t> seen_targets;
        exit_block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            if (succ == nullptr || scope_blocks[from].contains(succ)) { return; }
            if (auto it = block_to_scope.find(succ); it != block_to_scope.end() && it->second != from) {
                if (seen_targets.emplace(it->second).second) {
                    visit(it->second);
                }
            }
        });
    };

    result.transition_edges.clear();
    for (size_t from = 0u; from < n; ++from) {
        for (auto &sp : result.scopes[from].suspend_points) {
            auto iter = trigger_to_scope.find(sp.token);
            if (iter == trigger_to_scope.end()) { continue; }
            auto to = iter->second;
            CoroCfgDistillResult::Edge edge;
            edge.from_scope = from;
            edge.to_scope = to;
            edge.token = sp.token;
            edge.exit_block = sp.block;
            edge.is_suspend = true;
            if (auto killed = scope_data[from].killed_at_exit.find(sp.block);
                killed != scope_data[from].killed_at_exit.end()) {
                for (auto *value : killed->second) { edge.killed_values.emplace_back(value); }
            }
            if (auto touched = scope_data[from].touched_at_exit.find(sp.block);
                touched != scope_data[from].touched_at_exit.end()) {
                for (auto *value : touched->second) { edge.touched_values.emplace_back(value); }
            }
            result.transition_edges.emplace_back(std::move(edge));
        }
        for (auto *bb : result.scopes[from].blocks) {
            append_cross_scope_successor_edges(from, bb, [&](size_t to) noexcept {
                CoroCfgDistillResult::Edge edge;
                edge.from_scope = from;
                edge.to_scope = to;
                edge.token = result.scopes[to].trigger_token;
                edge.exit_block = bb;
                if (auto killed = scope_data[from].killed_at_exit.find(bb);
                    killed != scope_data[from].killed_at_exit.end()) {
                    for (auto *value : killed->second) { edge.killed_values.emplace_back(value); }
                }
                if (auto touched = scope_data[from].touched_at_exit.find(bb);
                    touched != scope_data[from].touched_at_exit.end()) {
                    for (auto *value : touched->second) { edge.touched_values.emplace_back(value); }
                }
                result.transition_edges.emplace_back(std::move(edge));
            });
        }
    }

    luisa::vector<luisa::unordered_set<Value *>> live_begin(n);
    for (;;) {
        auto changed = false;
        for (size_t ri = 0u; ri < n; ++ri) {
            auto s = n - 1u - ri;
            auto next = scope_data[s].external;
            for (auto &edge : result.transition_edges) {
                if (edge.from_scope != s || edge.to_scope >= n) { continue; }
                luisa::unordered_set<Value *> killed;
                for (auto *value : edge.killed_values) { killed.emplace(value); }
                auto propagated = set_difference(live_begin[edge.to_scope], killed);
                append_set(next, propagated);
            }
            if (!same_set(live_begin[s], next)) {
                live_begin[s] = std::move(next);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    luisa::vector<luisa::unordered_set<Value *>> live_in(n);
    luisa::vector<luisa::unordered_set<Value *>> live_out(n);
    for (size_t s = 0u; s < n; ++s) {
        live_in[s] = scope_data[s].external;
        for (auto &edge : result.transition_edges) {
            if (edge.from_scope != s || edge.to_scope >= n) { continue; }
            luisa::unordered_set<Value *> killed;
            for (auto *value : edge.killed_values) { killed.emplace(value); }
            auto propagated = set_difference(live_begin[edge.to_scope], killed);
            auto reload = set_intersection(propagated, scope_data[s].touched);
            append_set(live_in[s], reload);
            luisa::unordered_set<Value *> touched;
            for (auto *value : edge.touched_values) { touched.emplace(value); }
            auto store = set_intersection(live_begin[edge.to_scope], touched);
            edge.live_values.clear();
            for (auto *value : live_begin[edge.to_scope]) {
                edge.live_values.emplace_back(value);
            }
            edge.store_values.clear();
            for (auto *value : store) {
                edge.store_values.emplace_back(value);
                live_out[s].emplace(value);
            }
        }
    }

    luisa::unordered_set<Value *> frame_value_set;
    for (size_t i = 0u; i < n; ++i) {
        append_set(frame_value_set, live_begin[i]);
        append_set(frame_value_set, live_in[i]);
        append_set(frame_value_set, live_out[i]);
    }

    luisa::unordered_map<Value *, size_t> order;
    if (def != nullptr) {
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (frame_value_set.contains(inst) && !order.contains(inst)) {
                order.emplace(inst, order.size());
            }
        });
    }

    result.frame_values.clear();
    result.frame_values.reserve(frame_value_set.size());
    luisa::vector<Value *> ordered_frame_values;
    append_ordered_values(ordered_frame_values, frame_value_set, order);
    sort_frame_values_by_layout(ordered_frame_values);
    luisa::unordered_map<Value *, luisa::string> names;
    luisa::unordered_set<luisa::string> used_names;
    for (auto *value : ordered_frame_values) {
        auto name = frame_value_name(value, result.frame_values.size());
        if (!used_names.emplace(name).second) {
            auto base = name;
            auto suffix = result.frame_values.size();
            do {
                name = luisa::format("{}#{}", base, suffix++);
            } while (!used_names.emplace(name).second);
        }
        names.emplace(value, name);
        result.frame_values.emplace_back(CoroCfgDistillResult::FrameValue{
            .value = value,
            .name = std::move(name),
            .type = value->type(),
        });
    }

    for (size_t i = 0u; i < n; ++i) {
        append_ordered_values(result.scopes[i].external_values, scope_data[i].external, order);
        append_ordered_values(result.scopes[i].touched_values, scope_data[i].touched, order);
        append_ordered_values(result.scopes[i].live_in_values, live_in[i], order);
        append_ordered_values(result.scopes[i].live_out_values, live_out[i], order);
        append_names_from_values(result.scopes[i].external_variables, result.scopes[i].external_values, names);
        append_names_from_values(result.scopes[i].touched_variables, result.scopes[i].touched_values, names);
        append_names_from_values(result.scopes[i].live_in_variables, result.scopes[i].live_in_values, names);
        append_names_from_values(result.scopes[i].live_out_variables, result.scopes[i].live_out_values, names);
    }

    for (auto &edge : result.transition_edges) {
        luisa::unordered_set<Value *> killed_set;
        luisa::unordered_set<Value *> touched_set;
        luisa::unordered_set<Value *> live_set;
        luisa::unordered_set<Value *> store_set;
        for (auto *value : edge.killed_values) { killed_set.emplace(value); }
        for (auto *value : edge.touched_values) { touched_set.emplace(value); }
        for (auto *value : edge.live_values) { live_set.emplace(value); }
        for (auto *value : edge.store_values) { store_set.emplace(value); }
        append_ordered_values(edge.killed_values, killed_set, order);
        append_ordered_values(edge.touched_values, touched_set, order);
        append_ordered_values(edge.live_values, live_set, order);
        append_ordered_values(edge.store_values, store_set, order);
        append_names_from_values(edge.killed_variables, edge.killed_values, names);
        append_names_from_values(edge.touched_variables, edge.touched_values, names);
        append_names_from_values(edge.live_variables, edge.live_values, names);
        append_names_from_values(edge.store_variables, edge.store_values, names);
    }
}

[[nodiscard]] static CoroCfgDistillResult distill_function(FunctionDefinition *def) noexcept {

    CoroCfgDistillResult result;
    if (def == nullptr || def->body_block() == nullptr) { return result; }

    luisa::unordered_map<uint32_t, BasicBlock *> token_to_resume;
    luisa::unordered_map<uint32_t, luisa::string> token_to_name;
    for (auto *bb : def->basic_blocks()) {
        for (auto *inst : bb->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_RESUME: {
                    auto *r = static_cast<CoroResumeInst *>(inst);
                    token_to_resume.emplace(r->token(), bb);
                    break;
                }
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(inst);
                    token_to_name.emplace(s->token(), s->name());
                    break;
                }
                default:
                    break;
            }
        }
    }

    auto reachable = collect_coro_reachable_blocks(def, token_to_resume);

    struct Root {
        BasicBlock *block;
        uint32_t token;
        luisa::optional<luisa::string> name;
    };
    luisa::vector<Root> roots;
    roots.emplace_back(Root{def->body_block(), 0u, luisa::nullopt});
    luisa::vector<uint32_t> resume_tokens;
    resume_tokens.reserve(token_to_resume.size());
    for (auto &[token, bb] : token_to_resume) {
        if (!reachable.contains(bb)) { continue; }
        resume_tokens.emplace_back(token);
    }
    std::sort(resume_tokens.begin(), resume_tokens.end());
    for (auto token : resume_tokens) {
        auto name = luisa::optional<luisa::string>{};
        if (auto it = token_to_name.find(token); it != token_to_name.end()) {
            name = it->second;
        }
        roots.emplace_back(Root{token_to_resume.at(token), token, std::move(name)});
    }

    result.scopes.reserve(roots.size());
    for (size_t i = 0u; i < roots.size(); ++i) {
        auto &scope = result.scopes.emplace_back();
        scope.scope_id = static_cast<int>(i);
        scope.trigger_token = roots[i].token;
        scope.trigger_name = roots[i].name;

        luisa::unordered_set<BasicBlock *> visited;
        luisa::deque<BasicBlock *> worklist;
        worklist.emplace_back(roots[i].block);

        while (!worklist.empty()) {
            auto *bb = worklist.front();
            worklist.pop_front();
            if (bb == nullptr || !reachable.contains(bb)) { continue; }
            if (bb != roots[i].block) {
                if (auto resume_token = resume_token_of_block(bb);
                    resume_token.has_value() && *resume_token != roots[i].token) {
                    continue;
                }
            }
            if (!visited.emplace(bb).second) { continue; }
            scope.blocks.emplace_back(bb);
            if (!bb->is_terminated()) { continue; }
            auto *term = bb->terminator();
            switch (term->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(term);
                    scope.suspend_points.emplace_back(
                        CoroCfgDistillResult::Scope::SuspendPoint{bb, s->token(), s->name()});
                    if (!scope.suspend_token.has_value()) {
                        scope.suspend_token = s->token();
                        scope.suspend_name = s->name();
                    }
                    break;
                }
                case DerivedInstructionTag::CORO_TERMINATE:
                    scope.is_terminal = true;
                    break;
                default:
                    bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                        worklist.emplace_back(succ);
                    });
                    break;
            }
        }
    }

    luisa::vector<luisa::unordered_set<BasicBlock *>> scope_blocks;
    scope_blocks.reserve(result.scopes.size());
    for (auto &scope : result.scopes) {
        auto &set = scope_blocks.emplace_back();
        for (auto *bb : scope.blocks) { set.emplace(bb); }
    }

    luisa::unordered_map<uint32_t, size_t> trigger_to_scope;
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        trigger_to_scope.emplace(result.scopes[i].trigger_token, i);
    }

    luisa::vector<luisa::unordered_set<size_t>> edge_sets(result.scopes.size());
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        for (auto &sp : result.scopes[i].suspend_points) {
            if (auto it = trigger_to_scope.find(sp.token); it != trigger_to_scope.end()) {
                edge_sets[i].emplace(it->second);
            }
        }
        for (auto *bb : result.scopes[i].blocks) {
            if (!bb->is_terminated()) { continue; }
            bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                if (scope_blocks[i].contains(succ)) { return; }
                for (size_t j = 0u; j < scope_blocks.size(); ++j) {
                    if (j != i && scope_blocks[j].contains(succ)) {
                        edge_sets[i].emplace(j);
                    }
                }
            });
        }
    }

    result.edges.resize(edge_sets.size());
    for (size_t i = 0u; i < edge_sets.size(); ++i) {
        result.edges[i].assign(edge_sets[i].begin(), edge_sets[i].end());
        std::sort(result.edges[i].begin(), result.edges[i].end());
    }

    analyze_live_variables(result, def);
    return result;
}

}// namespace detail

void CoroCfgDistillResult::_seal(FunctionDefinition *definition) noexcept {
    _source_definition = definition;
    _validation_hash =
        detail::compute_distill_validation_hash(*this, definition);
}

bool CoroCfgDistillResult::validation_certificate_matches(
    const FunctionDefinition *definition) const noexcept {
    return definition != nullptr &&
           definition == _source_definition &&
           _validation_hash ==
               detail::compute_distill_validation_hash(*this, definition);
}

CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(Function *f) noexcept {
    if (f == nullptr) {
        CoroCfgDistillResult result;
        result.invalid_input_error_count = 1u;
        return result;
    }
    auto *def = f->definition();
    if (def == nullptr || def->body_block() == nullptr) {
        CoroCfgDistillResult result;
        result.invalid_input_error_count = 1u;
        return result;
    }
    auto verification = xir_verify_function(f);
    if (!verification.succeeded()) {
        CoroCfgDistillResult result;
        result.invalid_cfg_error_count = 1u;
        return result;
    }
    CoroCfgDistillResult result;
    result.structured_cfg_error_count =
        contains_structured_control_flow(def) ? 1u : 0u;
    result.invalid_cfg_error_count =
        detail::validate_coroutine_input_language(def) &&
                detail::coro_cfg_distill_validate_coroutine_tokens(def) ?
            0u :
            1u;
    if (!result.succeeded()) {
        return result;
    }
    result = detail::distill_function(def);
    result._seal(def);
    return result;
}

size_t coro_cfg_distill_pass_run_on_module(Module *m) noexcept {
    if (m == nullptr) { return 0u; }
    size_t count = 0u;
    for (auto *f : m->function_list()) {
        if (f->is_definition()) {
            auto result = coro_cfg_distill_pass_run_on_function(f);
            count += result.succeeded() ? 1u : 0u;
        }
    }
    return count;
}

}// namespace luisa::compute::xir
