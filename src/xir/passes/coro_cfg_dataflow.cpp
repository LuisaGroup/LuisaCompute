#include "coro_cfg_dataflow.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/deque.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "../pointer_containers.h"
#include "coro_replayable.h"

namespace luisa::compute::xir::detail {

namespace {

template<typename T>
[[nodiscard]] bool same_set(const luisa::unordered_set<T> &a,
                            const luisa::unordered_set<T> &b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (auto &v : a) {
        if (!b.contains(v)) { return false; }
    }
    return true;
}

[[nodiscard]] bool is_always_available(Value *value) noexcept {
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

struct PointerScopeDataflowState {
    const DenseValueDomain *domain{nullptr};
    CoroReplayableValueAnalysis *replayable{nullptr};
    luisa::unordered_set<size_t> killed;
    luisa::unordered_set<size_t> external;
    luisa::unordered_set<size_t> touched;

    PointerScopeDataflowState() noexcept = default;

    explicit PointerScopeDataflowState(
        const DenseValueDomain &value_domain,
        CoroReplayableValueAnalysis &analysis) noexcept
        : domain{&value_domain}, replayable{&analysis} {}

    void kill(size_t index) noexcept { killed.emplace(index); }
    void expose(size_t index) noexcept { external.emplace(index); }
    void touch(size_t index) noexcept { touched.emplace(index); }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return killed.contains(index);
    }
};

struct SparseBlockEffect {
    luisa::vector<size_t> killed;
    luisa::vector<size_t> external;
    luisa::vector<size_t> touched;
};

// Block effects are sparse while the scope projection is being discovered.
// Three bits per global atom give O(1) membership without a hash allocation
// for every block. Only coordinates recorded in the effect are cleared before
// the marker array is reused by the next block.
struct SparseBlockTransferState {
    static constexpr auto killed_bit = uint8_t{1u << 0u};
    static constexpr auto external_bit = uint8_t{1u << 1u};
    static constexpr auto touched_bit = uint8_t{1u << 2u};

    const DenseValueDomain *domain{nullptr};
    CoroReplayableValueAnalysis *replayable{nullptr};
    luisa::span<uint8_t> marks;
    SparseBlockEffect *effect{nullptr};

    void record(size_t index, uint8_t bit,
                luisa::vector<size_t> &indices) noexcept {
        if ((marks[index] & bit) == 0u) {
            marks[index] |= bit;
            indices.emplace_back(index);
        }
    }
    void kill(size_t index) noexcept {
        record(index, killed_bit, effect->killed);
    }
    void expose(size_t index) noexcept {
        record(index, external_bit, effect->external);
    }
    void touch(size_t index) noexcept {
        record(index, touched_bit, effect->touched);
    }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return (marks[index] & killed_bit) != 0u;
    }
    void reset_marks() noexcept {
        for (auto index : effect->killed) { marks[index] = 0u; }
        for (auto index : effect->external) { marks[index] = 0u; }
        for (auto index : effect->touched) { marks[index] = 0u; }
    }
};

struct DenseBlockEffect {
    DenseValueSet killed;
    DenseValueSet external;
    DenseValueSet touched;

    explicit DenseBlockEffect(size_t value_count) noexcept
        : killed{value_count},
          external{value_count},
          touched{value_count} {}
};

template<typename State>
void touch_index(size_t index, State &state) noexcept {
    state.kill(index);
    state.touch(index);
}

template<typename State>
void use_index(size_t index, State &state) noexcept {
    if (!state.is_killed(index)) { state.expose(index); }
}

template<typename State>
void use_value(Value *value, State &state) noexcept {
    if (is_always_available(value)) { return; }
    if (state.replayable->detect(value)) { return; }
    if (auto index = state.domain->ssa_index(value)) {
        use_index(*index, state);
        return;
    }
    for (auto access : state.domain->memory_accesses(value)) {
        use_index(access.atom_index, state);
    }
}

template<typename State>
void touch_value(Value *value, State &state) noexcept {
    if (value == nullptr) { return; }
    if (auto index = state.domain->ssa_index(value)) {
        touch_index(*index, state);
    }
}

template<typename State>
void use_memory(Value *pointer, State &state) noexcept {
    for (auto access : state.domain->memory_accesses(pointer)) {
        use_index(access.atom_index, state);
    }
}

template<typename State>
void touch_memory(Value *pointer, State &state,
                  bool definite) noexcept {
    for (auto access : state.domain->memory_accesses(pointer)) {
        if (definite) {
            if (access.covers_atom) {
                state.kill(access.atom_index);
            } else {
                // A partial store preserves the bytes outside its path.
                use_index(access.atom_index, state);
            }
        }
        state.touch(access.atom_index);
    }
}

template<typename State>
void begin_memory_lifetime(Value *pointer, State &state) noexcept {
    // ALLOCA creates undefined storage and therefore kills every atom rooted
    // at the local, but it does not create a value that must be stored.
    for (auto access : state.domain->memory_accesses(pointer)) {
        state.kill(access.atom_index);
    }
}

template<typename State>
void use_pointer_indices(Value *value, State &state) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        if (inst->derived_instruction_tag() !=
            DerivedInstructionTag::GEP) {
            break;
        }
        auto *gep = static_cast<GEPInst *>(inst);
        for (size_t i = 0u; i < gep->index_count(); ++i) {
            use_value(gep->index(i), state);
        }
        value = gep->base();
    }
}

template<typename State>
void transfer_call_instruction(CallInst *call, State &state) noexcept {
    auto arg_iter = call->callee()->arguments().begin();
    for (auto *arg_use : call->argument_uses()) {
        auto *argument = arg_use->value();
        if (arg_iter != call->callee()->arguments().end() &&
            (*arg_iter)->is_reference()) {
            use_pointer_indices(argument, state);
            if (trace_local_alloca(argument) != nullptr) {
                use_memory(argument, state);
                touch_memory(argument, state, false);
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

template<typename State>
void transfer_instruction(Instruction *inst, State &state) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA: {
            auto *alloca = static_cast<AllocaInst *>(inst);
            if (alloca->is_local()) {
                begin_memory_lifetime(alloca, state);
            }
            break;
        }
        case DerivedInstructionTag::GEP:
            use_pointer_indices(inst, state);
            break;
        case DerivedInstructionTag::CALL:
            transfer_call_instruction(static_cast<CallInst *>(inst), state);
            break;
        case DerivedInstructionTag::LOAD: {
            auto *load = static_cast<LoadInst *>(inst);
            use_pointer_indices(load->variable(), state);
            if (trace_local_alloca(load->variable()) != nullptr) {
                use_memory(load->variable(), state);
            } else {
                use_value(load->variable(), state);
            }
            touch_value(inst, state);
            break;
        }
        case DerivedInstructionTag::STORE: {
            auto *store = static_cast<StoreInst *>(inst);
            use_value(store->value(), state);
            use_pointer_indices(store->variable(), state);
            if (trace_local_alloca(store->variable()) != nullptr) {
                touch_memory(store->variable(), state, true);
            } else {
                use_value(store->variable(), state);
            }
            break;
        }
        case DerivedInstructionTag::ATOMIC: {
            auto *atomic = static_cast<AtomicInst *>(inst);
            for (auto *index : atomic->index_uses()) {
                use_value(index->value(), state);
            }
            for (auto *value : atomic->value_uses()) {
                use_value(value->value(), state);
            }
            if (trace_local_alloca(atomic->base()) != nullptr) {
                use_memory(atomic->base(), state);
                touch_memory(atomic->base(), state, false);
            } else {
                use_value(atomic->base(), state);
            }
            if (atomic->type() != nullptr && !atomic->is_lvalue()) {
                touch_value(atomic, state);
            }
            break;
        }
        case DerivedInstructionTag::CORO_SUSPEND: {
            auto *suspend = static_cast<CoroSuspendInst *>(inst);
            for (size_t i = 0u;
                 i < suspend->frame_export_count(); ++i) {
                use_value(suspend->frame_export_value(i), state);
            }
            // Extension operands participate in the same use/def domain as
            // ordinary coroutine state. Reads consume the pre-suspend value;
            // writes are definitions performed outside this source scope and
            // are therefore attached to the transition by CFG distillation,
            // not manufactured here as reads of an uninitialized lvalue.
            for (auto &&extension : suspend->extensions()) {
                for (auto &&binding : extension->bindings()) {
                    auto *value = suspend->extension_binding_value(
                        binding.index);
                    switch (binding.access) {
                        case CoroSuspendBindingAccess::read:
                            use_value(value, state);
                            break;
                        case CoroSuspendBindingAccess::read_write:
                            use_pointer_indices(value, state);
                            if (trace_local_alloca(value) != nullptr) {
                                use_memory(value, state);
                            } else {
                                use_value(value, state);
                            }
                            break;
                        case CoroSuspendBindingAccess::write:
                            use_pointer_indices(value, state);
                            break;
                    }
                }
            }
            break;
        }
        case DerivedInstructionTag::CORO_TERMINATE:
            break;
        default: {
            for (auto *op_use : inst->operand_uses()) {
                use_value(op_use->value(), state);
            }
            if (inst->type() != nullptr && !inst->is_lvalue() &&
                !inst->is_terminator()) {
                touch_value(inst, state);
            }
            break;
        }
    }
}

struct PointerScopeDataflowResult {
    luisa::unordered_set<size_t> external;
    luisa::unordered_set<size_t> touched;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<size_t>>
        killed_at_exit;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<size_t>>
        touched_at_exit;
};

[[nodiscard]] bool same_pointer_state(
    const PointerScopeDataflowState &a,
    const PointerScopeDataflowState &b) noexcept {
    return same_set(a.killed, b.killed) &&
           same_set(a.external, b.external) &&
           same_set(a.touched, b.touched);
}

void merge_pointer_state_into_entry(
    PointerScopeDataflowState &dst,
    const PointerScopeDataflowState &src,
    bool first_predecessor) noexcept {
    for (auto value : src.external) { dst.external.emplace(value); }
    for (auto value : src.touched) { dst.touched.emplace(value); }
    if (first_predecessor) {
        dst.killed = src.killed;
    } else {
        luisa::unordered_set<size_t> killed;
        for (auto value : dst.killed) {
            if (src.killed.contains(value)) { killed.emplace(value); }
        }
        dst.killed = std::move(killed);
    }
}

[[nodiscard]] PointerScopeDataflowResult
analyze_scope_use_def_pointer_oracle(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    CoroReplayableValueAnalysis &replayable) noexcept {
    PointerScopeDataflowResult result;
    if (scope.blocks.empty()) { return result; }
    luisa::unordered_set<BasicBlock *> scope_blocks;
    for (auto *block : scope.blocks) { scope_blocks.emplace(block); }
    luisa::unordered_map<BasicBlock *, PointerScopeDataflowState> in_states;
    luisa::unordered_map<BasicBlock *, PointerScopeDataflowState> out_states;
    for (;;) {
        auto changed = false;
        for (auto *block : scope.blocks) {
            PointerScopeDataflowState next_in{value_domain, replayable};
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
        for (auto value : state.external) { result.external.emplace(value); }
        for (auto value : state.touched) { result.touched.emplace(value); }
    }
    return result;
}

struct ProjectedBlockEffects {
    luisa::vector<size_t> local_to_global;
    luisa::vector<DenseBlockEffect> blocks;
};

[[nodiscard]] ProjectedBlockEffects collect_projected_block_effects(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    CoroReplayableValueAnalysis &replayable) noexcept {
    auto global_count = value_domain.size();
    luisa::vector<SparseBlockEffect> sparse_effects;
    sparse_effects.reserve(scope.blocks.size());
    luisa::vector<uint8_t> transfer_marks(global_count, 0u);
    luisa::vector<uint8_t> active(global_count, 0u);
    for (auto *block : scope.blocks) {
        auto &effect = sparse_effects.emplace_back();
        SparseBlockTransferState state{
            .domain = &value_domain,
            .replayable = &replayable,
            .marks = transfer_marks,
            .effect = &effect};
        for (auto *instruction : block->instructions()) {
            transfer_instruction(instruction, state);
        }
        for (auto index : effect.killed) { active[index] = 1u; }
        for (auto index : effect.external) { active[index] = 1u; }
        for (auto index : effect.touched) { active[index] = 1u; }
        state.reset_marks();
    }

    ProjectedBlockEffects result;
    result.local_to_global.reserve(global_count);
    luisa::vector<size_t> global_to_local(
        global_count, static_cast<size_t>(-1));
    for (size_t global = 0u; global < global_count; ++global) {
        if (active[global] == 0u) { continue; }
        global_to_local[global] = result.local_to_global.size();
        result.local_to_global.emplace_back(global);
    }
    auto local_count = result.local_to_global.size();
    result.blocks.reserve(sparse_effects.size());
    for (auto &sparse : sparse_effects) {
        auto &dense = result.blocks.emplace_back(local_count);
        auto project = [&](const luisa::vector<size_t> &source,
                           DenseValueSet &destination) noexcept {
            for (auto global : source) {
                auto local = global_to_local[global];
                LUISA_DEBUG_ASSERT(
                    local != static_cast<size_t>(-1),
                    "Active coroutine atom is missing from its projection.");
                destination.set(local);
            }
        };
        project(sparse.killed, dense.killed);
        project(sparse.external, dense.external);
        project(sparse.touched, dense.touched);
    }
    return result;
}

}// namespace

DenseValueSet::DenseValueSet(size_t bit_count) noexcept
    : _bit_count{bit_count},
      _words((bit_count + 63u) / 64u, 0u) {}

DenseValueSet DenseValueSet::full(size_t bit_count) noexcept {
    DenseValueSet result{bit_count};
    std::fill(result._words.begin(), result._words.end(), ~uint64_t{0u});
    if (auto tail_bit_count = bit_count % 64u;
        tail_bit_count != 0u) {
        result._words.back() &=
            (uint64_t{1u} << tail_bit_count) - uint64_t{1u};
    }
    return result;
}

void DenseValueSet::set(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _bit_count,
                       "Dense value index is out of range.");
    _words[index / 64u] |= uint64_t{1u} << (index % 64u);
}

bool DenseValueSet::test(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < _bit_count,
                       "Dense value index is out of range.");
    return (_words[index / 64u] &
            (uint64_t{1u} << (index % 64u))) != 0u;
}

void DenseValueSet::union_with(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    for (size_t i = 0u; i < _words.size(); ++i) {
        _words[i] |= other._words[i];
    }
}

void DenseValueSet::intersect_with(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    for (size_t i = 0u; i < _words.size(); ++i) {
        _words[i] &= other._words[i];
    }
}

void DenseValueSet::subtract(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    for (size_t i = 0u; i < _words.size(); ++i) {
        _words[i] &= ~other._words[i];
    }
}

bool DenseValueSet::operator==(const DenseValueSet &other) const noexcept {
    return _bit_count == other._bit_count && _words == other._words;
}

size_t DenseValueSet::count_size() const noexcept {
    auto count = size_t{0u};
    for (auto word : _words) {
        count += static_cast<size_t>(std::popcount(word));
    }
    return count;
}

DenseValueDomain::DenseValueDomain(
    FunctionDefinition *definition,
    luisa::span<Value *const> designated_values) noexcept
    : _atoms{definition, designated_values} {}

luisa::optional<size_t> DenseValueDomain::ssa_index(
    Value *value) const noexcept {
    return _atoms.ssa_index(value);
}

luisa::span<const CoroFrameAtomDomain::MemoryAccess>
DenseValueDomain::memory_accesses(Value *pointer) const noexcept {
    return _atoms.memory_accesses(pointer);
}

const CoroFrameAtomDomain::Atom &DenseValueDomain::atom(
    size_t index) const noexcept {
    return _atoms.atom(index);
}

size_t DenseValueDomain::split_alloca_count() const noexcept {
    return _atoms.split_alloca_count();
}

size_t DenseValueDomain::split_atom_count() const noexcept {
    return _atoms.split_atom_count();
}

void DenseValueDomain::append_indices(
    luisa::vector<size_t> &destination,
    const DenseValueSet &source) const noexcept {
    LUISA_DEBUG_ASSERT(source.bit_count() == _atoms.size(),
                       "Coroutine atom sets use incompatible domains.");
    destination.clear();
    destination.reserve(source.count_size());
    source.for_each_set_bit([&](size_t i) noexcept {
        destination.emplace_back(i);
    });
}

DenseScopeDataflowResult::DenseScopeDataflowResult(
    size_t block_count, size_t global_count,
    luisa::vector<size_t> projection) noexcept
    : global_value_count{global_count},
      local_to_global{std::move(projection)},
      external{local_to_global.size()},
      touched{local_to_global.size()},
      killed_at_exit(block_count,
                     DenseValueSet{local_to_global.size()}),
      touched_at_exit(block_count,
                      DenseValueSet{local_to_global.size()}) {}

DenseValueSet DenseScopeDataflowResult::expand_to_global(
    const DenseValueSet &source) const noexcept {
    LUISA_DEBUG_ASSERT(source.bit_count() == local_to_global.size(),
                       "Scope set does not use its projected atom domain.");
    DenseValueSet result{global_value_count};
    source.for_each_set_bit([&](size_t local) noexcept {
        LUISA_DEBUG_ASSERT(local < local_to_global.size(),
                           "Projected coroutine atom is out of range.");
        result.set(local_to_global[local]);
    });
    return result;
}

AllocaInst *trace_local_alloca(Value *value) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *inst = static_cast<Instruction *>(value);
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ALLOCA: {
                auto *alloca = static_cast<AllocaInst *>(inst);
                return alloca->is_local() ? alloca : nullptr;
            }
            case DerivedInstructionTag::GEP:
                value = static_cast<GEPInst *>(inst)->base();
                break;
            default:
                return nullptr;
        }
    }
    return nullptr;
}

DenseScopeDataflowResult analyze_scope_use_def(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    CoroReplayableValueAnalysis &replayable) noexcept {
    auto block_count = scope.blocks.size();
    auto projected = collect_projected_block_effects(
        scope, value_domain, replayable);
    DenseScopeDataflowResult result{
        block_count, value_domain.size(),
        std::move(projected.local_to_global)};
    if (scope.blocks.empty()) { return result; }
    auto value_count = result.local_value_count();
    auto local_transfers = std::move(projected.blocks);

    DensePointerMap<BasicBlock *, size_t> block_indices;
    block_indices.reserve(block_count);
    for (size_t i = 0u; i < block_count; ++i) {
        block_indices.emplace(scope.blocks[i], i);
    }

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

    // A depth-first reverse postorder is the natural evaluation order for a
    // forward monotone problem: every acyclic predecessor is evaluated before
    // its successor, while loop backedges alone can cause revisits. Scope
    // construction guarantees root reachability, but the outer loop keeps the
    // analysis total if an independently constructed scope ever contains a
    // disconnected component. Ordering changes no equation or lattice fact.
    luisa::vector<size_t> reverse_postorder;
    reverse_postorder.reserve(block_count);
    luisa::vector<uint8_t> visited(block_count, 0u);
    struct DfsFrame {
        size_t block;
        size_t next_successor;
    };
    auto visit_component = [&](size_t root) noexcept {
        luisa::vector<DfsFrame> stack;
        visited[root] = 1u;
        stack.emplace_back(DfsFrame{root, 0u});
        while (!stack.empty()) {
            auto &frame = stack.back();
            if (frame.next_successor <
                successors[frame.block].size()) {
                auto successor =
                    successors[frame.block][frame.next_successor++];
                if (visited[successor] == 0u) {
                    visited[successor] = 1u;
                    stack.emplace_back(DfsFrame{successor, 0u});
                }
            } else {
                reverse_postorder.emplace_back(frame.block);
                stack.pop_back();
            }
        }
    };
    visit_component(0u);
    for (size_t i = 0u; i < block_count; ++i) {
        if (visited[i] == 0u) { visit_component(i); }
    }
    std::reverse(reverse_postorder.begin(),
                 reverse_postorder.end());

    auto solve_worklist = [&](size_t &evaluation_count,
                              auto &&update) noexcept {
        luisa::deque<size_t> worklist;
        luisa::vector<uint8_t> queued(block_count, 1u);
        for (auto block : reverse_postorder) {
            worklist.emplace_back(block);
        }
        while (!worklist.empty()) {
            auto block = worklist.front();
            worklist.pop_front();
            queued[block] = 0u;
            ++evaluation_count;
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

    // Formal projection: let U_s be the union of every block-local kill,
    // touch, and exposed-use generator in this scope. The transfer equations
    // and their union/intersection joins are coordinate-wise. For any atom
    // outside U_s all generators are zero; every scope block is reachable
    // from scope.blocks.front(), so the empty entry boundary forces its must
    // solution to zero, while both may solutions remain zero. Solving on U_s
    // and extending absent coordinates with zero is therefore exactly the
    // same fixed point as solving over the complete global atom product.
    auto killed_top = DenseValueSet::full(value_count);
    luisa::vector<DenseValueSet> killed_in(block_count, killed_top);
    luisa::vector<DenseValueSet> killed_out(block_count, killed_top);
    killed_in.front() = DenseValueSet{value_count};
    killed_out.front() = local_transfers.front().killed;
    solve_worklist(result.must_block_evaluations,
                   [&](size_t block) noexcept {
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
    solve_worklist(result.may_block_evaluations,
                   [&](size_t block) noexcept {
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

    for (size_t i = 0u; i < block_count; ++i) {
        result.external.union_with(external_out[i]);
        result.touched.union_with(touched_out[i]);
        result.killed_at_exit[i] = killed_out[i];
        result.touched_at_exit[i] = touched_out[i];
    }

    if (auto *flag = std::getenv("LUISA_CORO_VERIFY_DENSE_DATAFLOW");
        flag != nullptr && luisa::string_view{flag} == "1") {
        auto oracle = analyze_scope_use_def_pointer_oracle(
            scope, value_domain, replayable);
        auto to_pointer_set = [](const DenseValueSet &dense) noexcept {
            luisa::unordered_set<size_t> indices;
            dense.for_each_set_bit([&](size_t index) noexcept {
                indices.emplace(index);
            });
            return indices;
        };
        auto pointer_set_difference = [](auto &a, auto &b) noexcept {
            luisa::unordered_set<size_t> difference;
            for (auto value : a) {
                if (!b.contains(value)) { difference.emplace(value); }
            }
            return difference;
        };
        auto dense_external = to_pointer_set(
            result.expand_to_global(result.external));
        auto dense_touched = to_pointer_set(
            result.expand_to_global(result.touched));
        auto dense_only_external = pointer_set_difference(
            dense_external, oracle.external);
        auto oracle_only_external = pointer_set_difference(
            oracle.external, dense_external);
        auto dense_only_touched = pointer_set_difference(
            dense_touched, oracle.touched);
        auto oracle_only_touched = pointer_set_difference(
            oracle.touched, dense_touched);
        LUISA_ASSERT(
            dense_only_external.empty() &&
                oracle_only_external.empty() &&
                dense_only_touched.empty() &&
                oracle_only_touched.empty(),
            "Projected dense coroutine dataflow differs from the pointer "
            "oracle for scope token {} (external dense-only={}, "
            "oracle-only={}; touched dense-only={}, oracle-only={}).",
            scope.trigger_token,
            dense_only_external.size(), oracle_only_external.size(),
            dense_only_touched.size(), oracle_only_touched.size());
        for (size_t block_index = 0u;
             block_index < scope.blocks.size(); ++block_index) {
            auto *block = scope.blocks[block_index];
            if (!block->is_terminated()) { continue; }
            auto dense_killed = to_pointer_set(
                result.expand_to_global(
                    result.killed_at_exit[block_index]));
            auto dense_touched_at_exit = to_pointer_set(
                result.expand_to_global(
                    result.touched_at_exit[block_index]));
            auto oracle_killed = oracle.killed_at_exit.find(block);
            auto oracle_touched = oracle.touched_at_exit.find(block);
            auto empty = luisa::unordered_set<size_t>{};
            auto &oracle_killed_set =
                oracle_killed == oracle.killed_at_exit.end() ?
                    empty :
                    oracle_killed->second;
            auto &oracle_touched_set =
                oracle_touched == oracle.touched_at_exit.end() ?
                    empty :
                    oracle_touched->second;
            LUISA_ASSERT(
                same_set(dense_killed, oracle_killed_set) &&
                    same_set(dense_touched_at_exit,
                             oracle_touched_set),
                "Projected dense coroutine exit dataflow differs from the "
                "pointer oracle for scope token {}.",
                scope.trigger_token);
        }
    }
    return result;
}

void append_legacy_values(
    luisa::vector<Value *> &dst, const DenseValueSet &atoms,
    const DenseValueDomain &domain) noexcept {
    LUISA_DEBUG_ASSERT(atoms.bit_count() == domain.size(),
                       "Legacy coroutine values require the global domain.");
    dst.clear();
    luisa::unordered_set<Value *> seen;
    atoms.for_each_set_bit([&](size_t atom_index) noexcept {
        auto *root = domain.atom(atom_index).root;
        if (root != nullptr && seen.emplace(root).second) {
            dst.emplace_back(root);
        }
    });
}

void append_frame_value_indices(
    luisa::vector<size_t> &dst, const DenseValueSet &atoms,
    luisa::span<const std::pair<size_t, size_t>>
        atom_to_frame_value_range) noexcept {
    LUISA_DEBUG_ASSERT(
        atoms.bit_count() == atom_to_frame_value_range.size(),
        "Frame-value conversion requires the global atom domain.");
    dst.clear();
    atoms.for_each_set_bit([&](size_t atom_index) noexcept {
        auto [first, count] = atom_to_frame_value_range[atom_index];
        if (first != static_cast<size_t>(-1)) {
            for (size_t i = 0u; i < count; ++i) {
                dst.emplace_back(first + i);
            }
        }
    });
}

}// namespace luisa::compute::xir::detail
