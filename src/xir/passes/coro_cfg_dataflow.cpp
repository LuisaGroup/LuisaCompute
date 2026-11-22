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
#include "coro_semantic_graph.h"
#include <luisa/core/stl/map.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/indexed_branch.h>

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
    luisa::unordered_set<size_t> external_defined;
    luisa::unordered_set<size_t> external;
    luisa::unordered_set<size_t> touched;

    PointerScopeDataflowState() noexcept = default;

    explicit PointerScopeDataflowState(
        const DenseValueDomain &value_domain,
        CoroReplayableValueAnalysis &analysis) noexcept
        : domain{&value_domain}, replayable{&analysis} {}

    void kill(size_t index) noexcept { killed.emplace(index); }
    void define_external(size_t index) noexcept {
        external_defined.emplace(index);
    }
    void expose(size_t index) noexcept { external.emplace(index); }
    void touch(size_t index) noexcept { touched.emplace(index); }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return killed.contains(index) ||
               external_defined.contains(index);
    }
};

struct SparseBlockEffect {
    luisa::vector<size_t> killed;
    // Temporary version barriers inside one ordered Extension sequence. They
    // suppress later exposed uses but are not source-code kill generators.
    luisa::vector<size_t> external_defined;
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
    static constexpr auto external_defined_bit = uint8_t{1u << 3u};

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
    void define_external(size_t index) noexcept {
        record(index, external_defined_bit,
               effect->external_defined);
    }
    void expose(size_t index) noexcept {
        record(index, external_bit, effect->external);
    }
    void touch(size_t index) noexcept {
        record(index, touched_bit, effect->touched);
    }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return (marks[index] &
                (killed_bit | external_defined_bit)) != 0u;
    }
    void reset_marks() noexcept {
        for (auto index : effect->killed) { marks[index] = 0u; }
        for (auto index : effect->external) { marks[index] = 0u; }
        for (auto index : effect->touched) { marks[index] = 0u; }
        for (auto index : effect->external_defined) {
            marks[index] = 0u;
        }
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
void define_memory(Value *pointer, State &state) noexcept {
    // An external-stage write is a must-definition of every logical atom in
    // the binding projection. It changes the value version seen by later
    // extensions, but it is not a source-continuation touch: the host writes
    // the backing frame only after this kernel has suspended.
    for (auto access : state.domain->memory_accesses(pointer)) {
        state.define_external(access.atom_index);
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
                // All uses of one Extension observe its input version; all
                // must-defs become visible together to the next Extension.
                // This is the same ordered transfer later certified by CFG
                // distillation, applied here only to exposed-use discovery.
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
                for (auto &&binding : extension->bindings()) {
                    if (binding.lifetime ==
                            CoroSuspendBindingLifetime::boundary ||
                        binding.access ==
                            CoroSuspendBindingAccess::read) {
                        continue;
                    }
                    define_memory(
                        suspend->extension_binding_value(binding.index),
                        state);
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
           same_set(a.external_defined, b.external_defined) &&
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
                    if (!scope.allows_successor(predecessor, block)) { return; }
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

namespace {

[[nodiscard]] const luisa::shared_ptr<DenseValueSet::Page> &
shared_all_ones_page() noexcept {
    static const auto page = []() noexcept {
        auto p = luisa::make_shared<DenseValueSet::Page>();
        p->words.fill(~uint64_t{0u});
        return p;
    }();
    return page;
}

[[nodiscard]] luisa::shared_ptr<DenseValueSet::Page>
ensure_unique_page(luisa::shared_ptr<DenseValueSet::Page> &page) noexcept {
    if (page.use_count() != 1) {
        page = luisa::make_shared<DenseValueSet::Page>(*page);
    }
    return page;
}

}// namespace

DenseValueSet::DenseValueSet(size_t bit_count) noexcept
    : _bit_count{bit_count} {}

DenseValueSet DenseValueSet::full(size_t bit_count) noexcept {
    DenseValueSet result{bit_count};
    if (bit_count == 0u) { return result; }
    const auto page_count = (bit_count + bits_per_page - 1u) / bits_per_page;
    result._pages.reserve(page_count);
    const auto &ones = shared_all_ones_page();
    for (size_t p = 0u; p + 1u < page_count; ++p) {
        result._pages.emplace_back(
            PageEntry{static_cast<uint32_t>(p), ones});
    }
    const auto last_page = page_count - 1u;
    const auto rem_bits = bit_count - last_page * bits_per_page;
    if (rem_bits == bits_per_page) {
        result._pages.emplace_back(
            PageEntry{static_cast<uint32_t>(last_page), ones});
    } else {
        auto tail = luisa::make_shared<Page>();
        const auto full_words = rem_bits / 64u;
        for (size_t w = 0u; w < full_words; ++w) {
            tail->words[w] = ~uint64_t{0u};
        }
        if (const auto tail_bits = rem_bits % 64u; tail_bits != 0u) {
            tail->words[full_words] =
                (uint64_t{1u} << tail_bits) - uint64_t{1u};
        }
        result._pages.emplace_back(
            PageEntry{static_cast<uint32_t>(last_page), std::move(tail)});
    }
    return result;
}

void DenseValueSet::set(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _bit_count,
                       "Dense value index is out of range.");
    const auto pi = static_cast<uint32_t>(index / bits_per_page);
    const auto wi = (index % bits_per_page) / 64u;
    const auto mask = uint64_t{1u} << (index % 64u);
    auto it = std::lower_bound(
        _pages.begin(), _pages.end(), pi,
        [](const PageEntry &e, uint32_t key) noexcept {
            return e.page_index < key;
        });
    if (it != _pages.end() && it->page_index == pi) {
        if ((it->page->words[wi] & mask) != 0u) { return; }
        auto p = ensure_unique_page(it->page);
        p->words[wi] |= mask;
    } else {
        auto p = luisa::make_shared<Page>();
        p->words[wi] = mask;
        _pages.emplace(it, PageEntry{pi, std::move(p)});
    }
}

void DenseValueSet::clear(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _bit_count,
                       "Dense value index is out of range.");
    const auto pi = static_cast<uint32_t>(index / bits_per_page);
    const auto wi = (index % bits_per_page) / 64u;
    const auto mask = uint64_t{1u} << (index % 64u);
    auto it = std::lower_bound(
        _pages.begin(), _pages.end(), pi,
        [](const PageEntry &e, uint32_t key) noexcept {
            return e.page_index < key;
        });
    if (it == _pages.end() || it->page_index != pi ||
        (it->page->words[wi] & mask) == 0u) {
        return;
    }
    auto remaining_any = (it->page->words[wi] & ~mask) != 0u;
    if (!remaining_any) {
        for (size_t w = 0u; w < words_per_page; ++w) {
            if (w != wi && it->page->words[w] != 0u) {
                remaining_any = true;
                break;
            }
        }
    }
    if (!remaining_any) {
        _pages.erase(it);
    } else {
        auto p = ensure_unique_page(it->page);
        p->words[wi] &= ~mask;
    }
}

bool DenseValueSet::test(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < _bit_count,
                       "Dense value index is out of range.");
    const auto pi = static_cast<uint32_t>(index / bits_per_page);
    const auto wi = (index % bits_per_page) / 64u;
    const auto mask = uint64_t{1u} << (index % 64u);
    auto it = std::lower_bound(
        _pages.begin(), _pages.end(), pi,
        [](const PageEntry &e, uint32_t key) noexcept {
            return e.page_index < key;
        });
    return it != _pages.end() && it->page_index == pi &&
           (it->page->words[wi] & mask) != 0u;
}

void DenseValueSet::union_with(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    if (this == &other || other._pages.empty()) { return; }
    if (_pages.empty()) {
        _pages = other._pages;
        return;
    }
    // Fast scan when no new page_index entries need to be inserted.
    size_t i = 0u;
    size_t j = 0u;
    auto missing_page = false;
    while (i < _pages.size() && j < other._pages.size()) {
        const auto a_idx = _pages[i].page_index;
        const auto b_idx = other._pages[j].page_index;
        if (a_idx < b_idx) {
            ++i;
        } else if (b_idx < a_idx) {
            missing_page = true;
            break;
        } else {
            if (_pages[i].page != other._pages[j].page) {
                const auto &bw = other._pages[j].page->words;
                const auto &aw = _pages[i].page->words;
                uint64_t diff = 0u;
                for (size_t w = 0u; w < words_per_page; ++w) {
                    diff |= bw[w] & ~aw[w];
                }
                if (diff != 0u) {
                    auto p = ensure_unique_page(_pages[i].page);
                    for (size_t w = 0u; w < words_per_page; ++w) {
                        p->words[w] |= bw[w];
                    }
                }
            }
            ++i;
            ++j;
        }
    }
    if (!missing_page && j == other._pages.size()) { return; }

    luisa::vector<PageEntry> merged;
    merged.reserve(_pages.size() + other._pages.size());
    i = 0u;
    j = 0u;
    while (i < _pages.size() && j < other._pages.size()) {
        const auto a_idx = _pages[i].page_index;
        const auto b_idx = other._pages[j].page_index;
        if (a_idx < b_idx) {
            merged.emplace_back(std::move(_pages[i++]));
        } else if (b_idx < a_idx) {
            merged.emplace_back(other._pages[j++]);
        } else {
            if (_pages[i].page != other._pages[j].page) {
                const auto &bw = other._pages[j].page->words;
                const auto &aw = _pages[i].page->words;
                uint64_t diff = 0u;
                for (size_t w = 0u; w < words_per_page; ++w) {
                    diff |= bw[w] & ~aw[w];
                }
                if (diff != 0u) {
                    auto p = ensure_unique_page(_pages[i].page);
                    for (size_t w = 0u; w < words_per_page; ++w) {
                        p->words[w] |= bw[w];
                    }
                }
            }
            merged.emplace_back(std::move(_pages[i++]));
            ++j;
        }
    }
    while (i < _pages.size()) { merged.emplace_back(std::move(_pages[i++])); }
    while (j < other._pages.size()) { merged.emplace_back(other._pages[j++]); }
    _pages = std::move(merged);
}

void DenseValueSet::intersect_with(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    if (this == &other || _pages.empty()) { return; }
    if (other._pages.empty()) {
        _pages.clear();
        return;
    }
    size_t write = 0u;
    size_t i = 0u;
    size_t j = 0u;
    while (i < _pages.size() && j < other._pages.size()) {
        const auto a_idx = _pages[i].page_index;
        const auto b_idx = other._pages[j].page_index;
        if (a_idx < b_idx) {
            ++i;
        } else if (b_idx < a_idx) {
            ++j;
        } else {
            if (_pages[i].page == other._pages[j].page) {
                if (write != i) { _pages[write] = std::move(_pages[i]); }
                ++write;
            } else {
                const auto &aw = _pages[i].page->words;
                const auto &bw = other._pages[j].page->words;
                uint64_t any = 0u;
                uint64_t diff = 0u;
                for (size_t w = 0u; w < words_per_page; ++w) {
                    const auto nw = aw[w] & bw[w];
                    any |= nw;
                    diff |= nw ^ aw[w];
                }
                if (any != 0u) {
                    if (diff != 0u) {
                        auto p = ensure_unique_page(_pages[i].page);
                        for (size_t w = 0u; w < words_per_page; ++w) {
                            p->words[w] &= bw[w];
                        }
                    }
                    if (write != i) { _pages[write] = std::move(_pages[i]); }
                    ++write;
                }
            }
            ++i;
            ++j;
        }
    }
    _pages.resize(write);
}

void DenseValueSet::subtract(const DenseValueSet &other) noexcept {
    LUISA_DEBUG_ASSERT(_bit_count == other._bit_count,
                       "Dense value domains do not match.");
    if (_pages.empty() || other._pages.empty()) { return; }
    if (this == &other) {
        _pages.clear();
        return;
    }
    size_t write = 0u;
    size_t i = 0u;
    size_t j = 0u;
    while (i < _pages.size() && j < other._pages.size()) {
        const auto a_idx = _pages[i].page_index;
        const auto b_idx = other._pages[j].page_index;
        if (a_idx < b_idx) {
            if (write != i) { _pages[write] = std::move(_pages[i]); }
            ++write;
            ++i;
        } else if (b_idx < a_idx) {
            ++j;
        } else {
            if (_pages[i].page != other._pages[j].page) {
                const auto &aw = _pages[i].page->words;
                const auto &bw = other._pages[j].page->words;
                uint64_t any = 0u;
                uint64_t diff = 0u;
                for (size_t w = 0u; w < words_per_page; ++w) {
                    const auto nw = aw[w] & ~bw[w];
                    any |= nw;
                    diff |= nw ^ aw[w];
                }
                if (any != 0u) {
                    if (diff != 0u) {
                        auto p = ensure_unique_page(_pages[i].page);
                        for (size_t w = 0u; w < words_per_page; ++w) {
                            p->words[w] &= ~bw[w];
                        }
                    }
                    if (write != i) { _pages[write] = std::move(_pages[i]); }
                    ++write;
                }
            }
            ++i;
            ++j;
        }
    }
    while (i < _pages.size()) {
        if (write != i) { _pages[write] = std::move(_pages[i]); }
        ++write;
        ++i;
    }
    _pages.resize(write);
}

bool DenseValueSet::operator==(const DenseValueSet &other) const noexcept {
    if (_bit_count != other._bit_count || _pages.size() != other._pages.size()) {
        return false;
    }
    for (size_t i = 0u; i < _pages.size(); ++i) {
        if (_pages[i].page_index != other._pages[i].page_index) { return false; }
        if (_pages[i].page != other._pages[i].page &&
            _pages[i].page->words != other._pages[i].page->words) {
            return false;
        }
    }
    return true;
}

size_t DenseValueSet::count_size() const noexcept {
    auto count = size_t{0u};
    for (const auto &entry : _pages) {
        for (auto word : entry.page->words) {
            count += static_cast<size_t>(std::popcount(word));
        }
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
                if (!scope.allows_successor(scope.blocks[i], successor)) { return; }
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

    // Direct exact external union over all blocks (killed_in kills local
    // exposed uses) plus one linear Tarjan SCC pass for touched reachability.
    for (size_t i = 0u; i < block_count; ++i) {
        auto local_external = local_transfers[i].external;
        local_external.subtract(killed_in[i]);
        result.external.union_with(local_external);
    }

    constexpr auto invalid_scc = static_cast<size_t>(-1);
    luisa::vector<size_t> tarjan_index(block_count, invalid_scc);
    luisa::vector<size_t> lowlink(block_count, 0u);
    luisa::vector<uint8_t> on_stack(block_count, 0u);
    luisa::vector<size_t> tarjan_stack;
    tarjan_stack.reserve(block_count);
    luisa::vector<size_t> scc_id(block_count, invalid_scc);
    luisa::vector<luisa::vector<size_t>> sccs;
    size_t next_tarjan_index = 0u;

    struct TarjanFrame {
        size_t u;
        size_t next_edge;
    };
    luisa::vector<TarjanFrame> dfs_stack;
    for (size_t root = 0u; root < block_count; ++root) {
        if (tarjan_index[root] != invalid_scc) { continue; }
        tarjan_index[root] = lowlink[root] = next_tarjan_index++;
        tarjan_stack.emplace_back(root);
        on_stack[root] = 1u;
        dfs_stack.emplace_back(TarjanFrame{root, 0u});
        while (!dfs_stack.empty()) {
            auto &frame = dfs_stack.back();
            auto u = frame.u;
            const auto &succs = successors[u];
            if (frame.next_edge < succs.size()) {
                auto v = succs[frame.next_edge++];
                if (tarjan_index[v] == invalid_scc) {
                    tarjan_index[v] = lowlink[v] = next_tarjan_index++;
                    tarjan_stack.emplace_back(v);
                    on_stack[v] = 1u;
                    dfs_stack.emplace_back(TarjanFrame{v, 0u});
                } else if (on_stack[v] != 0u) {
                    lowlink[u] = std::min(lowlink[u], tarjan_index[v]);
                }
            } else {
                if (lowlink[u] == tarjan_index[u]) {
                    auto sid = sccs.size();
                    auto &members = sccs.emplace_back();
                    for (;;) {
                        auto w = tarjan_stack.back();
                        tarjan_stack.pop_back();
                        on_stack[w] = 0u;
                        scc_id[w] = sid;
                        members.emplace_back(w);
                        if (w == u) { break; }
                    }
                }
                auto completed = u;
                dfs_stack.pop_back();
                if (!dfs_stack.empty()) {
                    auto parent = dfs_stack.back().u;
                    lowlink[parent] = std::min(lowlink[parent], lowlink[completed]);
                }
            }
        }
    }

    luisa::vector<DenseValueSet> scc_touched(
        sccs.size(), DenseValueSet{value_count});
    luisa::vector<uint32_t> pred_scc_epoch(sccs.size(), 0u);
    uint32_t scc_epoch = 0u;
    for (size_t s = sccs.size(); s-- > 0u;) {
        ++scc_epoch;
        auto &acc = scc_touched[s];
        for (auto u : sccs[s]) {
            acc.union_with(local_transfers[u].touched);
            for (auto p : predecessors[u]) {
                auto ps = scc_id[p];
                if (ps != s && pred_scc_epoch[ps] != scc_epoch) {
                    pred_scc_epoch[ps] = scc_epoch;
                    acc.union_with(scc_touched[ps]);
                }
            }
        }
        result.touched.union_with(acc);
    }
    result.may_block_evaluations += block_count;

    for (size_t i = 0u; i < block_count; ++i) {
        result.killed_at_exit[i] = killed_out[i];
        result.touched_at_exit[i] = scc_touched[scc_id[i]];
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

CoroCallContextDataflow analyze_coro_call_contexts(
    FunctionDefinition *definition, const CoroCfgDistillResult &cfg,
    const DenseValueDomain &domain, CoroReplayableValueAnalysis &replayable) noexcept {
    CoroCallContextDataflow result;
    luisa::unordered_map<Value *, size_t> selectors;
    for (auto *block : definition->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<AllocaInst>() &&
                static_cast<AllocaInst *>(inst)->coro_return_selector() != 0u) {
                auto active_selector = false;
                for (auto *use : inst->use_list()) {
                    auto *user = use == nullptr ? nullptr : use->user();
                    if (user != nullptr && user->isa<LoadInst>()) {
                        for (auto *lu : static_cast<LoadInst *>(user)->use_list()) {
                            auto *luser = lu == nullptr ? nullptr : lu->user();
                            if (luser != nullptr && luser->isa<IndexedBranchInst>() &&
                                static_cast<IndexedBranchInst *>(luser)->value() == user) {
                                active_selector = true;
                                break;
                            }
                        }
                    }
                    if (active_selector) { break; }
                }
                if (active_selector) {
                    selectors.emplace(inst, selectors.size());
                }
            }
        }
    }
    if (selectors.empty()) { return result; }
    CoroSemanticGraph graph{definition};
    LUISA_ASSERT(graph.valid(), "Invalid shared-callable semantic CFG.");

    // Intern sparse non-zero selector assignments instead of storing full
    // selector vectors in an ordered map on every CFG state.
    using SparseCtx = luisa::vector<std::pair<uint32_t, uint32_t>>;
    struct SparseCtxHash {
        [[nodiscard]] size_t operator()(const SparseCtx &ctx) const noexcept {
            auto h = luisa::hash64_default_seed;
            auto sz = ctx.size();
            h = luisa::hash64(&sz, sizeof(sz), h);
            if (!ctx.empty()) {
                h = luisa::hash64(ctx.data(), ctx.size() * sizeof(ctx[0]), h);
            }
            return static_cast<size_t>(h);
        }
    };
    luisa::vector<SparseCtx> contexts;
    luisa::unordered_map<SparseCtx, uint32_t, SparseCtxHash> context_ids;
    auto intern_context = [&](SparseCtx ctx) noexcept -> uint32_t {
        if (auto it = context_ids.find(ctx); it != context_ids.end()) {
            return it->second;
        }
        auto id = static_cast<uint32_t>(contexts.size());
        context_ids.emplace(ctx, id);
        contexts.emplace_back(std::move(ctx));
        return id;
    };
    auto update_context = [&](uint32_t ctx_id, uint32_t sel_idx, uint32_t val) noexcept -> uint32_t {
        const auto &cur = contexts[ctx_id];
        auto it = std::lower_bound(
            cur.begin(), cur.end(), sel_idx,
            [](const auto &entry, uint32_t key) noexcept { return entry.first < key; });
        if (it != cur.end() && it->first == sel_idx) {
            if (it->second == val) { return ctx_id; }
        } else if (val == 0u) {
            return ctx_id;
        }
        SparseCtx next;
        next.reserve(cur.size() + 1u);
        for (auto pos = cur.begin(); pos != it; ++pos) { next.emplace_back(*pos); }
        if (val != 0u) { next.emplace_back(sel_idx, val); }
        for (auto pos = (it != cur.end() && it->first == sel_idx) ? std::next(it) : it;
             pos != cur.end(); ++pos) {
            next.emplace_back(*pos);
        }
        return intern_context(std::move(next));
    };
    auto lookup_context = [&](uint32_t ctx_id, uint32_t sel_idx) noexcept -> uint32_t {
        const auto &cur = contexts[ctx_id];
        auto it = std::lower_bound(
            cur.begin(), cur.end(), sel_idx,
            [](const auto &entry, uint32_t key) noexcept { return entry.first < key; });
        return (it != cur.end() && it->first == sel_idx) ? it->second : 0u;
    };

    struct State {
        size_t block;
        uint32_t context;
        luisa::vector<size_t> successors;
        luisa::vector<size_t> predecessors;
    };
    luisa::vector<State> states;
    using Key = std::pair<size_t, uint32_t>;
    struct KeyHash {
        [[nodiscard]] size_t operator()(const Key &k) const noexcept {
            return static_cast<size_t>(luisa::hash64(&k, sizeof(k), luisa::hash64_default_seed));
        }
    };
    luisa::unordered_map<Key, size_t, KeyHash> state_ids;
    auto intern = [&](size_t block, uint32_t context) noexcept {
        auto [i, added] = state_ids.emplace(Key{block, context}, states.size());
        if (added) { states.push_back({block, context, {}, {}}); }
        return i->second;
    };
    auto initial_ctx = intern_context({});
    intern(graph.block_id(definition->body_block()), initial_ctx);
    for (size_t id = 0; id < states.size(); ++id) {
        auto context = states[id].context;
        auto block_id = states[id].block;
        auto *block = graph.block(block_id);
        auto *term = block->terminator();
        auto *dispatch = term->isa<IndexedBranchInst>() ? static_cast<IndexedBranchInst *>(term) : nullptr;
        luisa::optional<uint32_t> return_site;
        for (auto *inst : block->instructions()) {
            if (inst->isa<LoadInst>()) {
                auto *load = static_cast<LoadInst *>(inst);
                if (auto i = selectors.find(load->variable()); i != selectors.end()) {
                    if (dispatch != nullptr && dispatch->value() == load) {
                        return_site = lookup_context(context, static_cast<uint32_t>(i->second));
                    }
                }
            } else if (inst->isa<StoreInst>()) {
                auto *store = static_cast<StoreInst *>(inst);
                if (auto i = selectors.find(store->variable()); i != selectors.end()) {
                    uint64_t site = 0;
                    LUISA_ASSERT(try_decode_constant_nonnegative_integer(store->value(), site) && site <= UINT32_MAX,
                                 "Coroutine return selector requires a constant call site.");
                    context = update_context(
                        context, static_cast<uint32_t>(i->second),
                        static_cast<uint32_t>(site));
                }
            }
        }
        auto append = [&](size_t next_block) {
            auto next = intern(next_block, context);
            states[id].successors.emplace_back(next);
            states[next].predecessors.emplace_back(id);
        };
        if (return_site) {
            auto *target = dispatch->default_block();
            for (size_t i = 0; i < dispatch->case_count(); ++i) {
                if (dispatch->case_value(i) == *return_site) {
                    target = dispatch->case_block(i);
                    break;
                }
            }
            append(graph.block_id(target));
        } else {
            for (auto next : graph.successors(block_id)) { append(next); }
        }
    }
    result.state_count = states.size();
    auto count = domain.size();
    luisa::vector<DenseBlockEffect> effects;
    effects.reserve(graph.block_count());
    luisa::vector<uint8_t> transfer_marks(count, 0u);
    for (size_t i = 0; i < graph.block_count(); ++i) {
        SparseBlockEffect sparse;
        SparseBlockTransferState transfer{
            .domain = &domain,
            .replayable = &replayable,
            .marks = transfer_marks,
            .effect = &sparse};
        for (auto *inst : graph.block(i)->instructions()) {
            transfer_instruction(inst, transfer);
        }
        auto &effect = effects.emplace_back(count);
        for (auto atom : sparse.killed) { effect.killed.set(atom); }
        for (auto atom : sparse.external) { effect.external.set(atom); }
        for (auto atom : sparse.touched) { effect.touched.set(atom); }
        transfer.reset_marks();
    }
    auto transfer_extensions = [&](BasicBlock *block, DenseValueSet live) {
        auto *term = block->terminator();
        if (!term->isa<CoroSuspendInst>()) { return live; }
        auto *suspend = static_cast<CoroSuspendInst *>(term);
        auto extensions = suspend->extensions();
        for (size_t reverse = 0; reverse < extensions.size(); ++reverse) {
            auto &extension = extensions[extensions.size() - 1u - reverse];
            DenseValueSet uses{count}, defs{count};
            for (auto &binding : extension->bindings()) {
                if (binding.lifetime == CoroSuspendBindingLifetime::boundary) { continue; }
                auto *value = suspend->extension_binding_value(binding.index);
                if (binding.access == CoroSuspendBindingAccess::read) {
                    if (auto index = domain.ssa_index(value)) { uses.set(*index); }
                } else {
                    for (auto access : domain.memory_accesses(value)) {
                        defs.set(access.atom_index);
                        if (binding.access == CoroSuspendBindingAccess::read_write) { uses.set(access.atom_index); }
                    }
                }
            }
            live.subtract(defs);
            live.union_with(uses);
        }
        return live;
    };
    // Compose ordered external stages with the source block once. Each
    // transfer is distributive: use union (out - kill).
    for (size_t i = 0; i < graph.block_count(); ++i) {
        auto *block = graph.block(i);
        auto extension_input = transfer_extensions(block, DenseValueSet{count});
        extension_input.subtract(effects[i].killed);
        effects[i].external.union_with(extension_input);
        if (auto *term = block->terminator(); term->isa<CoroSuspendInst>()) {
            auto *suspend = static_cast<CoroSuspendInst *>(term);
            for (auto &extension : suspend->extensions()) {
                for (auto &binding : extension->bindings()) {
                    if (binding.lifetime != CoroSuspendBindingLifetime::boundary && binding.access != CoroSuspendBindingAccess::read) {
                        for (auto access : domain.memory_accesses(suspend->extension_binding_value(binding.index))) { effects[i].killed.set(access.atom_index); }
                    }
                }
            }
        }
    }
    auto solve = [&](const luisa::vector<uint8_t> &active, bool cut_suspend) {
        luisa::vector<DenseValueSet> live(states.size(), DenseValueSet{count});
        luisa::deque<size_t> work;
        luisa::vector<uint8_t> queued = active;
        // Initialize in DFS postorder (successors before predecessors) so
        // backward liveness converges along acyclic call chains in one sweep.
        luisa::vector<uint8_t> post_visited(states.size(), 0u);
        struct PostFrame {
            size_t id;
            size_t next_succ;
        };
        luisa::vector<PostFrame> post_stack;
        for (size_t root = 0u; root < states.size(); ++root) {
            if (!active[root] || post_visited[root] != 0u) { continue; }
            post_visited[root] = 1u;
            post_stack.emplace_back(PostFrame{root, 0u});
            while (!post_stack.empty()) {
                auto &frame = post_stack.back();
                auto id = frame.id;
                const auto &succs = states[id].successors;
                auto is_cut = cut_suspend &&
                              graph.block(states[id].block)->terminator()->isa<CoroSuspendInst>();
                if (!is_cut && frame.next_succ < succs.size()) {
                    auto next_id = succs[frame.next_succ++];
                    if (active[next_id] && post_visited[next_id] == 0u) {
                        post_visited[next_id] = 1u;
                        post_stack.emplace_back(PostFrame{next_id, 0u});
                    }
                } else {
                    work.emplace_back(id);
                    post_stack.pop_back();
                }
            }
        }
        while (!work.empty()) {
            auto id = work.front();
            work.pop_front();
            queued[id] = 0u;
            auto &state = states[id];
            auto *block = graph.block(state.block);
            DenseValueSet next{count};
            if (!cut_suspend || !block->terminator()->isa<CoroSuspendInst>()) {
                for (auto successor : state.successors) {
                    if (active[successor]) { next.union_with(live[successor]); }
                }
            }
            next.subtract(effects[state.block].killed);
            next.union_with(effects[state.block].external);
            if (!(next == live[id])) {
                live[id] = std::move(next);
                for (auto pred : state.predecessors) {
                    if (active[pred] && !queued[pred]) {
                        queued[pred] = 1u;
                        work.emplace_back(pred);
                    }
                }
            }
        }
        if (auto *flag = std::getenv("LUISA_CORO_VERIFY_DENSE_DATAFLOW"); flag != nullptr && luisa::string_view{flag} == "1") {
            // Independent full-sweep set oracle over the matched state graph.
            // The legacy scope-only oracle has a different path domain.
            luisa::vector<luisa::unordered_set<size_t>> oracle(states.size());
            bool changed;
            do {
                changed = false;
                // States are discovered forward from entry. Sweep backward so
                // liveness can cross a long call-context path in one round,
                // instead of advancing only one predecessor per forward sweep.
                // Every active state is still recomputed until the complete
                // monotone set equations reach their least fixed point.
                for (size_t reverse = states.size(); reverse > 0u; --reverse) {
                    auto id = reverse - 1u;
                    if (!active[id]) { continue; }
                    auto &state = states[id];
                    luisa::unordered_set<size_t> next;
                    if (!cut_suspend || !graph.block(state.block)->terminator()->isa<CoroSuspendInst>()) {
                        for (auto successor : state.successors) {
                            if (active[successor]) { next.insert(oracle[successor].begin(), oracle[successor].end()); }
                        }
                    }
                    effects[state.block].killed.for_each_set_bit([&](size_t atom) { next.erase(atom); });
                    effects[state.block].external.for_each_set_bit([&](size_t atom) { next.emplace(atom); });
                    if (!same_set(next, oracle[id])) {
                        oracle[id] = std::move(next);
                        changed = true;
                    }
                }
            } while (changed);
            for (size_t id = 0; id < states.size(); ++id) {
                LUISA_ASSERT(live[id].count_size() == oracle[id].size(), "Matched coroutine liveness differs from set oracle.");
                live[id].for_each_set_bit([&](size_t atom) { LUISA_ASSERT(oracle[id].contains(atom), "Matched coroutine liveness atom differs from set oracle."); });
            }
        }
        return live;
    };
    auto live = solve(luisa::vector<uint8_t>(states.size(), 1u), false);
    result.scope_live.assign(cfg.scopes.size(), DenseValueSet{count});
    result.scope_external.assign(cfg.scopes.size(), DenseValueSet{count});
    result.edge_target_live.assign(cfg.transition_edges.size(), DenseValueSet{count});
    for (size_t scope_id = 0; scope_id < cfg.scopes.size(); ++scope_id) {
        auto &scope = cfg.scopes[scope_id];
        if (scope.blocks.empty()) { continue; }
        luisa::vector<uint8_t> member(graph.block_count(), 0u);
        for (auto *block : scope.blocks) { member[graph.block_id(block)] = 1u; }
        auto root = graph.block_id(scope.blocks.front());
        luisa::vector<uint8_t> active(states.size(), 0u);
        luisa::vector<size_t> reachable;
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i].block == root) {
                active[i] = 1u;
                reachable.emplace_back(i);
                result.scope_live[scope_id].union_with(live[i]);
            }
        }
        for (size_t cursor = 0; cursor < reachable.size(); ++cursor) {
            auto id = reachable[cursor];
            auto *block = graph.block(states[id].block);
            if (block->terminator()->isa<CoroSuspendInst>()) { continue; }
            for (auto next : states[id].successors) {
                if (member[states[next].block] && !active[next]) {
                    active[next] = 1u;
                    reachable.emplace_back(next);
                }
            }
        }
        auto external = solve(active, true);
        for (auto id : reachable) {
            if (states[id].block == root) { result.scope_external[scope_id].union_with(external[id]); }
        }
        for (size_t edge_id = 0; edge_id < cfg.transition_edges.size(); ++edge_id) {
            auto &edge = cfg.transition_edges[edge_id];
            if (edge.from_scope != scope_id) { continue; }
            auto exit = graph.block_id(edge.exit_block);
            for (auto id : reachable) {
                if (states[id].block != exit) { continue; }
                for (auto next : states[id].successors) {
                    if (edge.is_suspend || !member[states[next].block]) {
                        result.edge_target_live[edge_id].union_with(live[next]);
                    }
                }
            }
        }
    }
    return result;
}

}// namespace luisa::compute::xir::detail
