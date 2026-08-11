#include <algorithm>
#include <bit>
#include <cstdlib>
#include <type_traits>
#include <utility>

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

#include "../pointer_containers.h"
#include "coro_frame_abi.h"
#include "coro_frame_access.h"
#include "coro_replayable.h"
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
    h.add(uint64_t{3u});
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
        auto hash_indices = [&](auto &indices) noexcept {
            h.add(indices.size());
            for (auto index : indices) { h.add(index); }
        };
        hash_values(scope.external_values);
        hash_values(scope.touched_values);
        hash_values(scope.live_in_values);
        hash_values(scope.live_out_values);
        hash_indices(scope.external_frame_value_indices);
        hash_indices(scope.touched_frame_value_indices);
        hash_indices(scope.live_in_frame_value_indices);
        hash_indices(scope.live_out_frame_value_indices);
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
        auto hash_indices = [&](auto &indices) noexcept {
            h.add(indices.size());
            for (auto index : indices) { h.add(index); }
        };
        hash_values(edge.killed_values);
        hash_values(edge.touched_values);
        hash_values(edge.live_values);
        hash_values(edge.store_values);
        hash_indices(edge.killed_frame_value_indices);
        hash_indices(edge.touched_frame_value_indices);
        hash_indices(edge.live_frame_value_indices);
        hash_indices(edge.store_frame_value_indices);
        hash_names(edge.killed_variables);
        hash_names(edge.touched_variables);
        hash_names(edge.live_variables);
        hash_names(edge.store_variables);
    }

    h.add(result.frame_values.size());
    for (auto &frame_value : result.frame_values) {
        h.add_pointer(frame_value.value);
        h.add(frame_value.access_chain.size());
        for (auto index : frame_value.access_chain) { h.add(index); }
        h.add_string(frame_value.name);
        h.add_pointer(frame_value.type);
        h.add(frame_value.slot);
        h.add(frame_value.bit_offset.has_value());
        if (frame_value.bit_offset) { h.add(*frame_value.bit_offset); }
    }
    h.add(result.frame_slots.size());
    for (auto &frame_slot : result.frame_slots) {
        h.add_string(frame_slot.name);
        h.add_pointer(frame_slot.type);
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

[[nodiscard]] static luisa::string frame_value_name(
    Value *value, luisa::span<const uint32_t> access_chain,
    size_t index) noexcept {
    if (auto *alloca = trace_local_alloca(value)) {
        if (auto name = alloca->name()) {
            auto result = luisa::string{name.value()};
            for (auto component : access_chain) {
                result.append(luisa::format(".{}", component));
            }
            return result;
        }
    }
    return luisa::format("_coro_frame_{}", index);
}

template<typename T>
[[nodiscard]] static bool same_set(const luisa::unordered_set<T> &a,
                                   const luisa::unordered_set<T> &b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (auto &v : a) {
        if (!b.contains(v)) { return false; }
    }
    return true;
}

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

    [[nodiscard]] size_t count_size() const noexcept {
        auto count = size_t{0u};
        for (auto word : _words) {
            count += static_cast<size_t>(std::popcount(word));
        }
        return count;
    }

    template<typename F>
    void for_each_set_bit(F &&visit) const noexcept {
        for (size_t word_index = 0u;
             word_index < _words.size(); ++word_index) {
            auto word = _words[word_index];
            while (word != 0u) {
                auto bit = static_cast<size_t>(std::countr_zero(word));
                visit(word_index * 64u + bit);
                word &= word - 1u;
            }
        }
    }
};

// One immutable value-number domain is shared by every block, scope, edge, and
// inter-scope fixed point. All possible frame values are instructions in the
// source definition: local allocas represent addressable state, while typed
// non-lvalue non-terminators represent SSA state. Numbering this exact
// superset once lets every subsequent relation use the same bit coordinates.
class DenseValueDomain {

private:
    detail::CoroFrameAtomDomain _atoms;

public:
    explicit DenseValueDomain(FunctionDefinition *definition) noexcept
        : _atoms{definition} {}

    [[nodiscard]] size_t size() const noexcept { return _atoms.size(); }

    [[nodiscard]] luisa::optional<size_t> ssa_index(
        Value *value) const noexcept {
        return _atoms.ssa_index(value);
    }

    [[nodiscard]] luisa::span<const detail::CoroFrameAtomDomain::MemoryAccess>
    memory_accesses(
        Value *pointer) const noexcept {
        return _atoms.memory_accesses(pointer);
    }

    [[nodiscard]] const auto &atom(size_t index) const noexcept {
        return _atoms.atom(index);
    }

    [[nodiscard]] const auto &atom_domain() const noexcept { return _atoms; }

    [[nodiscard]] size_t split_alloca_count() const noexcept {
        return _atoms.split_alloca_count();
    }

    [[nodiscard]] size_t split_atom_count() const noexcept {
        return _atoms.split_atom_count();
    }

    void append_indices(luisa::vector<size_t> &destination,
                        const DenseValueSet &source) const noexcept {
        destination.clear();
        destination.reserve(source.count_size());
        source.for_each_set_bit([&](size_t i) noexcept {
            LUISA_DEBUG_ASSERT(i < _atoms.size(),
                               "Coroutine atom bit exceeds its domain.");
            destination.emplace_back(i);
        });
    }
};

struct PointerScopeDataflowState {
    const DenseValueDomain *domain{nullptr};
    detail::CoroReplayableValueAnalysis *replayable{nullptr};
    luisa::unordered_set<size_t> killed;
    luisa::unordered_set<size_t> external;
    luisa::unordered_set<size_t> touched;

    PointerScopeDataflowState() noexcept = default;

    explicit PointerScopeDataflowState(
        const DenseValueDomain &value_domain,
        detail::CoroReplayableValueAnalysis &analysis) noexcept
        : domain{&value_domain}, replayable{&analysis} {}

    void kill(size_t index) noexcept { killed.emplace(index); }
    void expose(size_t index) noexcept { external.emplace(index); }
    void touch(size_t index) noexcept { touched.emplace(index); }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return killed.contains(index);
    }
};

struct DenseScopeDataflowState {
    const DenseValueDomain *domain;
    detail::CoroReplayableValueAnalysis *replayable;
    DenseValueSet killed;
    DenseValueSet external;
    DenseValueSet touched;

    explicit DenseScopeDataflowState(
        const DenseValueDomain &value_domain,
        detail::CoroReplayableValueAnalysis &analysis) noexcept
        : domain{&value_domain},
          replayable{&analysis},
          killed{value_domain.size()},
          external{value_domain.size()},
          touched{value_domain.size()} {}

    void kill(size_t index) noexcept {
        killed.set(index);
    }
    void expose(size_t index) noexcept {
        external.set(index);
    }
    void touch(size_t index) noexcept {
        touched.set(index);
    }
    [[nodiscard]] bool is_killed(size_t index) const noexcept {
        return killed.test(index);
    }
};

template<typename State>
static void touch_index(size_t index, State &state) noexcept {
    state.kill(index);
    state.touch(index);
}

template<typename State>
static void use_index(size_t index, State &state) noexcept {
    if (!state.is_killed(index)) { state.expose(index); }
}

template<typename State>
static void use_value(Value *value, State &state) noexcept {
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
static void touch_value(Value *value, State &state) noexcept {
    if (value == nullptr) { return; }
    if (auto index = state.domain->ssa_index(value)) {
        touch_index(*index, state);
    }
}

template<typename State>
static void use_memory(Value *pointer, State &state) noexcept {
    for (auto access : state.domain->memory_accesses(pointer)) {
        use_index(access.atom_index, state);
    }
}

template<typename State>
static void touch_memory(Value *pointer, State &state,
                         bool definite) noexcept {
    for (auto access : state.domain->memory_accesses(pointer)) {
        if (definite) {
            if (access.covers_atom) {
                state.kill(access.atom_index);
            } else {
                // A partial store must preserve the bytes outside its path.
                // Model that dependence before recording the write so a live
                // outgoing aggregate is reloaded on entry to this scope.
                use_index(access.atom_index, state);
            }
        }
        state.touch(access.atom_index);
    }
}

template<typename State>
static void begin_memory_lifetime(Value *pointer, State &state) noexcept {
    // ALLOCA denotes fresh, undefined storage each time execution reaches the
    // instruction. This is a must-kill for every atom rooted at the local,
    // including all leaves of a split aggregate, but it is not a write: no
    // value becomes live or needs to be stored into the coroutine frame.
    for (auto access : state.domain->memory_accesses(pointer)) {
        state.kill(access.atom_index);
    }
}

template<typename State>
static void use_pointer_indices(Value *value, State &state) noexcept {
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

template<typename State>
static void transfer_call_instruction(CallInst *call, State &state) noexcept {
    auto arg_iter = call->callee()->arguments().begin();
    for (auto *arg_use : call->argument_uses()) {
        auto *argument = arg_use->value();
        if (arg_iter != call->callee()->arguments().end() &&
            (*arg_iter)->is_reference()) {
            use_pointer_indices(argument, state);
            if (auto *alloca = trace_local_alloca(argument)) {
                static_cast<void>(alloca);
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
static void transfer_instruction(Instruction *inst, State &state) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA: {
            auto *alloca = static_cast<AllocaInst *>(inst);
            if (alloca->is_local()) {
                begin_memory_lifetime(alloca, state);
            }
            break;
        }
        case DerivedInstructionTag::GEP: {
            // Computing an address does not read the pointee. Only the index
            // expressions are SSA uses; the eventual load/store/call models
            // the selected memory atom.
            use_pointer_indices(inst, state);
            break;
        }
        case DerivedInstructionTag::CALL: {
            transfer_call_instruction(static_cast<CallInst *>(inst), state);
            break;
        }
        case DerivedInstructionTag::LOAD: {
            auto *load = static_cast<LoadInst *>(inst);
            use_pointer_indices(load->variable(), state);
            if (auto *alloca = trace_local_alloca(load->variable())) {
                static_cast<void>(alloca);
                use_memory(load->variable(), state);
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
                static_cast<void>(alloca);
                touch_memory(store->variable(), state, true);
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
                static_cast<void>(alloca);
                use_memory(atomic->base(), state);
                touch_memory(atomic->base(), state, false);
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

struct PointerScopeDataflowResult {
    luisa::unordered_set<size_t> external;
    luisa::unordered_set<size_t> touched;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<size_t>> killed_at_exit;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<size_t>> touched_at_exit;
};

[[nodiscard]] static bool same_pointer_state(
    const PointerScopeDataflowState &a,
    const PointerScopeDataflowState &b) noexcept {
    return same_set(a.killed, b.killed) &&
           same_set(a.external, b.external) &&
           same_set(a.touched, b.touched);
}

static void merge_pointer_state_into_entry(
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

[[nodiscard]] static PointerScopeDataflowResult
analyze_scope_use_def_pointer_oracle(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    detail::CoroReplayableValueAnalysis &replayable) noexcept {
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

struct DenseScopeDataflowResult {
    DenseValueSet external;
    DenseValueSet touched;
    luisa::vector<DenseValueSet> killed_at_exit;
    luisa::vector<DenseValueSet> touched_at_exit;
    size_t fixed_point_block_evaluations{0u};

    DenseScopeDataflowResult(size_t block_count,
                             size_t value_count) noexcept
        : external{value_count},
          touched{value_count},
          killed_at_exit(block_count, DenseValueSet{value_count}),
          touched_at_exit(block_count, DenseValueSet{value_count}) {}
};

[[nodiscard]] static DenseScopeDataflowResult analyze_scope_use_def(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    detail::CoroReplayableValueAnalysis &replayable) noexcept {
    auto block_count = scope.blocks.size();
    auto value_count = value_domain.size();
    DenseScopeDataflowResult result{block_count, value_count};
    if (scope.blocks.empty()) { return result; }

    DensePointerMap<BasicBlock *, size_t> block_indices;
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
    luisa::vector<DenseScopeDataflowState> local_transfers;
    local_transfers.reserve(block_count);
    for (auto *block : scope.blocks) {
        auto &local = local_transfers.emplace_back(
            value_domain, replayable);
        for (auto *instruction : block->instructions()) {
            transfer_instruction(instruction, local);
        }
    }

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
            ++result.fixed_point_block_evaluations;
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
        auto to_pointer_set = [&](const DenseValueSet &dense) noexcept {
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
        auto dense_external = to_pointer_set(result.external);
        auto dense_touched = to_pointer_set(result.touched);
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
            "Dense coroutine dataflow differs from pointer oracle for scope "
            "token {} (external dense-only={}, oracle-only={}; touched "
            "dense-only={}, oracle-only={}).",
            scope.trigger_token,
            dense_only_external.size(), oracle_only_external.size(),
            dense_only_touched.size(), oracle_only_touched.size());
        for (size_t block_index = 0u;
             block_index < scope.blocks.size(); ++block_index) {
            auto *block = scope.blocks[block_index];
            if (!block->is_terminated()) { continue; }
            auto dense_killed =
                to_pointer_set(result.killed_at_exit[block_index]);
            auto dense_touched_at_exit =
                to_pointer_set(result.touched_at_exit[block_index]);
            auto oracle_killed = oracle.killed_at_exit.find(block);
            auto oracle_touched = oracle.touched_at_exit.find(block);
            auto empty = luisa::unordered_set<size_t>{};
            auto &oracle_killed_set = oracle_killed == oracle.killed_at_exit.end() ?
                                          empty :
                                          oracle_killed->second;
            auto &oracle_touched_set = oracle_touched == oracle.touched_at_exit.end() ?
                                           empty :
                                           oracle_touched->second;
            LUISA_ASSERT(
                same_set(dense_killed, oracle_killed_set) &&
                    same_set(dense_touched_at_exit, oracle_touched_set),
                "Dense coroutine exit dataflow differs from pointer oracle "
                "for scope token {}.",
                scope.trigger_token);
        }
    }
    return result;
}

static void append_legacy_values(
    luisa::vector<Value *> &dst, const DenseValueSet &atoms,
    const DenseValueDomain &domain) noexcept {
    dst.clear();
    luisa::unordered_set<Value *> seen;
    atoms.for_each_set_bit([&](size_t atom_index) noexcept {
        auto *root = domain.atom(atom_index).root;
        if (root != nullptr && seen.emplace(root).second) {
            dst.emplace_back(root);
        }
    });
}

static void append_frame_value_indices(
    luisa::vector<size_t> &dst, const DenseValueSet &atoms,
    luisa::span<const std::pair<size_t, size_t>>
        atom_to_frame_value_range) noexcept {
    dst.clear();
    atoms.for_each_set_bit([&](size_t atom_index) noexcept {
        LUISA_DEBUG_ASSERT(atom_index < atom_to_frame_value_range.size(),
                           "Coroutine atom index is out of range.");
        auto [first, count] = atom_to_frame_value_range[atom_index];
        if (first != static_cast<size_t>(-1)) {
            for (size_t i = 0u; i < count; ++i) {
                dst.emplace_back(first + i);
            }
        }
    });
}

static void append_names_from_frame_values(
    luisa::vector<luisa::string> &dst,
    luisa::span<const size_t> frame_value_indices,
    const CoroCfgDistillResult &result) noexcept {
    dst.clear();
    dst.reserve(frame_value_indices.size());
    for (auto index : frame_value_indices) {
        LUISA_DEBUG_ASSERT(index < result.frame_values.size(),
                           "Coroutine frame value index is out of range.");
        dst.emplace_back(result.frame_values[index].name);
    }
}

[[nodiscard]] static size_t frame_slot_abi_size(
    luisa::span<const size_t> order,
    luisa::span<const CoroCfgDistillResult::FrameSlot> slots) noexcept {
    auto offset = size_t{0u};
    auto structure_alignment = Type::of<uint>()->alignment();
    auto append = [&](const Type *type) noexcept {
        LUISA_DEBUG_ASSERT(type != nullptr && type->alignment() != 0u,
                           "Invalid coroutine frame field type.");
        auto alignment = type->alignment();
        offset = (offset + alignment - 1u) / alignment * alignment;
        offset += type->size();
        structure_alignment =
            std::max(structure_alignment, alignment);
    };
    // The scheduler ABI fixes seven uint fields before user state.
    for (auto i = 0u; i < CORO_FRAME_RESERVED_FIELD_COUNT; ++i) {
        append(Type::of<uint>());
    }
    for (auto index : order) {
        LUISA_DEBUG_ASSERT(index < slots.size(),
                           "Coroutine frame slot index is out of range.");
        append(slots[index].type);
    }
    return (offset + structure_alignment - 1u) /
           structure_alignment * structure_alignment;
}

static void optimize_frame_slot_abi_order(
    CoroCfgDistillResult &result) noexcept {
    auto slot_count = result.frame_slots.size();
    if (slot_count < 2u) { return; }

    // Slot identities are purely physical: any permutation is semantics
    // preserving when every logical value is remapped by the same bijection.
    // Choose an ABI order against the real fixed-prefix offset rather than
    // assuming the user payload starts at its maximum alignment. At each
    // offset, list scheduling first minimizes the padding inserted before the
    // next field and then prefers the most aligned field. The candidate is
    // accepted only when the exact structure-layout objective is strictly
    // smaller, so heuristic quality can affect opportunity but never regress
    // frame size or correctness.
    luisa::vector<size_t> original_order;
    original_order.reserve(slot_count);
    for (size_t i = 0u; i < slot_count; ++i) {
        original_order.emplace_back(i);
    }
    auto candidate_order = luisa::vector<size_t>{};
    candidate_order.reserve(slot_count);
    auto remaining = original_order;
    auto offset = CORO_FRAME_RESERVED_FIELD_COUNT *
                  Type::of<uint>()->size();
    while (!remaining.empty()) {
        auto best = size_t{0u};
        auto best_padding = static_cast<size_t>(-1);
        auto best_alignment = size_t{0u};
        for (size_t i = 0u; i < remaining.size(); ++i) {
            auto *type = result.frame_slots[remaining[i]].type;
            auto alignment = type->alignment();
            auto aligned =
                (offset + alignment - 1u) / alignment * alignment;
            auto padding = aligned - offset;
            if (padding < best_padding ||
                (padding == best_padding &&
                 alignment > best_alignment)) {
                best = i;
                best_padding = padding;
                best_alignment = alignment;
            }
        }
        auto slot = remaining[best];
        auto *type = result.frame_slots[slot].type;
        offset += best_padding + type->size();
        candidate_order.emplace_back(slot);
        remaining.erase(remaining.begin() + best);
    }
    auto original_size = frame_slot_abi_size(
        original_order, result.frame_slots);
    auto candidate_size = frame_slot_abi_size(
        candidate_order, result.frame_slots);
    if (candidate_size >= original_size) { return; }

    luisa::vector<size_t> old_to_new(slot_count);
    luisa::vector<CoroCfgDistillResult::FrameSlot> reordered;
    reordered.reserve(slot_count);
    for (size_t new_index = 0u;
         new_index < candidate_order.size(); ++new_index) {
        auto old_index = candidate_order[new_index];
        old_to_new[old_index] = new_index;
        reordered.emplace_back(
            std::move(result.frame_slots[old_index]));
    }
    for (auto &value : result.frame_values) {
        LUISA_DEBUG_ASSERT(value.slot < old_to_new.size(),
                           "Coroutine frame slot index is out of range.");
        value.slot = old_to_new[value.slot];
    }
    result.frame_slots = std::move(reordered);
}

static void color_frame_slots(CoroCfgDistillResult &result) noexcept {
    auto value_count = result.frame_values.size();
    result.frame_slots.clear();
    if (value_count == 0u) { return; }

    luisa::vector<DenseValueSet> interference;
    interference.reserve(value_count);
    for (size_t i = 0u; i < value_count; ++i) {
        interference.emplace_back(value_count);
    }
    auto add_clique = [&](luisa::span<const size_t> values) noexcept {
        for (size_t i = 0u; i < values.size(); ++i) {
            LUISA_DEBUG_ASSERT(values[i] < value_count,
                               "Coroutine frame value index is out of range.");
            for (size_t j = i + 1u; j < values.size(); ++j) {
                LUISA_DEBUG_ASSERT(values[j] < value_count,
                                   "Coroutine frame value index is out of range.");
                interference[values[i]].set(values[j]);
                interference[values[j]].set(values[i]);
            }
        }
    };
    // All continuation inputs are loaded before the cloned body executes, so
    // they must occupy distinct fields. More importantly, edge.live_values is
    // the complete state that must coexist after a transition, including a
    // dormant value that the source scope neither reloads nor stores but that
    // a later continuation still needs. Coloring only edge.store_values would
    // let a newly stored value overwrite such pass-through state. Values that
    // only occur in disjoint post-transition live sets intentionally do not
    // interfere.
    for (auto &scope : result.scopes) {
        add_clique(scope.live_in_frame_value_indices);
    }
    for (auto &edge : result.transition_edges) {
        add_clique(edge.live_frame_value_indices);
    }

    luisa::vector<const Type *> type_order;
    luisa::unordered_map<const Type *, luisa::vector<size_t>> values_by_type;
    luisa::vector<size_t> bool_values;
    for (size_t i = 0u; i < value_count; ++i) {
        auto *type = result.frame_values[i].type;
        result.frame_values[i].bit_offset.reset();
        if (type == Type::of<bool>()) {
            bool_values.emplace_back(i);
            continue;
        }
        auto [iter, inserted] = values_by_type.try_emplace(type);
        if (inserted) { type_order.emplace_back(type); }
        iter->second.emplace_back(i);
    }
    std::stable_sort(
        type_order.begin(), type_order.end(),
        [](auto *lhs, auto *rhs) noexcept {
            if (lhs->alignment() != rhs->alignment()) {
                return lhs->alignment() > rhs->alignment();
            }
            return lhs->size() > rhs->size();
        });

    luisa::vector<luisa::vector<size_t>> slot_occupants;
    luisa::unordered_map<const Type *, luisa::vector<size_t>> slots_by_type;
    for (auto *type : type_order) {
        auto &values = values_by_type.at(type);
        std::stable_sort(
            values.begin(), values.end(),
            [&](size_t lhs, size_t rhs) noexcept {
                return interference[lhs].count_size() >
                       interference[rhs].count_size();
            });
        for (auto value_index : values) {
            auto &value = result.frame_values[value_index];
            auto slot_index = static_cast<size_t>(-1);
            for (auto candidate : slots_by_type[type]) {
                auto conflict = false;
                for (auto occupant : slot_occupants[candidate]) {
                    if (interference[value_index].test(occupant)) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) {
                    slot_index = candidate;
                    break;
                }
            }
            if (slot_index == static_cast<size_t>(-1)) {
                slot_index = result.frame_slots.size();
                result.frame_slots.emplace_back(
                    CoroCfgDistillResult::FrameSlot{
                        .name = value.name,
                        .type = value.type});
                slot_occupants.emplace_back();
                slots_by_type[type].emplace_back(slot_index);
            }
            value.slot = slot_index;
            slot_occupants[slot_index].emplace_back(value_index);
        }
    }

    // Boolean storage is a graph-coloring problem at bit granularity. Each
    // lane is one interference color: values assigned to it never coexist.
    // Thirty-two lanes share one dedicated uint field, while distinct lanes
    // preserve simultaneously live Boolean values in distinct bits.
    std::stable_sort(
        bool_values.begin(), bool_values.end(),
        [&](size_t lhs, size_t rhs) noexcept {
            return interference[lhs].count_size() >
                   interference[rhs].count_size();
        });
    luisa::vector<luisa::vector<size_t>> bool_lane_occupants;
    luisa::vector<size_t> bool_lane_slots;
    for (auto value_index : bool_values) {
        auto lane_index = static_cast<size_t>(-1);
        for (size_t candidate = 0u;
             candidate < bool_lane_occupants.size(); ++candidate) {
            auto conflict = false;
            for (auto occupant : bool_lane_occupants[candidate]) {
                if (interference[value_index].test(occupant)) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                lane_index = candidate;
                break;
            }
        }
        if (lane_index == static_cast<size_t>(-1)) {
            lane_index = bool_lane_occupants.size();
            bool_lane_occupants.emplace_back();
            if (lane_index % 32u == 0u) {
                result.frame_slots.emplace_back(
                    CoroCfgDistillResult::FrameSlot{
                        .name = result.frame_values[value_index].name,
                        .type = Type::of<uint>()});
            }
            bool_lane_slots.emplace_back(
                result.frame_slots.size() - 1u);
        }
        auto &value = result.frame_values[value_index];
        value.slot = bool_lane_slots[lane_index];
        value.bit_offset = static_cast<uint32_t>(lane_index % 32u);
        bool_lane_occupants[lane_index].emplace_back(value_index);
    }
    optimize_frame_slot_abi_order(result);
}

static void analyze_live_variables(CoroCfgDistillResult &result, FunctionDefinition *def) noexcept {
    auto n = result.scopes.size();
    DenseValueDomain value_domain{def};
    detail::CoroReplayableValueAnalysis replayable;
    auto value_count = value_domain.size();

    luisa::vector<DenseScopeDataflowResult> scope_data;
    scope_data.reserve(n);
    for (auto &scope : result.scopes) {
        scope_data.emplace_back(
            analyze_scope_use_def(scope, value_domain, replayable));
    }

    luisa::unordered_map<uint32_t, size_t> trigger_to_scope;
    for (size_t i = 0u; i < n; ++i) {
        trigger_to_scope.emplace(result.scopes[i].trigger_token, i);
    }

    // Scopes are rooted reachability regions, not a block partition: a block
    // reached both by a suspend continuation and by a non-suspending bypass
    // may occur in more than one scope. Preserve the established canonical
    // owner (the first root in trigger order) for cross-scope targets while
    // retaining an exact local index for every (scope, block) membership.
    luisa::vector<DensePointerMap<BasicBlock *, size_t>>
        scope_block_indices(n);
    DensePointerMap<BasicBlock *, size_t> canonical_block_scope;
    for (size_t i = 0u; i < n; ++i) {
        scope_block_indices[i].reserve(
            result.scopes[i].blocks.size());
        for (size_t j = 0u;
             j < result.scopes[i].blocks.size(); ++j) {
            auto *block = result.scopes[i].blocks[j];
            scope_block_indices[i].try_emplace(block, j);
            canonical_block_scope.try_emplace(block, i);
        }
    }

    auto append_cross_scope_successor_edges = [&](size_t from, BasicBlock *exit_block, auto visit) noexcept {
        if (exit_block == nullptr || !exit_block->is_terminated()) { return; }
        luisa::vector<uint8_t> seen_targets(n, 0u);
        exit_block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            if (scope_block_indices[from].contains(succ)) {
                return;
            }
            auto iter = canonical_block_scope.find(succ);
            if (iter == canonical_block_scope.end() ||
                iter->second == from) {
                return;
            }
            auto target = iter->second;
            if (seen_targets[target] == 0u) {
                seen_targets[target] = 1u;
                visit(target);
            }
        });
    };

    struct DenseTransitionData {
        DenseValueSet killed;
        DenseValueSet touched;
        DenseValueSet live;
        DenseValueSet store;

        explicit DenseTransitionData(size_t count) noexcept
            : killed{count},
              touched{count},
              live{count},
              store{count} {}
    };

    result.transition_edges.clear();
    luisa::vector<DenseTransitionData> edge_data;
    auto append_transition = [&](size_t from, size_t to,
                                 uint32_t token,
                                 BasicBlock *exit_block,
                                 bool is_suspend) noexcept {
        auto location = scope_block_indices[from].find(exit_block);
        LUISA_ASSERT(
            location != scope_block_indices[from].end(),
            "Coroutine transition exit is not owned by its source scope.");
        auto &edge = result.transition_edges.emplace_back();
        edge.from_scope = from;
        edge.to_scope = to;
        edge.token = token;
        edge.exit_block = exit_block;
        edge.is_suspend = is_suspend;
        auto &dense = edge_data.emplace_back(value_count);
        dense.killed =
            scope_data[from].killed_at_exit[location->second];
        dense.touched =
            scope_data[from].touched_at_exit[location->second];
    };

    for (size_t from = 0u; from < n; ++from) {
        for (auto &sp : result.scopes[from].suspend_points) {
            auto iter = trigger_to_scope.find(sp.token);
            if (iter == trigger_to_scope.end()) { continue; }
            append_transition(
                from, iter->second, sp.token, sp.block, true);
        }
        for (auto *bb : result.scopes[from].blocks) {
            append_cross_scope_successor_edges(from, bb, [&](size_t to) noexcept {
                append_transition(
                    from, to, result.scopes[to].trigger_token,
                    bb, false);
            });
        }
    }

    luisa::vector<luisa::vector<size_t>> outgoing_edges(n);
    luisa::vector<luisa::vector<size_t>> dependent_scopes(n);
    for (size_t edge_index = 0u;
         edge_index < result.transition_edges.size(); ++edge_index) {
        auto &edge = result.transition_edges[edge_index];
        if (edge.from_scope >= n || edge.to_scope >= n) { continue; }
        outgoing_edges[edge.from_scope].emplace_back(edge_index);
        auto &dependents = dependent_scopes[edge.to_scope];
        if (std::find(dependents.begin(), dependents.end(),
                      edge.from_scope) == dependents.end()) {
            dependents.emplace_back(edge.from_scope);
        }
    }

    // This is a backward may analysis over the distilled scope graph:
    //
    //   L_s = E_s union U_(s -> t) (L_t - K_(s -> t)).
    //
    // Starting at E and applying the monotone transfer to a worklist computes
    // the least fixed point, including cyclic sample/bounce schedules. The
    // domain and every edge relation share one value numbering.
    luisa::vector<DenseValueSet> live_begin;
    live_begin.reserve(n);
    for (auto &data : scope_data) {
        live_begin.emplace_back(data.external);
    }
    luisa::deque<size_t> worklist;
    luisa::vector<uint8_t> queued(n, 1u);
    for (size_t i = 0u; i < n; ++i) { worklist.emplace_back(i); }
    auto inter_scope_evaluations = size_t{0u};
    while (!worklist.empty()) {
        auto scope = worklist.front();
        worklist.pop_front();
        queued[scope] = 0u;
        ++inter_scope_evaluations;
        auto next = scope_data[scope].external;
        for (auto edge_index : outgoing_edges[scope]) {
            auto &edge = result.transition_edges[edge_index];
            auto propagated = live_begin[edge.to_scope];
            propagated.subtract(edge_data[edge_index].killed);
            next.union_with(propagated);
        }
        if (!(next == live_begin[scope])) {
            live_begin[scope] = std::move(next);
            for (auto dependent : dependent_scopes[scope]) {
                if (queued[dependent] == 0u) {
                    queued[dependent] = 1u;
                    worklist.emplace_back(dependent);
                }
            }
        }
    }

    luisa::vector<DenseValueSet> live_in(
        n, DenseValueSet{value_count});
    luisa::vector<DenseValueSet> live_out(
        n, DenseValueSet{value_count});
    for (size_t s = 0u; s < n; ++s) {
        live_in[s] = scope_data[s].external;
        for (auto edge_index : outgoing_edges[s]) {
            auto &edge = result.transition_edges[edge_index];
            auto propagated = live_begin[edge.to_scope];
            propagated.subtract(edge_data[edge_index].killed);
            auto reload = propagated;
            reload.intersect_with(scope_data[s].touched);
            live_in[s].union_with(reload);
            auto store = live_begin[edge.to_scope];
            store.intersect_with(edge_data[edge_index].touched);
            edge_data[edge_index].live = live_begin[edge.to_scope];
            edge_data[edge_index].store = std::move(store);
            live_out[s].union_with(edge_data[edge_index].store);
        }
    }

    auto frame_value_set = DenseValueSet{value_count};
    for (size_t i = 0u; i < n; ++i) {
        frame_value_set.union_with(live_begin[i]);
        frame_value_set.union_with(live_in[i]);
        frame_value_set.union_with(live_out[i]);
    }

    struct PlannedFrameAtom {
        size_t atom_index;
        detail::CoroFrameAbiPlan abi;
    };
    luisa::vector<size_t> frame_atoms;
    value_domain.append_indices(frame_atoms, frame_value_set);
    luisa::vector<PlannedFrameAtom> planned_frame_atoms;
    planned_frame_atoms.reserve(frame_atoms.size());
    auto abi_decomposed_atom_count = size_t{0u};
    auto abi_nominal_padding_saved = size_t{0u};
    for (auto atom_index : frame_atoms) {
        auto abi = detail::plan_coro_frame_atom_abi(
            value_domain.atom(atom_index));
        if (abi.decomposed) {
            ++abi_decomposed_atom_count;
            abi_nominal_padding_saved +=
                value_domain.atom(atom_index).type->size() -
                abi.payload_size;
        }
        planned_frame_atoms.emplace_back(PlannedFrameAtom{
            .atom_index = atom_index,
            .abi = std::move(abi)});
    }
    std::stable_sort(
        planned_frame_atoms.begin(), planned_frame_atoms.end(),
        [](auto &lhs, auto &rhs) noexcept {
            if (lhs.abi.max_alignment != rhs.abi.max_alignment) {
                return lhs.abi.max_alignment > rhs.abi.max_alignment;
            }
            return lhs.abi.payload_size > rhs.abi.payload_size;
        });

    result.frame_values.clear();
    result.frame_values.reserve(
        frame_value_set.count_size() + abi_decomposed_atom_count);
    luisa::vector<std::pair<size_t, size_t>> atom_to_frame_value_range(
        value_count, {static_cast<size_t>(-1), 0u});
    luisa::unordered_set<luisa::string> used_names;
    for (auto &planned : planned_frame_atoms) {
        auto &atom = value_domain.atom(planned.atom_index);
        auto first = result.frame_values.size();
        for (auto &field : planned.abi.fields) {
            auto name = frame_value_name(
                atom.root, field.access_chain,
                result.frame_values.size());
            if (!used_names.emplace(name).second) {
                auto base = name;
                auto suffix = result.frame_values.size();
                do {
                    name = luisa::format("{}#{}", base, suffix++);
                } while (!used_names.emplace(name).second);
            }
            result.frame_values.emplace_back(
                CoroCfgDistillResult::FrameValue{
                    .value = atom.root,
                    .access_chain = field.access_chain,
                    .name = std::move(name),
                    .type = field.type,
                    .slot = 0u,
                    .bit_offset = luisa::nullopt,
                });
        }
        atom_to_frame_value_range[planned.atom_index] = {
            first, result.frame_values.size() - first};
    }

    for (size_t i = 0u; i < n; ++i) {
        auto &scope = result.scopes[i];
        append_legacy_values(
            scope.external_values, scope_data[i].external, value_domain);
        append_legacy_values(
            scope.touched_values, scope_data[i].touched, value_domain);
        append_legacy_values(scope.live_in_values, live_in[i], value_domain);
        append_legacy_values(scope.live_out_values, live_out[i], value_domain);
        append_frame_value_indices(
            scope.external_frame_value_indices, scope_data[i].external,
            atom_to_frame_value_range);
        append_frame_value_indices(
            scope.touched_frame_value_indices, scope_data[i].touched,
            atom_to_frame_value_range);
        append_frame_value_indices(
            scope.live_in_frame_value_indices, live_in[i],
            atom_to_frame_value_range);
        append_frame_value_indices(
            scope.live_out_frame_value_indices, live_out[i],
            atom_to_frame_value_range);
        append_names_from_frame_values(
            scope.external_variables, scope.external_frame_value_indices,
            result);
        append_names_from_frame_values(
            scope.touched_variables, scope.touched_frame_value_indices,
            result);
        append_names_from_frame_values(
            scope.live_in_variables, scope.live_in_frame_value_indices,
            result);
        append_names_from_frame_values(
            scope.live_out_variables, scope.live_out_frame_value_indices,
            result);
    }

    for (size_t edge_index = 0u;
         edge_index < result.transition_edges.size(); ++edge_index) {
        auto &edge = result.transition_edges[edge_index];
        auto &dense = edge_data[edge_index];
        append_legacy_values(edge.killed_values, dense.killed, value_domain);
        append_legacy_values(edge.touched_values, dense.touched, value_domain);
        append_legacy_values(edge.live_values, dense.live, value_domain);
        append_legacy_values(edge.store_values, dense.store, value_domain);
        append_frame_value_indices(
            edge.killed_frame_value_indices, dense.killed,
            atom_to_frame_value_range);
        append_frame_value_indices(
            edge.touched_frame_value_indices, dense.touched,
            atom_to_frame_value_range);
        append_frame_value_indices(
            edge.live_frame_value_indices, dense.live,
            atom_to_frame_value_range);
        append_frame_value_indices(
            edge.store_frame_value_indices, dense.store,
            atom_to_frame_value_range);
        append_names_from_frame_values(
            edge.killed_variables, edge.killed_frame_value_indices, result);
        append_names_from_frame_values(
            edge.touched_variables, edge.touched_frame_value_indices, result);
        append_names_from_frame_values(
            edge.live_variables, edge.live_frame_value_indices, result);
        append_names_from_frame_values(
            edge.store_variables, edge.store_frame_value_indices, result);
    }

    color_frame_slots(result);

    if (auto *flag = std::getenv("LUISA_CORO_VERIFY_DENSE_DATAFLOW");
        flag != nullptr && luisa::string_view{flag} == "1") {
        auto to_pointer_set = [&](const DenseValueSet &dense) noexcept {
            luisa::unordered_set<size_t> indices;
            dense.for_each_set_bit([&](size_t index) noexcept {
                indices.emplace(index);
            });
            return indices;
        };
        auto difference = [](const auto &lhs,
                             const auto &rhs) noexcept {
            auto result = luisa::unordered_set<size_t>{};
            for (auto value : lhs) {
                if (!rhs.contains(value)) { result.emplace(value); }
            }
            return result;
        };
        auto append = [](auto &destination,
                         const auto &source) noexcept {
            for (auto value : source) {
                destination.emplace(value);
            }
        };
        auto intersection = [](const auto &lhs,
                               const auto &rhs) noexcept {
            auto result = luisa::unordered_set<size_t>{};
            for (auto value : lhs) {
                if (rhs.contains(value)) { result.emplace(value); }
            }
            return result;
        };

        luisa::vector<luisa::unordered_set<size_t>>
            oracle_external;
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_touched;
        oracle_external.reserve(n);
        oracle_touched.reserve(n);
        for (auto &data : scope_data) {
            oracle_external.emplace_back(
                to_pointer_set(data.external));
            oracle_touched.emplace_back(
                to_pointer_set(data.touched));
        }
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_killed;
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_touched;
        oracle_edge_killed.reserve(edge_data.size());
        oracle_edge_touched.reserve(edge_data.size());
        for (auto &data : edge_data) {
            oracle_edge_killed.emplace_back(
                to_pointer_set(data.killed));
            oracle_edge_touched.emplace_back(
                to_pointer_set(data.touched));
        }

        luisa::vector<luisa::unordered_set<size_t>>
            oracle_live_begin(n);
        for (;;) {
            auto changed = false;
            for (size_t reverse_index = 0u;
                 reverse_index < n; ++reverse_index) {
                auto scope = n - 1u - reverse_index;
                auto next = oracle_external[scope];
                for (auto edge_index : outgoing_edges[scope]) {
                    auto &edge = result.transition_edges[edge_index];
                    auto propagated = difference(
                        oracle_live_begin[edge.to_scope],
                        oracle_edge_killed[edge_index]);
                    append(next, propagated);
                }
                if (!same_set(oracle_live_begin[scope], next)) {
                    oracle_live_begin[scope] = std::move(next);
                    changed = true;
                }
            }
            if (!changed) { break; }
        }

        luisa::vector<luisa::unordered_set<size_t>>
            oracle_live_in(n);
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_live_out(n);
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_live(edge_data.size());
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_store(edge_data.size());
        for (size_t scope = 0u; scope < n; ++scope) {
            oracle_live_in[scope] = oracle_external[scope];
            for (auto edge_index : outgoing_edges[scope]) {
                auto &edge = result.transition_edges[edge_index];
                auto propagated = difference(
                    oracle_live_begin[edge.to_scope],
                    oracle_edge_killed[edge_index]);
                auto reload = intersection(
                    propagated, oracle_touched[scope]);
                append(oracle_live_in[scope], reload);
                oracle_edge_live[edge_index] =
                    oracle_live_begin[edge.to_scope];
                oracle_edge_store[edge_index] = intersection(
                    oracle_live_begin[edge.to_scope],
                    oracle_edge_touched[edge_index]);
                append(oracle_live_out[scope],
                       oracle_edge_store[edge_index]);
            }
        }

        for (size_t scope = 0u; scope < n; ++scope) {
            LUISA_ASSERT(
                same_set(to_pointer_set(live_begin[scope]),
                         oracle_live_begin[scope]) &&
                    same_set(to_pointer_set(live_in[scope]),
                             oracle_live_in[scope]) &&
                    same_set(to_pointer_set(live_out[scope]),
                             oracle_live_out[scope]),
                "Dense inter-scope liveness differs from the pointer oracle "
                "for scope token {}.",
                result.scopes[scope].trigger_token);
        }
        for (size_t edge_index = 0u;
             edge_index < edge_data.size(); ++edge_index) {
            LUISA_ASSERT(
                same_set(to_pointer_set(edge_data[edge_index].live),
                         oracle_edge_live[edge_index]) &&
                    same_set(to_pointer_set(edge_data[edge_index].store),
                             oracle_edge_store[edge_index]),
                "Dense inter-scope edge liveness differs from the pointer "
                "oracle at edge {}.",
                edge_index);
        }
    }

    if (auto *flag = std::getenv("LUISA_CORO_PROFILE_COMPILATION");
        flag != nullptr && luisa::string_view{flag} == "1") {
        auto block_memberships = size_t{0u};
        auto block_evaluations = size_t{0u};
        for (size_t i = 0u; i < n; ++i) {
            block_memberships += result.scopes[i].blocks.size();
            block_evaluations +=
                scope_data[i].fixed_point_block_evaluations;
        }
        luisa::unordered_set<Value *> named_alloca_roots;
        for (auto &value : result.frame_values) {
            if (value.value != nullptr && value.value->isa<AllocaInst>() &&
                static_cast<Instruction *>(value.value)->name().has_value()) {
                named_alloca_roots.emplace(value.value);
            }
        }
        LUISA_INFO(
            "Coroutine dense dataflow: atoms={} words={} scopes={} "
            "block_memberships={} block_evaluations={} transitions={} "
            "scope_evaluations={} replayable_values={} "
            "rejected_replay_values={} logical_frame_values={} "
            "named_frame_alloca_roots={} split_allocas={} split_atoms={} "
            "abi_decomposed_atoms={} abi_nominal_padding_saved={} "
            "physical_frame_slots={}.",
            value_count, (value_count + 63u) / 64u, n,
            block_memberships, block_evaluations,
            result.transition_edges.size(), inter_scope_evaluations,
            replayable.replayable_value_count(),
            replayable.rejected_value_count(),
            result.frame_values.size(), named_alloca_roots.size(),
            value_domain.split_alloca_count(),
            value_domain.split_atom_count(),
            abi_decomposed_atom_count,
            abi_nominal_padding_saved,
            result.frame_slots.size());
    }
    if (auto *flag = std::getenv("LUISA_CORO_DUMP_FRAME_LAYOUT");
        flag != nullptr && luisa::string_view{flag} == "1") {
        for (size_t i = 0u; i < result.scopes.size(); ++i) {
            auto &scope = result.scopes[i];
            LUISA_INFO(
                "Coroutine scope {}: trigger_token={} blocks={} "
                "external_values={} touched_values={} live_in_values={} "
                "live_out_values={} terminal={}.",
                i, scope.trigger_token, scope.blocks.size(),
                scope.external_frame_value_indices.size(),
                scope.touched_frame_value_indices.size(),
                scope.live_in_frame_value_indices.size(),
                scope.live_out_frame_value_indices.size(),
                scope.is_terminal);
        }
        for (size_t i = 0u; i < result.transition_edges.size(); ++i) {
            auto &edge = result.transition_edges[i];
            LUISA_INFO(
                "Coroutine transition edge {}: {} -> {} token={} "
                "suspend={} killed_values={} touched_values={} "
                "live_values={} store_values={}.",
                i, edge.from_scope, edge.to_scope, edge.token,
                edge.is_suspend,
                edge.killed_frame_value_indices.size(),
                edge.touched_frame_value_indices.size(),
                edge.live_frame_value_indices.size(),
                edge.store_frame_value_indices.size());
        }
        for (size_t i = 0u; i < result.frame_values.size(); ++i) {
            auto &value = result.frame_values[i];
            auto append_membership = [](luisa::string &membership,
                                        size_t owner) noexcept {
                if (!membership.empty()) { membership.append(","); }
                membership.append(luisa::format("{}", owner));
            };
            luisa::string scope_external;
            luisa::string scope_touched;
            luisa::string scope_live_in;
            luisa::string scope_live_out;
            for (size_t scope_index = 0u;
                 scope_index < result.scopes.size(); ++scope_index) {
                const auto &scope = result.scopes[scope_index];
                if (std::find(scope.external_frame_value_indices.begin(),
                              scope.external_frame_value_indices.end(), i) !=
                    scope.external_frame_value_indices.end()) {
                    append_membership(scope_external, scope_index);
                }
                if (std::find(scope.touched_frame_value_indices.begin(),
                              scope.touched_frame_value_indices.end(), i) !=
                    scope.touched_frame_value_indices.end()) {
                    append_membership(scope_touched, scope_index);
                }
                if (std::find(scope.live_in_frame_value_indices.begin(),
                              scope.live_in_frame_value_indices.end(), i) !=
                    scope.live_in_frame_value_indices.end()) {
                    append_membership(scope_live_in, scope_index);
                }
                if (std::find(scope.live_out_frame_value_indices.begin(),
                              scope.live_out_frame_value_indices.end(), i) !=
                    scope.live_out_frame_value_indices.end()) {
                    append_membership(scope_live_out, scope_index);
                }
            }
            luisa::string edge_live;
            luisa::string edge_store;
            for (size_t edge_index = 0u;
                 edge_index < result.transition_edges.size(); ++edge_index) {
                const auto &edge = result.transition_edges[edge_index];
                if (std::find(edge.live_frame_value_indices.begin(),
                              edge.live_frame_value_indices.end(), i) !=
                    edge.live_frame_value_indices.end()) {
                    append_membership(edge_live, edge_index);
                }
                if (std::find(edge.store_frame_value_indices.begin(),
                              edge.store_frame_value_indices.end(), i) !=
                    edge.store_frame_value_indices.end()) {
                    append_membership(edge_store, edge_index);
                }
            }
            auto tag = luisa::string_view{"non-instruction"};
            if (value.value != nullptr && value.value->isa<Instruction>()) {
                tag = to_string(static_cast<Instruction *>(value.value)
                                    ->derived_instruction_tag());
            }
            LUISA_INFO(
                "Coroutine logical frame value {}: name='{}' kind={} "
                "path_depth={} type={} size={} align={} physical_slot={} "
                "bit_offset={} scope_external=[{}] scope_touched=[{}] "
                "scope_live_in=[{}] scope_live_out=[{}] edge_live=[{}] "
                "edge_store=[{}].",
                i, value.name, tag, value.access_chain.size(),
                value.type == nullptr ? luisa::string_view{"void"} :
                                        value.type->description(),
                value.type == nullptr ? 0u : value.type->size(),
                value.type == nullptr ? 0u : value.type->alignment(),
                value.slot,
                value.bit_offset ?
                    luisa::format("{}", *value.bit_offset) :
                    luisa::string{"none"},
                scope_external, scope_touched, scope_live_in,
                scope_live_out, edge_live, edge_store);
        }
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

CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(
    Function *f,
    const CoroCfgDistillOptions &options) noexcept {
    CoroCfgDistillResult result;
    if (f == nullptr) {
        result.invalid_input_error_count = 1u;
        return result;
    }
    auto *def = f->definition();
    if (def == nullptr || def->body_block() == nullptr) {
        result.invalid_input_error_count = 1u;
        return result;
    }
    if (xir_pass_has_standalone_verification(
            options.verification_transaction, f)) {
        ++result.boundary_verifier_count;
        auto verification = xir_verify_function(f);
        if (!verification.succeeded()) {
            result.invalid_cfg_error_count = 1u;
            return result;
        }
    }
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
    auto boundary_verifier_count =
        result.boundary_verifier_count;
    result = detail::distill_function(def);
    result.boundary_verifier_count =
        boundary_verifier_count;
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
