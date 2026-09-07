#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
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
#include "coro_cfg_dataflow.h"
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

static void hash_coro_suspend_extension(
    DistillCertificateHasher &h,
    const CoroSuspendExtension *extension) noexcept {
    h.add(extension != nullptr);
    if (extension == nullptr) { return; }
    h.add_string(extension->schema());
    h.add(extension->version());
    h.add(extension->is_annotation());
    h.add(extension->fallback());
    h.add(extension->bindings().size());
    for (auto &&binding : extension->bindings()) {
        h.add_string(binding.name);
        h.add(binding.access);
        h.add(binding.lifetime);
        h.add(binding.index);
    }
    h.add(extension->attributes().size());
    for (auto &&attribute : extension->attributes()) {
        h.add_string(attribute.name);
        h.add(attribute.value.index());
        luisa::visit(
            [&](auto &&value) noexcept {
                using T = std::remove_cvref_t<decltype(value)>;
                if constexpr (std::is_same_v<T, luisa::string>) {
                    h.add_string(value);
                } else if constexpr (std::is_same_v<T, double>) {
                    h.add(std::bit_cast<uint64_t>(value));
                } else {
                    h.add(value);
                }
            },
            attribute.value);
    }
}

static void hash_coro_suspend_extension_owner(
    DistillCertificateHasher &h,
    const CoroSuspendExtensionOwner &owner) noexcept {
    h.add(owner.extensions.size());
    for (auto &&extension : owner.extensions) {
        hash_coro_suspend_extension(h, extension.get());
    }
    h.add(owner.binding_values.size());
    for (auto *value : owner.binding_values) {
        h.add_pointer(value);
    }
}

[[nodiscard]] static CoroSuspendExtensionOwner
clone_coro_suspend_extension_owner(
    CoroSuspendInst *suspend) noexcept {
    CoroSuspendExtensionOwner owner;
    if (suspend == nullptr) { return owner; }
    owner.extensions.reserve(suspend->extensions().size());
    for (auto &&extension : suspend->extensions()) {
        owner.extensions.emplace_back(
            extension == nullptr ? nullptr : extension->clone());
    }
    owner.binding_values.reserve(
        suspend->extension_binding_value_count());
    for (size_t i = 0u;
         i < suspend->extension_binding_value_count(); ++i) {
        owner.binding_values.emplace_back(
            suspend->extension_binding_value(i));
    }
    return owner;
}

[[nodiscard]] static luisa::optional<luisa::vector<uint32_t>>
resolve_static_local_lvalue_access_chain(Value *value) noexcept {
    luisa::vector<luisa::vector<uint32_t>> reversed_segments;
    while (value != nullptr && value->isa<GEPInst>()) {
        auto *gep = static_cast<GEPInst *>(value);
        auto &segment = reversed_segments.emplace_back();
        segment.reserve(gep->index_count());
        for (size_t i = 0u; i < gep->index_count(); ++i) {
            uint64_t index = 0u;
            if (!try_decode_constant_nonnegative_integer(
                    gep->index(i), index) ||
                index > std::numeric_limits<uint32_t>::max()) {
                return luisa::nullopt;
            }
            segment.emplace_back(static_cast<uint32_t>(index));
        }
        value = gep->base();
    }
    if (value == nullptr || !value->isa<AllocaInst>() ||
        !static_cast<AllocaInst *>(value)->is_local()) {
        return luisa::nullopt;
    }
    luisa::vector<uint32_t> access_chain;
    for (auto iter = reversed_segments.rbegin();
         iter != reversed_segments.rend(); ++iter) {
        access_chain.insert(
            access_chain.end(), iter->begin(), iter->end());
    }
    return access_chain;
}

[[nodiscard]] static uint64_t compute_distill_validation_hash(
    const CoroCfgDistillResult &result,
    const FunctionDefinition *definition) noexcept {
    DistillCertificateHasher h;
    // Version the schema so adding a semantic field cannot silently retain a
    // certificate computed by an older layout.
    h.add(uint64_t{7u});
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
                        h.add(suspend->frame_export_count());
                        for (auto &name :
                             suspend->frame_export_names()) {
                            h.add_string(name);
                        }
                        h.add(suspend->extensions().size());
                        for (auto &&extension : suspend->extensions()) {
                            hash_coro_suspend_extension(
                                h, extension.get());
                        }
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
            hash_coro_suspend_extension_owner(
                h, point.extension_owner);
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
        hash_coro_suspend_extension_owner(
            h, edge.extension_owner);
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
        h.add(edge.extension_binding_frame_value_indices.size());
        for (auto &&binding_indices :
             edge.extension_binding_frame_value_indices) {
            hash_indices(binding_indices);
        }
        h.add(edge.extension_binding_access_chains.size());
        for (auto &&access_chain :
             edge.extension_binding_access_chains) {
            hash_indices(access_chain);
        }
        hash_indices(edge.target_live_frame_value_indices);
        h.add(edge.extension_stage_dataflow.size());
        for (auto &stage : edge.extension_stage_dataflow) {
            hash_indices(stage.use_frame_value_indices);
            hash_indices(stage.def_frame_value_indices);
            hash_indices(stage.live_in_frame_value_indices);
            hash_indices(stage.live_out_frame_value_indices);
        }
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
        h.add(frame_value.aliases.size());
        for (auto &alias : frame_value.aliases) {
            h.add_string(alias);
        }
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
    luisa::unordered_map<luisa::string, const Value *>
        designated_values;
    for (auto *block : def->basic_blocks()) {
        auto resume_count = 0u;
        for (auto *inst : block->instructions()) {
            if (inst->isa<PhiInst>()) { return false; }
            if (inst->isa<CoroResumeInst>()) { ++resume_count; }
            if (inst->isa<CoroSuspendInst>()) {
                auto *suspend =
                    static_cast<CoroSuspendInst *>(inst);
                luisa::unordered_set<luisa::string> local_names;
                for (size_t i = 0u;
                     i < suspend->frame_export_count(); ++i) {
                    auto &name = suspend->frame_export_name(i);
                    auto *value = suspend->frame_export_value(i);
                    if (name.empty() ||
                        !local_names.emplace(name).second ||
                        value == nullptr || value->type() == nullptr ||
                        value->is_lvalue() ||
                        !value->type()->is_basic()) {
                        return false;
                    }
                    auto inserted =
                        designated_values.try_emplace(name, value).second;
                    if (!inserted) {
                        // An ABI alias is declared exactly once. Reusing it at
                        // another boundary would make its validity interval
                        // path-dependent and scheduler reads ambiguous.
                        return false;
                    }
                }
            }
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
[[nodiscard]] static bool distill_same_set(const luisa::unordered_set<T> &a,
                                           const luisa::unordered_set<T> &b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (auto &v : a) {
        if (!b.contains(v)) { return false; }
    }
    return true;
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

static void analyze_live_variables(
    CoroCfgDistillResult &result, FunctionDefinition *def,
    CoroCfgDistillStats *stats) noexcept {
    auto n = result.scopes.size();
    // Collect the semantic scheduler ABI before constructing the immutable
    // atom domain. A designated value may otherwise be omitted as replayable
    // or always available even though the host needs a concrete frame field.
    luisa::vector<Value *> designated_values;
    luisa::vector<Value *> legacy_designated_values;
    luisa::unordered_map<luisa::string, Value *>
        designated_values_by_name;
    luisa::unordered_map<Value *, luisa::vector<luisa::string>>
        designated_aliases_by_value;
    for (auto *block : def->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<CoroSuspendInst>()) { continue; }
            auto *suspend =
                static_cast<CoroSuspendInst *>(instruction);
            for (size_t i = 0u;
                 i < suspend->frame_export_count(); ++i) {
                auto *value = suspend->frame_export_value(i);
                auto &name = suspend->frame_export_name(i);
                auto name_inserted =
                    designated_values_by_name.try_emplace(name, value)
                        .second;
                LUISA_ASSERT(
                    name_inserted,
                    "Validated coroutine designated-value alias '{}' "
                    "appeared more than once.",
                    name);
                auto [alias_iter, value_inserted] =
                    designated_aliases_by_value.try_emplace(value);
                if (value_inserted) {
                    designated_values.emplace_back(value);
                    legacy_designated_values.emplace_back(value);
                }
                if (std::find(alias_iter->second.begin(),
                              alias_iter->second.end(), name) ==
                    alias_iter->second.end()) {
                    alias_iter->second.emplace_back(name);
                }
            }
            // Queued and resumed read operands need a concrete snapshot even
            // when ordinary coroutine analysis would replay the expression or
            // treat it as always available. Add the exact XIR value to the
            // same immutable atom domain used by all other continuation data;
            // repeated bindings and legacy exports intentionally collapse to
            // one atom here.
            for (auto &&extension : suspend->extensions()) {
                for (auto &&binding : extension->bindings()) {
                    if (binding.lifetime ==
                            CoroSuspendBindingLifetime::boundary ||
                        binding.access !=
                            CoroSuspendBindingAccess::read) {
                        continue;
                    }
                    auto *value = suspend->extension_binding_value(
                        binding.index);
                    if (std::find(designated_values.begin(),
                                  designated_values.end(), value) ==
                        designated_values.end()) {
                        designated_values.emplace_back(value);
                    }
                }
            }
        }
    }
    DenseValueDomain value_domain{def,
                                  luisa::span{designated_values}};
    detail::CoroReplayableValueAnalysis replayable;
    auto value_count = value_domain.size();
    auto designated_atoms = DenseValueSet{value_count};
    luisa::unordered_map<size_t, luisa::vector<luisa::string>>
        designated_aliases_by_atom;
    for (auto *value : legacy_designated_values) {
        auto atom_index = value_domain.ssa_index(value);
        LUISA_ASSERT(
            atom_index.has_value(),
            "Validated coroutine designated value has no frame atom.");
        designated_atoms.set(*atom_index);
        designated_aliases_by_atom.emplace(
            *atom_index, designated_aliases_by_value.at(value));
    }

    luisa::vector<DenseScopeDataflowResult> scope_data;
    scope_data.reserve(n);
    for (auto &scope : result.scopes) {
        scope_data.emplace_back(
            analyze_scope_use_def(scope, value_domain, replayable));
    }
    // Intra-scope fixed points use compact local coordinates. All relations
    // crossing a scope boundary are embedded once into the immutable global
    // atom numbering, so the rest of the analysis remains unchanged.
    luisa::vector<DenseValueSet> scope_external;
    luisa::vector<DenseValueSet> scope_touched;
    scope_external.reserve(n);
    scope_touched.reserve(n);
    for (auto &data : scope_data) {
        scope_external.emplace_back(
            data.expand_to_global(data.external));
        scope_touched.emplace_back(
            data.expand_to_global(data.touched));
    }
    if (stats != nullptr) {
        stats->value_atom_count = value_count;
        stats->scope_count = n;
        for (size_t i = 0u; i < n; ++i) {
            auto local_count = scope_data[i].local_value_count();
            stats->projected_scope_atom_count += local_count;
            stats->max_projected_scope_atom_count = std::max(
                stats->max_projected_scope_atom_count, local_count);
            stats->block_membership_count +=
                result.scopes[i].blocks.size();
            stats->must_block_evaluation_count +=
                scope_data[i].must_block_evaluations;
            stats->may_block_evaluation_count +=
                scope_data[i].may_block_evaluations;
        }
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
        struct ExtensionStage {
            DenseValueSet use;
            DenseValueSet def;
            DenseValueSet live_in;
            DenseValueSet live_out;

            explicit ExtensionStage(size_t count) noexcept
                : use{count}, def{count}, live_in{count},
                  live_out{count} {}
        };

        DenseValueSet killed;
        // Definitions made by the source continuation before reaching the
        // suspension site. Extension definitions are intentionally kept out
        // of this set: the ordered stage transfer accounts for them.
        DenseValueSet source_killed;
        DenseValueSet touched;
        DenseValueSet designated;
        DenseValueSet extension_live;
        DenseValueSet live;
        DenseValueSet store;
        // Owner binding index -> existing global atom indices. These are
        // projections into the one DenseValueDomain above, never a second
        // extension-specific frame namespace.
        luisa::vector<luisa::vector<size_t>> extension_binding_atoms;
        luisa::vector<luisa::vector<uint32_t>>
            extension_binding_access_chains;
        luisa::vector<ExtensionStage> extension_stages;

        explicit DenseTransitionData(size_t count) noexcept
            : killed{count},
              source_killed{count},
              touched{count},
              designated{count},
              extension_live{count},
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
        dense.source_killed = scope_data[from].expand_to_global(
            scope_data[from].killed_at_exit[location->second]);
        dense.killed = dense.source_killed;
        dense.touched = scope_data[from].expand_to_global(
            scope_data[from].touched_at_exit[location->second]);
        if (is_suspend) {
            auto *suspend = static_cast<CoroSuspendInst *>(
                exit_block->terminator());
            edge.extension_owner =
                clone_coro_suspend_extension_owner(suspend);
            dense.extension_binding_atoms.resize(
                suspend->extension_binding_value_count());
            dense.extension_binding_access_chains.resize(
                suspend->extension_binding_value_count());
            for (size_t i = 0u;
                 i < suspend->frame_export_count(); ++i) {
                auto atom_index = value_domain.ssa_index(
                    suspend->frame_export_value(i));
                LUISA_ASSERT(
                    atom_index.has_value(),
                    "Validated coroutine designated value has no frame atom.");
                dense.designated.set(*atom_index);
            }
            dense.extension_stages.reserve(
                suspend->extensions().size());
            for (auto &&extension : suspend->extensions()) {
                auto &stage =
                    dense.extension_stages.emplace_back(value_count);
                for (auto &&binding : extension->bindings()) {
                    if (binding.lifetime ==
                        CoroSuspendBindingLifetime::boundary) {
                        continue;
                    }
                    auto *value = suspend->extension_binding_value(
                        binding.index);
                    auto &binding_atoms =
                        dense.extension_binding_atoms[binding.index];
                    auto append_atom = [&](size_t atom_index) noexcept {
                        if (std::find(binding_atoms.begin(),
                                      binding_atoms.end(), atom_index) ==
                            binding_atoms.end()) {
                            binding_atoms.emplace_back(atom_index);
                        }
                    };
                    if (binding.access ==
                        CoroSuspendBindingAccess::read) {
                        if (auto atom_index =
                                value_domain.ssa_index(value)) {
                            append_atom(*atom_index);
                        }
                    } else {
                        auto access_chain =
                            resolve_static_local_lvalue_access_chain(value);
                        if (!access_chain) {
                            ++result.invalid_cfg_error_count;
                            continue;
                        }
                        dense.extension_binding_access_chains[binding.index] = std::move(*access_chain);
                        for (auto access :
                             value_domain.memory_accesses(value)) {
                            append_atom(access.atom_index);
                        }
                    }
                    if (binding_atoms.empty()) {
                        // A queued/resumed operand must have a stable frame
                        // representation. Non-local lvalues, resources, and
                        // other non-materializable operands are rejected here
                        // instead of acquiring an ad-hoc extension slot.
                        ++result.invalid_cfg_error_count;
                        continue;
                    }
                    for (auto atom_index : binding_atoms) {
                        dense.extension_live.set(atom_index);
                        switch (binding.access) {
                            case CoroSuspendBindingAccess::read:
                                stage.use.set(atom_index);
                                break;
                            case CoroSuspendBindingAccess::write:
                                stage.def.set(atom_index);
                                dense.killed.set(atom_index);
                                break;
                            case CoroSuspendBindingAccess::read_write:
                                stage.use.set(atom_index);
                                stage.def.set(atom_index);
                                dense.killed.set(atom_index);
                                break;
                        }
                    }
                }
            }
        }
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
    if (result.invalid_cfg_error_count != 0u) { return; }

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

    // A scheduler observes an exported value by continuation token, not by
    // the source edge that produced the frame. Therefore its validity is a
    // must property over all transitions entering that token:
    //
    //   D(t) = D(e)  for every edge e whose target is t.
    //
    // A suspension plus an ordinary cross-scope bypass into the same token
    // would otherwise make the field initialized only on one path. Reject
    // that partial ABI instead of widening liveness, inventing a value for
    // the bypass, or leaving scheduler reads path-dependent. Repeated dynamic
    // executions of one static exported suspend (including self-edges) carry
    // the same designated set and remain valid.
    luisa::vector<luisa::optional<DenseValueSet>>
        target_designated(n);
    for (size_t edge_index = 0u;
         edge_index < result.transition_edges.size(); ++edge_index) {
        auto target = result.transition_edges[edge_index].to_scope;
        if (target >= n) { continue; }
        auto &expected = target_designated[target];
        if (!expected) {
            expected.emplace(edge_data[edge_index].designated);
        } else if (!(*expected == edge_data[edge_index].designated)) {
            ++result.invalid_cfg_error_count;
            return;
        }
    }

    auto transfer_extension_stages_backward =
        [&](size_t edge_index,
            const DenseValueSet &target_live) noexcept {
            auto live = target_live;
            auto &stages = edge_data[edge_index].extension_stages;
            for (size_t reverse = 0u;
                 reverse < stages.size(); ++reverse) {
                auto index = stages.size() - 1u - reverse;
                live.subtract(stages[index].def);
                live.union_with(stages[index].use);
            }
            return live;
        };

    // This is a backward may analysis over the distilled scope graph. Every
    // suspend edge first applies its ordered external-stage transfer:
    //
    //   X_e = F_extensions(L_t)
    //   L_s = E_s union U_(s -> t) (X_e - K_source_(s -> t)).
    //
    // Starting at E and applying the monotone transfer to a worklist computes
    // the least fixed point, including cyclic sample/bounce schedules. The
    // domain and every edge relation share one value numbering.
    luisa::vector<DenseValueSet> live_begin;
    live_begin.reserve(n);
    for (auto &external : scope_external) {
        live_begin.emplace_back(external);
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
        auto next = scope_external[scope];
        for (auto edge_index : outgoing_edges[scope]) {
            auto &edge = result.transition_edges[edge_index];
            auto propagated = transfer_extension_stages_backward(
                edge_index, live_begin[edge.to_scope]);
            propagated.subtract(
                edge_data[edge_index].source_killed);
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

    // Refine every static suspend into an ordered external-stage liveness
    // chain. The target's complete resident set is the terminal condition;
    // it is intentionally larger than the immediate continuation reload set.
    //
    //   live_in_i = use_i union (live_out_i - def_i)
    //   live_out_i = live_in_(i + 1)
    //
    // This certificate lets graph consumers reconstruct only the partial
    // frame needed by one stage while preserving dormant state in backing
    // storage.
    for (size_t edge_index = 0u;
         edge_index < result.transition_edges.size(); ++edge_index) {
        auto &edge = result.transition_edges[edge_index];
        auto &dense = edge_data[edge_index];
        if (!edge.is_suspend) { continue; }
        auto next = live_begin[edge.to_scope];
        for (size_t reverse = 0u;
             reverse < dense.extension_stages.size(); ++reverse) {
            auto index = dense.extension_stages.size() - 1u - reverse;
            auto &stage = dense.extension_stages[index];
            stage.live_out = next;
            stage.live_in = next;
            stage.live_in.subtract(stage.def);
            stage.live_in.union_with(stage.use);
            next = stage.live_in;
        }
    }

    luisa::vector<DenseValueSet> live_in(
        n, DenseValueSet{value_count});
    luisa::vector<DenseValueSet> live_out(
        n, DenseValueSet{value_count});
    for (size_t s = 0u; s < n; ++s) {
        live_in[s] = scope_external[s];
        for (auto edge_index : outgoing_edges[s]) {
            auto &edge = result.transition_edges[edge_index];
            auto edge_live_in = transfer_extension_stages_backward(
                edge_index, live_begin[edge.to_scope]);
            auto propagated = edge_live_in;
            propagated.subtract(
                edge_data[edge_index].source_killed);
            auto reload = propagated;
            reload.intersect_with(scope_touched[s]);
            live_in[s].union_with(reload);
            auto store = edge_live_in;
            store.intersect_with(edge_data[edge_index].touched);
            // Extension operands that reach the boundary before any earlier
            // external definition must be snapshotted even when they are
            // source-callable arguments or replayable values and therefore do
            // not appear in the ordinary touched set. Applying the same stage
            // transfer to an empty terminal set selects exactly those inputs;
            // values produced by an earlier Extension are excluded.
            auto extension_input = transfer_extension_stages_backward(
                edge_index, DenseValueSet{value_count});
            store.union_with(extension_input);
            edge_data[edge_index].live = live_begin[edge.to_scope];
            edge_data[edge_index].live.union_with(
                edge_data[edge_index].designated);
            edge_data[edge_index].live.union_with(
                edge_data[edge_index].extension_live);
            store.union_with(edge_data[edge_index].designated);
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
    frame_value_set.union_with(designated_atoms);
    for (auto &dense : edge_data) {
        frame_value_set.union_with(dense.extension_live);
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
    for (auto &[atom_index, aliases] :
         designated_aliases_by_atom) {
        static_cast<void>(atom_index);
        for (auto &alias : aliases) {
            auto inserted = used_names.emplace(alias).second;
            LUISA_ASSERT(
                inserted,
                "Validated coroutine designated alias '{}' is duplicated.",
                alias);
        }
    }
    for (auto &planned : planned_frame_atoms) {
        auto &atom = value_domain.atom(planned.atom_index);
        auto designated_alias_iter =
            designated_aliases_by_atom.find(planned.atom_index);
        auto is_designated =
            designated_alias_iter != designated_aliases_by_atom.end();
        if (is_designated && planned.abi.fields.size() != 1u) {
            ++result.invalid_cfg_error_count;
            return;
        }
        auto first = result.frame_values.size();
        for (auto &field : planned.abi.fields) {
            auto aliases = luisa::vector<luisa::string>{};
            if (is_designated) {
                aliases = designated_alias_iter->second;
            }
            auto name = !aliases.empty() ?
                            aliases.front() :
                            frame_value_name(
                                atom.root, field.access_chain,
                                result.frame_values.size());
            if (!is_designated &&
                !used_names.emplace(name).second) {
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
                    .aliases = std::move(aliases),
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
            scope.external_values, scope_external[i], value_domain);
        append_legacy_values(
            scope.touched_values, scope_touched[i], value_domain);
        append_legacy_values(scope.live_in_values, live_in[i], value_domain);
        append_legacy_values(scope.live_out_values, live_out[i], value_domain);
        append_frame_value_indices(
            scope.external_frame_value_indices, scope_external[i],
            atom_to_frame_value_range);
        append_frame_value_indices(
            scope.touched_frame_value_indices, scope_touched[i],
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
        append_frame_value_indices(
            edge.target_live_frame_value_indices,
            live_begin[edge.to_scope], atom_to_frame_value_range);
        auto normalize_frame_indices = [](auto &indices) noexcept {
            std::sort(indices.begin(), indices.end());
            indices.erase(
                std::unique(indices.begin(), indices.end()),
                indices.end());
        };
        normalize_frame_indices(
            edge.target_live_frame_value_indices);
        edge.extension_binding_frame_value_indices.clear();
        edge.extension_binding_frame_value_indices.resize(
            dense.extension_binding_atoms.size());
        edge.extension_binding_access_chains =
            dense.extension_binding_access_chains;
        for (size_t binding_index = 0u;
             binding_index < dense.extension_binding_atoms.size();
             ++binding_index) {
            auto &projection =
                edge.extension_binding_frame_value_indices[binding_index];
            for (auto atom_index :
                 dense.extension_binding_atoms[binding_index]) {
                LUISA_DEBUG_ASSERT(
                    atom_index < atom_to_frame_value_range.size(),
                    "Coroutine extension binding atom is out of range.");
                auto [first, count] =
                    atom_to_frame_value_range[atom_index];
                LUISA_DEBUG_ASSERT(
                    first != static_cast<size_t>(-1),
                    "Coroutine extension binding atom was not materialized.");
                for (size_t i = 0u; i < count; ++i) {
                    auto frame_value_index = first + i;
                    auto &frame_value =
                        result.frame_values[frame_value_index];
                    auto *binding_value =
                        edge.extension_owner.binding_values[binding_index];
                    auto &base =
                        dense.extension_binding_access_chains[binding_index];
                    auto projection_matches_binding =
                        !binding_value->is_lvalue() ?
                            frame_value.value == binding_value :
                            frame_value.value ==
                                    detail::trace_local_alloca(binding_value) &&
                                base.size() <=
                                    frame_value.access_chain.size() &&
                                std::equal(
                                    base.begin(), base.end(),
                                    frame_value.access_chain.begin());
                    if (!projection_matches_binding) {
                        ++result.invalid_cfg_error_count;
                        continue;
                    }
                    projection.emplace_back(frame_value_index);
                }
            }
            std::sort(projection.begin(), projection.end());
            projection.erase(
                std::unique(projection.begin(), projection.end()),
                projection.end());
        }
        edge.extension_stage_dataflow.clear();
        edge.extension_stage_dataflow.reserve(
            dense.extension_stages.size());
        for (auto &dense_stage : dense.extension_stages) {
            auto &stage =
                edge.extension_stage_dataflow.emplace_back();
            append_frame_value_indices(
                stage.use_frame_value_indices,
                dense_stage.use, atom_to_frame_value_range);
            append_frame_value_indices(
                stage.def_frame_value_indices,
                dense_stage.def, atom_to_frame_value_range);
            append_frame_value_indices(
                stage.live_in_frame_value_indices,
                dense_stage.live_in, atom_to_frame_value_range);
            append_frame_value_indices(
                stage.live_out_frame_value_indices,
                dense_stage.live_out, atom_to_frame_value_range);
            normalize_frame_indices(stage.use_frame_value_indices);
            normalize_frame_indices(stage.def_frame_value_indices);
            normalize_frame_indices(stage.live_in_frame_value_indices);
            normalize_frame_indices(stage.live_out_frame_value_indices);
        }
        append_names_from_frame_values(
            edge.killed_variables, edge.killed_frame_value_indices, result);
        append_names_from_frame_values(
            edge.touched_variables, edge.touched_frame_value_indices, result);
        append_names_from_frame_values(
            edge.live_variables, edge.live_frame_value_indices, result);
        append_names_from_frame_values(
            edge.store_variables, edge.store_frame_value_indices, result);
    }

    if (result.invalid_cfg_error_count != 0u) { return; }

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
        for (size_t i = 0u; i < n; ++i) {
            oracle_external.emplace_back(
                to_pointer_set(scope_external[i]));
            oracle_touched.emplace_back(
                to_pointer_set(scope_touched[i]));
        }
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_source_killed;
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_touched;
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_designated;
        luisa::vector<luisa::unordered_set<size_t>>
            oracle_edge_extension_live;
        struct OracleExtensionStage {
            luisa::unordered_set<size_t> use;
            luisa::unordered_set<size_t> def;
        };
        luisa::vector<luisa::vector<OracleExtensionStage>>
            oracle_edge_extension_stages;
        oracle_edge_source_killed.reserve(edge_data.size());
        oracle_edge_touched.reserve(edge_data.size());
        oracle_edge_designated.reserve(edge_data.size());
        oracle_edge_extension_live.reserve(edge_data.size());
        oracle_edge_extension_stages.reserve(edge_data.size());
        for (auto &data : edge_data) {
            oracle_edge_source_killed.emplace_back(
                to_pointer_set(data.source_killed));
            oracle_edge_touched.emplace_back(
                to_pointer_set(data.touched));
            oracle_edge_designated.emplace_back(
                to_pointer_set(data.designated));
            oracle_edge_extension_live.emplace_back(
                to_pointer_set(data.extension_live));
            auto &stages =
                oracle_edge_extension_stages.emplace_back();
            stages.reserve(data.extension_stages.size());
            for (auto &stage : data.extension_stages) {
                stages.emplace_back(OracleExtensionStage{
                    .use = to_pointer_set(stage.use),
                    .def = to_pointer_set(stage.def)});
            }
        }

        auto transfer_oracle_extension_stages_backward =
            [&](size_t edge_index, const auto &target_live) noexcept {
                auto live = target_live;
                auto &stages =
                    oracle_edge_extension_stages[edge_index];
                for (size_t reverse = 0u;
                     reverse < stages.size(); ++reverse) {
                    auto index = stages.size() - 1u - reverse;
                    live = difference(live, stages[index].def);
                    append(live, stages[index].use);
                }
                return live;
            };

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
                    auto edge_live_in =
                        transfer_oracle_extension_stages_backward(
                            edge_index,
                            oracle_live_begin[edge.to_scope]);
                    auto propagated = difference(
                        edge_live_in,
                        oracle_edge_source_killed[edge_index]);
                    append(next, propagated);
                }
                if (!distill_same_set(oracle_live_begin[scope], next)) {
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
                auto edge_live_in =
                    transfer_oracle_extension_stages_backward(
                        edge_index,
                        oracle_live_begin[edge.to_scope]);
                auto propagated = difference(
                    edge_live_in,
                    oracle_edge_source_killed[edge_index]);
                auto reload = intersection(
                    propagated, oracle_touched[scope]);
                append(oracle_live_in[scope], reload);
                oracle_edge_live[edge_index] =
                    oracle_live_begin[edge.to_scope];
                append(oracle_edge_live[edge_index],
                       oracle_edge_designated[edge_index]);
                append(oracle_edge_live[edge_index],
                       oracle_edge_extension_live[edge_index]);
                oracle_edge_store[edge_index] = intersection(
                    edge_live_in,
                    oracle_edge_touched[edge_index]);
                auto extension_input =
                    transfer_oracle_extension_stages_backward(
                        edge_index,
                        luisa::unordered_set<size_t>{});
                append(oracle_edge_store[edge_index],
                       extension_input);
                append(oracle_edge_store[edge_index],
                       oracle_edge_designated[edge_index]);
                append(oracle_live_out[scope],
                       oracle_edge_store[edge_index]);
            }
        }

        for (size_t scope = 0u; scope < n; ++scope) {
            LUISA_ASSERT(
                distill_same_set(to_pointer_set(live_begin[scope]),
                                 oracle_live_begin[scope]) &&
                    distill_same_set(to_pointer_set(live_in[scope]),
                                     oracle_live_in[scope]) &&
                    distill_same_set(to_pointer_set(live_out[scope]),
                                     oracle_live_out[scope]),
                "Dense inter-scope liveness differs from the pointer oracle "
                "for scope token {}.",
                result.scopes[scope].trigger_token);
        }
        for (size_t edge_index = 0u;
             edge_index < edge_data.size(); ++edge_index) {
            LUISA_ASSERT(
                distill_same_set(to_pointer_set(edge_data[edge_index].live),
                                 oracle_edge_live[edge_index]) &&
                    distill_same_set(to_pointer_set(edge_data[edge_index].store),
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
        auto projected_atoms = size_t{0u};
        auto projected_words = size_t{0u};
        auto max_scope_atoms = size_t{0u};
        for (size_t i = 0u; i < n; ++i) {
            block_memberships += result.scopes[i].blocks.size();
            block_evaluations +=
                scope_data[i].fixed_point_block_evaluations();
            auto local_count = scope_data[i].local_value_count();
            projected_atoms += local_count;
            projected_words += (local_count + 63u) / 64u;
            max_scope_atoms = std::max(max_scope_atoms, local_count);
            LUISA_INFO(
                "Coroutine dense scope: index={} token={} atoms={} words={} "
                "blocks={} must_evaluations={} may_evaluations={}.",
                i, result.scopes[i].trigger_token, local_count,
                (local_count + 63u) / 64u,
                result.scopes[i].blocks.size(),
                scope_data[i].must_block_evaluations,
                scope_data[i].may_block_evaluations);
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
            "projected_atoms={} projected_words={} max_scope_atoms={} "
            "block_memberships={} block_evaluations={} transitions={} "
            "scope_evaluations={} replayable_values={} "
            "rejected_replay_values={} logical_frame_values={} "
            "named_frame_alloca_roots={} split_allocas={} split_atoms={} "
            "abi_decomposed_atoms={} abi_nominal_padding_saved={} "
            "physical_frame_slots={}.",
            value_count, (value_count + 63u) / 64u, n,
            projected_atoms, projected_words, max_scope_atoms,
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

[[nodiscard]] static CoroCfgDistillResult distill_function(
    FunctionDefinition *def, CoroCfgDistillStats *stats) noexcept {

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
                    auto &point = scope.suspend_points.emplace_back();
                    point.block = bb;
                    point.token = s->token();
                    point.name = s->name();
                    point.extension_owner =
                        clone_coro_suspend_extension_owner(s);
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

    analyze_live_variables(result, def, stats);
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
    if (options.stats != nullptr) {
        *options.stats = {};
    }
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
    result = detail::distill_function(def, options.stats);
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
