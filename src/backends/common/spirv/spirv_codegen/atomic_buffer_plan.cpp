#include "atomic_buffer_plan.h"

#include "structural_closure.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/atomic.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

bool spirv_type_contains_bool(
    const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BOOL: return true;
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
        case Type::Tag::ARRAY:
            return spirv_type_contains_bool(type->element());
        case Type::Tag::STRUCTURE:
            for (auto *member : type->members()) {
                if (spirv_type_contains_bool(member)) { return true; }
            }
            return false;
        default: return false;
    }
}

namespace {

[[nodiscard]] constexpr SpirvTargetFeatures
maximal_float32_buffer_atomic_features() noexcept {
    return SpirvTargetFeatures{
        .shader_buffer_float32_atomics = true,
        .shader_buffer_float32_atomic_add = true,
        .shader_buffer_float32_atomic_min_max = true};
}

}// namespace

SpirvAtomicBufferModulePlan plan_spirv_atomic_buffers(
    luisa::span<const xir::Function *const> functions,
    SpirvAtomicBufferPlanOptions options) noexcept {
    struct PlanningState {
        const Type *buffer_type{nullptr};
        SpirvAtomicBufferStorageRequirements requirements;
        const xir::Function *int64_function{nullptr};
        const xir::Function *float32_word_function{nullptr};
        const xir::AtomicInst *int64_atomic{nullptr};
        const xir::AtomicInst *float32_word_atomic{nullptr};
    };

    SpirvAtomicBufferModulePlan result;
    luisa::vector<PlanningState> states;
    luisa::unordered_map<const Type *, size_t> state_indices;
    auto dialect_features =
        maximal_float32_buffer_atomic_features();
    auto &features = options.target_features == nullptr ?
                         dialect_features :
                         *options.target_features;

    for (auto *function : functions) {
        if (function == nullptr || !function->is_definition()) {
            continue;
        }
        traverse_spirv_codegen_structural_instructions(
            function->definition(),
            [&](const xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::AtomicInst>()) { return; }
                auto *atomic =
                    static_cast<const xir::AtomicInst *>(instruction);
                auto *base = atomic->base();
                if (base == nullptr || base->type() == nullptr ||
                    !base->type()->is_buffer()) {
                    return;
                }
                auto *buffer_type = base->type();
                auto [index_iter, inserted] =
                    state_indices.try_emplace(
                        buffer_type, states.size());
                if (inserted) {
                    states.emplace_back(PlanningState{
                        .buffer_type = buffer_type,
                        .requirements = {
                            .contains_bool = spirv_type_contains_bool(
                                buffer_type->element())}});
                }
                auto &state = states[index_iter->second];
                auto *leaf_type = atomic->type();
                if (leaf_type == nullptr || !leaf_type->is_scalar()) {
                    result.diagnostics.emplace_back(
                        SpirvAtomicBufferPlanDiagnostic{
                            .function = function,
                            .instruction = atomic,
                            .buffer_type = buffer_type,
                            .message = luisa::format(
                                "Native XIR-to-SPIR-V atomic-buffer planning requires a scalar addressed leaf for Buffer<{}>.",
                                buffer_type->element() == nullptr ?
                                    "<null>" :
                                    buffer_type->element()->description())});
                    return;
                }
                if (leaf_type->is_int64() || leaf_type->is_uint64()) {
                    state.requirements.has_int64_atomic = true;
                    state.int64_function = function;
                    state.int64_atomic = atomic;
                } else if (leaf_type->is_float32()) {
                    auto implementation = plan_spirv_float_atomic(
                        atomic->op(), 32u,
                        SpirvFloatAtomicStorage::BUFFER, features);
                    if (!spirv_float_atomic_implementation_is_native(
                            implementation)) {
                        auto capability_implementation =
                            plan_spirv_float_atomic_capability_driven(
                                atomic->op(), 32u,
                                SpirvFloatAtomicStorage::BUFFER,
                                features);
                        if (spirv_float_atomic_implementation_is_native(
                                capability_implementation)) {
                            state.requirements
                                .prefers_float32_word_fallback = true;
                        } else {
                            state.requirements
                                .has_float32_word_fallback = true;
                            state.float32_word_function = function;
                            state.float32_word_atomic = atomic;
                        }
                    }
                }
            });
    }

    result.assignments.reserve(states.size());
    for (auto &&state : states) {
        auto storage = plan_spirv_atomic_buffer_storage(
            state.requirements);
        if (storage != SpirvAtomicBufferStoragePlan::CONFLICT) {
            result.assignments.emplace_back(
                SpirvAtomicBufferAssignment{
                    .buffer_type = state.buffer_type,
                    .storage = storage});
            continue;
        }
        auto *element_type = state.buffer_type->element();
        auto element_description = element_type == nullptr ?
                                       luisa::string_view{"<null>"} :
                                       element_type->description();
        if (state.float32_word_atomic != nullptr) {
            result.diagnostics.emplace_back(
                SpirvAtomicBufferPlanDiagnostic{
                    .function = state.float32_word_function,
                    .instruction = state.float32_word_atomic,
                    .buffer_type = state.buffer_type,
                    .message = luisa::format(
                        "Native XIR-to-SPIR-V cannot represent Buffer<{}> with one SPIR-V Logical pointer type: a 64-bit integer atomic leaf requires typed storage, while float32 {} requires the uint32 word fallback. Enable the exact native float32 atomic feature (compare-exchange has no typed float form), or split these leaves into separate buffers.",
                        element_description,
                        xir::to_string(
                            state.float32_word_atomic->op()))});
        } else {
            result.diagnostics.emplace_back(
                SpirvAtomicBufferPlanDiagnostic{
                    .function = state.int64_function,
                    .instruction = state.int64_atomic,
                    .buffer_type = state.buffer_type,
                    .message = luisa::format(
                        "Native XIR-to-SPIR-V cannot represent Buffer<{}> with one SPIR-V Logical pointer type: a 64-bit integer atomic leaf requires typed storage, while a logical-bool member requires the uint32 word ABI. Split the bool and 64-bit atomic leaves into separate buffers.",
                        element_description)});
        }
    }
    return result;
}

}// namespace lc::spirv
