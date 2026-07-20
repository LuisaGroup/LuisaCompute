#include "runtime_target_plan.h"

#include "argument_usage.h"
#include "structural_closure.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/thread_group.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] bool thread_group_op_uses_value_type(
    xir::ThreadGroupOp op) noexcept {
    switch (op) {
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
        case xir::ThreadGroupOp::WARP_PREFIX_SUM:
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
        case xir::ThreadGroupOp::WARP_READ_LANE:
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
            return true;
        default: return false;
    }
}

}// namespace

bool spirv_subgroup_type_requires_extended_types(
    const Type *type) noexcept {
    while (type != nullptr &&
           (type->is_vector() || type->is_matrix())) {
        type = type->element();
    }
    return type != nullptr &&
           (type->is_int8() || type->is_uint8() ||
            type->is_int16() || type->is_uint16() ||
            type->is_int64() || type->is_uint64() ||
            type->is_float16());
}

SpirvRuntimeTargetPlanResult plan_spirv_runtime_target_contract(
    luisa::span<const xir::Function *const> functions,
    SpirvBindlessResourceUsage xir_bindless,
    const SpirvTargetFeatures &features) noexcept {
    SpirvRuntimeTargetPlanResult result;
    result.plan.bindless_resources = xir_bindless;

    const xir::Function *ray_function = nullptr;
    const xir::Instruction *ray_instruction = nullptr;
    const xir::Function *subgroup_function = nullptr;
    const xir::Instruction *subgroup_instruction = nullptr;
    const xir::Function *clock_function = nullptr;
    const xir::Instruction *clock_instruction = nullptr;
    const xir::Function *device_address_function = nullptr;
    const xir::Instruction *device_address_instruction = nullptr;
    for (auto *function : functions) {
        if (function == nullptr || !function->is_definition()) { continue; }
        traverse_spirv_codegen_structural_instructions(
            function->definition(),
            [&](const xir::Instruction *instruction) noexcept {
                if (instruction == nullptr) { return; }
                if (!result.plan.uses_semantic_ray_query) {
                    auto tag = instruction->derived_instruction_tag();
                    auto is_ray_query_instruction =
                        tag == xir::DerivedInstructionTag::RAY_QUERY_LOOP ||
                        tag == xir::DerivedInstructionTag::RAY_QUERY_DISPATCH ||
                        tag == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ ||
                        tag == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE ||
                        tag == xir::DerivedInstructionTag::RAY_QUERY_PIPELINE;
                    if (instruction->isa<xir::ResourceQueryInst>()) {
                        is_ray_query_instruction |=
                            spirv_resource_query_requires_accel_traversal_descriptor(
                                static_cast<const xir::ResourceQueryInst *>(instruction)->op());
                    }
                    if (is_ray_query_instruction) {
                        result.plan.uses_semantic_ray_query = true;
                        ray_function = function;
                        ray_instruction = instruction;
                    }
                }
                if (!result.plan.uses_subgroup_extended_types &&
                    instruction->isa<xir::ThreadGroupInst>()) {
                    auto *thread_group =
                        static_cast<const xir::ThreadGroupInst *>(instruction);
                    if (thread_group_op_uses_value_type(thread_group->op()) &&
                        thread_group->operand_count() != 0u &&
                        thread_group->operand(0u) != nullptr &&
                        spirv_subgroup_type_requires_extended_types(
                            thread_group->operand(0u)->type())) {
                        result.plan.uses_subgroup_extended_types = true;
                        subgroup_function = function;
                        subgroup_instruction = instruction;
                    }
                }
                if (!result.plan.uses_shader_device_clock &&
                    instruction->derived_instruction_tag() ==
                        xir::DerivedInstructionTag::CLOCK) {
                    result.plan.uses_shader_device_clock = true;
                    clock_function = function;
                    clock_instruction = instruction;
                }
                if (!result.plan.uses_buffer_device_address &&
                    instruction->isa<xir::ResourceQueryInst>()) {
                    auto op = static_cast<const xir::ResourceQueryInst *>(
                                  instruction)
                                  ->op();
                    if (op == xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS ||
                        op == xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS) {
                        result.plan.uses_buffer_device_address = true;
                        device_address_function = function;
                        device_address_instruction = instruction;
                    }
                }
            });
    }

    auto uses_sampled_heap =
        result.plan.bindless_resources.texture_2d ||
        result.plan.bindless_resources.texture_3d;
    if (uses_sampled_heap ||
        result.plan.bindless_resources.buffer_heap) {
        result.plan.required_features |=
            target_feature::descriptor_indexing |
            target_feature::runtime_descriptor_array |
            target_feature::descriptor_binding_partially_bound;
    }
    if (uses_sampled_heap) {
        result.plan.required_features |=
            target_feature::descriptor_binding_sampled_image_update_after_bind;
    }
    if (result.plan.bindless_resources.buffer_heap) {
        result.plan.required_features |=
            target_feature::descriptor_binding_storage_buffer_update_after_bind;
    }
    if (result.plan.uses_semantic_ray_query) {
        result.plan.required_features |= target_feature::ray_query;
    }
    if (result.plan.uses_subgroup_extended_types) {
        result.plan.required_features |=
            target_feature::subgroup_extended_types;
    }
    if (result.plan.uses_shader_device_clock) {
        result.plan.required_features |=
            target_feature::shader_device_clock;
    }
    if (result.plan.uses_buffer_device_address) {
        result.plan.required_features |=
            target_feature::buffer_device_address |
            target_feature::shader_int64;
    }

    result.missing_features =
        result.plan.required_features & ~features.enabled_mask();
    for (auto feature : list_spirv_target_features(
             result.missing_features)) {
        auto semantic_ray = feature.bit == target_feature::ray_query;
        auto subgroup_extended =
            feature.bit == target_feature::subgroup_extended_types;
        auto shader_clock =
            feature.bit == target_feature::shader_device_clock;
        auto device_address =
            feature.bit == target_feature::buffer_device_address ||
            (feature.bit == target_feature::shader_int64 &&
             result.plan.uses_buffer_device_address);
        result.diagnostics.emplace_back(
            SpirvRuntimeTargetDiagnostic{
                .function = semantic_ray      ? ray_function :
                            subgroup_extended ? subgroup_function :
                            shader_clock      ? clock_function :
                            device_address    ? device_address_function :
                                                nullptr,
                .instruction = semantic_ray      ? ray_instruction :
                               subgroup_extended ? subgroup_instruction :
                               shader_clock      ? clock_instruction :
                               device_address    ? device_address_instruction :
                                                   nullptr,
                .feature = feature.bit,
                .message = luisa::format(
                    "Native XIR-to-SPIR-V runtime target plan requires feature '{}'.",
                    feature.name)});
    }
    return result;
}

}// namespace lc::spirv
