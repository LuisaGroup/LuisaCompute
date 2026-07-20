#include "texture_sampling.h"
#include "structural_closure.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/value.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

bool spirv_sampler_selector_type_supported(
    const Type *type) noexcept {
    return type != nullptr && type->is_uint32();
}

SpirvSamplerSelectorDecodeResult
decode_spirv_sampler_selector_constant(
    const xir::Value *value) noexcept {
    if (value == nullptr || !value->isa<xir::Constant>()) { return {}; }
    auto *constant = static_cast<const xir::Constant *>(value);
    if (constant->type() == nullptr) {
        return {
            .diagnostic =
                "Sampler selector constant has no type."};
    }
    if (!constant->type()->is_uint32()) {
        return {
            .diagnostic = luisa::format(
                "Sampler selector constant must be uint32, got {}.",
                constant->type()->description())};
    }
    return {.value = constant->as<uint32_t>()};
}

SpirvSamplerTargetValidationResult
validate_spirv_sampler_target_contract(
    luisa::span<const xir::Function *const> functions,
    bool sampler_anisotropy_enabled) noexcept {
    SpirvSamplerTargetValidationResult result;
    for (auto *function : functions) {
        if (function == nullptr || !function->is_definition()) {
            continue;
        }
        traverse_spirv_codegen_structural_instructions(
            function->definition(),
            [&](const xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::ResourceQueryInst>()) {
                    return;
                }
                auto *query = static_cast<
                    const xir::ResourceQueryInst *>(instruction);
                auto info = spirv_texture_sample_op_info(query->op());
                if (!info.valid || !info.sampler_operands ||
                    query->operand_count() < 2u) {
                    return;
                }
                // Type and enum-range validity belong to the dialect
                // boundary. This target-only pass decides whether the exact
                // filter can reach the anisotropic heap entries.
                auto filter = decode_spirv_sampler_selector_constant(
                    query->operand(query->operand_count() - 2u));
                if (!filter) {
                    result.diagnostics.emplace_back(
                        SpirvSamplerTargetDiagnostic{
                            .function = function,
                            .instruction = query,
                            .message = luisa::format(
                                "Native XIR-to-SPIR-V texture sampler filter selector is invalid: {}",
                                filter.diagnostic)});
                    return;
                }
                switch (plan_spirv_sampler_filter(
                    filter.value.has_value(),
                    filter.value.value_or(0u),
                    sampler_anisotropy_enabled)) {
                    case SpirvSamplerFilterPlan::SUPPORTED: return;
                    case SpirvSamplerFilterPlan::INVALID_SELECTOR:
                        result.diagnostics.emplace_back(
                            SpirvSamplerTargetDiagnostic{
                                .function = function,
                                .instruction = query,
                                .message = luisa::format(
                                    "Native XIR-to-SPIR-V texture sampler filter selector {} is outside [0, 4).",
                                    *filter.value)});
                        return;
                    case SpirvSamplerFilterPlan::REQUIRES_ANISOTROPY:
                        result.diagnostics.emplace_back(
                            SpirvSamplerTargetDiagnostic{
                                .function = function,
                                .instruction = query,
                                .message =
                                    "Native XIR-to-SPIR-V texture sampler filter may select ANISOTROPIC, but samplerAnisotropy is not enabled on the target Vulkan device."});
                        return;
                }
            });
    }
    return result;
}

}// namespace lc::spirv
