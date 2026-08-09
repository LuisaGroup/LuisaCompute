#pragma once

#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/inline.h>

namespace lc::spirv {

enum class SpirvPointerLegalizationStatus : uint8_t {
    SUCCESS,
    UNSUPPORTED_STRUCTURED_CONTROL_FLOW,
    DESTRUCTURE_FAILED,
    INLINE_RETRY_FAILED,
};

enum class SpirvCallableReferenceActualStatus : uint8_t {
    SUCCESS,
    NULL_ACTUAL,
    SHARED_ALLOCATION,
    DERIVED_POINTER,
    UNSUPPORTED_VALUE,
};

struct SpirvCallableReferenceActualValidation {
    SpirvCallableReferenceActualStatus status{
        SpirvCallableReferenceActualStatus::SUCCESS};
    luisa::string_view diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SpirvCallableReferenceActualStatus::SUCCESS;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

struct SpirvPointerLegalizationInfo {
    SpirvPointerLegalizationStatus status{
        SpirvPointerLegalizationStatus::SUCCESS};
    size_t planned_pointer_call_count{0u};
    size_t blocking_function_count{0u};
    size_t destructured_blocking_function_count{0u};
    size_t destructured_switch_count{0u};
    size_t remaining_pointer_call_count{0u};
    size_t argument_usage_analysis_count{0u};
    size_t argument_usage_structural_closure_count{0u};
    size_t argument_usage_instruction_scan_count{0u};
    size_t argument_usage_call_dependency_count{0u};
    size_t argument_usage_worklist_pop_count{0u};
    size_t argument_usage_dependency_visit_count{0u};
    luisa::compute::xir::InlineInfo inline_info;
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SpirvPointerLegalizationStatus::SUCCESS;
    }
};

[[nodiscard]] SpirvCallableReferenceActualValidation
validate_spirv_callable_reference_actual(
    const luisa::compute::xir::Value *actual) noexcept;

[[nodiscard]] SpirvPointerLegalizationInfo
legalize_spirv_pointer_arguments(
    luisa::compute::xir::Module *module) noexcept;

}// namespace lc::spirv
