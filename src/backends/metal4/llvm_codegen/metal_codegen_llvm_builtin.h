#pragma once

#include "metal_codegen_llvm.h"

namespace luisa::compute::metal {

enum class MetalBuiltinLLVMProgram : uint8_t {
    UPDATE_ACCEL_INSTANCES,
    UPDATE_BINDLESS_ARRAY,
    PREPARE_INDIRECT_DISPATCHES,
    SWAPCHAIN_VERTEX,
    SWAPCHAIN_FRAGMENT,
};

[[nodiscard]] MetalCodegenLLVMResult
luisa_compute_metal_codegen_builtin_llvm(
    MetalBuiltinLLVMProgram program,
    const MetalCodegenLLVMConfig &config) noexcept;

}// namespace luisa::compute::metal
