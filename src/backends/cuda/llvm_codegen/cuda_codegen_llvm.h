//
// Created by mike on 9/17/25.
//

#pragma once

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::xir {
class Module;
}// namespace luisa::compute::xir

namespace luisa::compute::cuda {

struct CUDACodegenLLVMResult {
    luisa::string code;
    uint32_t ray_query_payload_count{2u};
};

[[nodiscard]] CUDACodegenLLVMResult luisa_compute_cuda_codegen_llvm(
    const xir::Module &xir_module,
    const CUDACodegenLLVMConfig &config) noexcept;

}// namespace luisa::compute::cuda
