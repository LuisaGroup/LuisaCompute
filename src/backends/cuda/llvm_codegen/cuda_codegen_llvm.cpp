//
// Created by mike on 9/17/25.
//

#include <luisa/core/clock.h>
#include "cuda_codegen_llvm_impl.h"
#include "cuda_codegen_llvm.h"

namespace luisa::compute::cuda {

luisa::string luisa_compute_cuda_codegen_llvm(const xir::Module &xir_module, const CUDACodegenLLVMConfig &config) noexcept {
    Clock clk;
    CUDACodegenLLVMImpl impl{config};
    auto code = impl.generate(xir_module);
    auto optix_ir = config.output_format == CUDACodegenLLVMConfig::OutputFormat::OPTIX_IR;
    LUISA_INFO_WITH_LOCATION("Generated {} with CUDA LLVM CodeGen in {} ms.", optix_ir ? "OptiX IR" : "PTX", clk.toc());
    static auto dump_ptx = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_PTX");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_ptx && !optix_ir) {
        LUISA_INFO("Generated PTX:\n{}", code);
    }
    return code;
}

}// namespace luisa::compute::cuda
