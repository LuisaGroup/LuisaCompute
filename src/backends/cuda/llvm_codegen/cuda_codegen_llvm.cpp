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
    auto ptx = impl.generate(xir_module);
    LUISA_INFO_WITH_LOCATION("Generated PTX with CUDA LLVM CodeGen in {} ms.", clk.toc());
    static auto dump_ptx = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_PTX");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_ptx) {
        LUISA_INFO("Generated PTX:\n{}", ptx);
    }
    return ptx;
}

}// namespace luisa::compute::cuda
