//
// Created by mike on 3/18/26.
//

#include <luisa/core/clock.h>
#include "hip_codegen_llvm_impl.h"
#include "hip_codegen_llvm.h"

namespace luisa::compute::hip {

HIPCodegenLLVMResult hip_codegen_llvm(const xir::Module &xir_module,
                                      const HIPCodegenLLVMConfig &config) noexcept {
    Clock clk;
    HIPCodegenLLVMImpl impl{config};
    auto code = impl.generate(xir_module);
    LUISA_INFO_WITH_LOCATION("Generated AMDGPU code with HIP LLVM CodeGen in {} ms.", clk.toc());
    static auto dump_code = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_AMDGPU");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_code) {
        LUISA_INFO("Generated AMDGPU code:\n{}", code);
    }
    auto requires_global_rt_stack = impl.requires_global_rt_stack();
    return HIPCodegenLLVMResult{
        .code = std::move(code),
        .format_types = std::move(impl).take_print_formats(),
        .requires_global_rt_stack = requires_global_rt_stack};
}

}// namespace luisa::compute::hip
