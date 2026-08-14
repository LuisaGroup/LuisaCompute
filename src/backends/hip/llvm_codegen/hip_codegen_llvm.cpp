//
// Created by mike on 3/18/26.
//

#include <utility>

#include <luisa/core/clock.h>
#include "hip_codegen_llvm_impl.h"
#include "hip_codegen_llvm.h"

namespace luisa::compute::hip {

HIPCodegenLLVMResult hip_codegen_llvm(const xir::Module &xir_module,
                                      const HIPCodegenLLVMConfig &config) noexcept {
    Clock clk;
    struct GenerationAttempt {
        luisa::string code;
        luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
        bool requires_global_rt_stack;
        bool retry_with_resumable_ray_query_pipeline;
    };
    auto generate_once = [&](HIPCodegenLLVMConfig attempt_config) noexcept {
        HIPCodegenLLVMImpl impl{std::move(attempt_config)};
        auto code = impl.generate(xir_module);
        auto requires_global_rt_stack = impl.requires_global_rt_stack();
        auto retry_with_resumable_ray_query_pipeline =
            impl.retry_with_resumable_ray_query_pipeline();
        auto format_types = std::move(impl).take_print_formats();
        return GenerationAttempt{
            .code = std::move(code),
            .format_types = std::move(format_types),
            .requires_global_rt_stack = requires_global_rt_stack,
            .retry_with_resumable_ray_query_pipeline =
                retry_with_resumable_ray_query_pipeline};
    };
    auto attempt = generate_once(config);
    if (attempt.retry_with_resumable_ray_query_pipeline) {
        auto resumable_config = config;
        resumable_config.force_resumable_ray_query_pipeline = true;
        attempt = generate_once(std::move(resumable_config));
        LUISA_ASSERT(
            !attempt.retry_with_resumable_ray_query_pipeline,
            "Resumable HIP RayQuery codegen requested a recursive retry.");
    }
    LUISA_INFO_WITH_LOCATION("Generated AMDGPU code with HIP LLVM CodeGen in {} ms.", clk.toc());
    static auto dump_code = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_AMDGPU");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_code) {
        LUISA_INFO("Generated AMDGPU code:\n{}", attempt.code);
    }
    return HIPCodegenLLVMResult{
        .code = std::move(attempt.code),
        .format_types = std::move(attempt.format_types),
        .requires_global_rt_stack = attempt.requires_global_rt_stack};
}

}// namespace luisa::compute::hip
