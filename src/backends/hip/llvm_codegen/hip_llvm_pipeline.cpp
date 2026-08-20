//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#include "hip_llvm_pipeline.h"

#include <string_view>

namespace luisa::compute::hip {

bool preserve_hip_backend_noinline_boundary(
    std::string_view function_name,
    bool has_noinline_attribute) noexcept {
    return has_noinline_attribute &&
           (function_name.starts_with("luisa_ray_query_") ||
            function_name.starts_with("luisa_motion_ray_query_") ||
            function_name.starts_with(
                "luisa_hiprt_stack_overflow_fallback_"));
}

size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept {
    // LLVM serializes SimplifyCFG options as semicolon-delimited tokens inside
    // angle brackets. Matching the delimiters is intentional: a pass name or
    // a future longer option containing this text must remain untouched.
    constexpr std::string_view noncanonical{";no-keep-loops;"};
    constexpr std::string_view canonical{";keep-loops;"};
    auto replacement_count = size_t{0u};
    auto offset = size_t{0u};
    while ((offset = pipeline.find(noncanonical, offset)) != std::string::npos) {
        pipeline.replace(offset, noncanonical.size(), canonical);
        offset += canonical.size();
        replacement_count++;
    }
    return replacement_count;
}

}// namespace luisa::compute::hip
