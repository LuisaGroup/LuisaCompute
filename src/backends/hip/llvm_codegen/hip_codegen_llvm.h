//
// Created by mike on 3/18/26.
//

#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/curve_basis.h>

namespace luisa::compute::xir {
class Module;
}// namespace luisa::compute::xir

namespace luisa::compute::hip {

struct HIPCodegenLLVMConfig {

    enum struct OptLevel : uint8_t {
        LEVEL_NONE = 0,
        LEVEL_LESS = 1,
        LEVEL_DEFAULT = 2,
        LEVEL_AGGRESSIVE = 3,
    };

    luisa::string source_file{};
    luisa::string native_include{};
    luisa::span<const Function::Binding> bindings{};
    std::array<uint32_t, 3> block_size{};
    luisa::string amdgpu_arch{};
    uint32_t wave_size{32};// 32 for RDNA wave32, 64 for wave64
    uint32_t max_register_count{0};
    OptLevel opt_level{OptLevel::LEVEL_AGGRESSIVE};
    bool enable_fast_math{true};
    bool enable_debug_info{false};
    bool requires_ray_tracing{false};
    bool requires_ray_query{false};
    bool requires_motion_blur{false};
    bool requires_static_trace{false};
    bool requires_motion_ray_query{false};
    bool requires_printing{false};
    CurveBasisSet curve_bases{CurveBasisSet::make_all()};
};

struct HIPCodegenLLVMResult {
    luisa::string code;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
    bool requires_global_rt_stack{false};
};

[[nodiscard]] HIPCodegenLLVMResult hip_codegen_llvm(
    const xir::Module &xir_module,
    const HIPCodegenLLVMConfig &config) noexcept;

// Fingerprint the exact per-architecture device wrapper linked into ray
// tracing kernels. Shader-cache identities must include this value because
// changes to the wrapper do not alter the user kernel's AST hash.
[[nodiscard]] uint64_t hip_codegen_llvm_embedded_rt_wrapper_hash(
    luisa::string_view amdgpu_arch) noexcept;

}// namespace luisa::compute::hip
