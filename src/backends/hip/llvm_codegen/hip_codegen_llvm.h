//
// Created by mike on 3/18/26.
//

#pragma once

#include <cstddef>

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
    // The externally visible AMDGPU kernel symbol. Anonymous JIT kernels use
    // a deterministic structural name so profilers can distinguish modules;
    // explicitly named AOT packages retain the stable `kernel_main` ABI.
    luisa::string entry_point{"kernel_main"};
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
    // Internal retry control. The first translation may prove that a
    // synchronous query's callback environment is too large to materialize
    // profitably; the second translation then keeps the resumable gfx12 ABI.
    bool force_resumable_ray_query_pipeline{false};
    CurveBasisSet curve_bases{CurveBasisSet::make_all()};
};

// The synchronous pipeline copies its projected callback product once per
// query and reloads it at every accepted candidate. Restrict that hot object
// to four 16-byte ABI quanta; larger environments use the resumable hardware
// query whose handler values remain in their ordinary SSA/callable context.
inline constexpr size_t hip_synchronous_ray_query_environment_budget = 64u;

[[nodiscard]] constexpr bool
hip_synchronous_ray_query_environment_is_profitable(
    size_t projected_bytes) noexcept {
    return projected_bytes <=
           hip_synchronous_ray_query_environment_budget;
}

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
