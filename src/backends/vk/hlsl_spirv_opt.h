#pragma once

#include "../common/hlsl/shader_compiler.h"
#include <luisa/vstl/common.h>

namespace lc::vk {

// Compiles HLSL compute source to validated, optimized SPIR-V.
// For large shaders (where DXC's built-in RegisterLegalizationPasses can hit
// 4,194,303 ID overflow or spend tens of seconds in unbounded loop-unroll),
// or as an automatic fallback when DXC hits "ID overflow", emits unlegalized
// SPIR-V via DXC `-fcgl` and runs Luisa's bundled SPIRV-Tools optimizer with
// interleaved compact-ids, bounded SROA, and post-optimization validation.
[[nodiscard]] hlsl::CompileResult compile_hlsl_compute_to_spirv(
    hlsl::ShaderCompiler *compiler,
    vstd::string_view code,
    bool optimize,
    uint shader_model,
    bool enable_unsafe_math,
    bool debug,
    vstd::string_view entry_point,
    luisa::span<const luisa::filesystem::path> include_dirs,
    hlsl::SpirvShaderCapabilities spirv_capabilities,
    vstd::string_view profile_label = {}) noexcept;

}// namespace lc::vk
