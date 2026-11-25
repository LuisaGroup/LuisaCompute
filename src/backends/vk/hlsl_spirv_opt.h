#pragma once

#include "../common/hlsl/shader_compiler.h"
#include <luisa/vstl/common.h>
#include <cstddef>
#include <cstdint>
#include <vector>

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

// Removes the 32-bit denormal `OpCapability`/`OpExecutionMode` declarations the
// device does not advertise. DXC emits `DenormFlushToZero` for the fast-math
// profile and `DenormPreserve` for the strict one regardless of the device's
// float-controls properties, and VUID-VkShaderModuleCreateInfo-pCode-08740
// rejects a module that declares a behaviour the device does not support. The
// driver default then applies, exactly as if DXC had emitted nothing.
// Returns the number of removed instructions.
// Cache identity of one HLSL-generated module: MD5 over the generated source
// *plus* the denormal-control policy. A module whose unsupported denormal
// declarations are stripped must not be served from an older cache entry that
// still declares them (VUID-VkShaderModuleCreateInfo-pCode-08740).
[[nodiscard]] vstd::MD5 hlsl_module_cache_md5(
    vstd::string_view source,
    bool denorm_preserve, bool denorm_flush) noexcept;

[[nodiscard]] size_t strip_unsupported_denorm_modes(
    std::vector<uint32_t> &words,
    bool keep_preserve, bool keep_flush) noexcept;

}// namespace lc::vk
