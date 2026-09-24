#pragma once

#include <cstddef>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace lc::vk {

struct GlslCompileResult {
    luisa::vector<std::byte> spirv;// raw words
    luisa::string error;           // empty == success
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !spirv.empty(); }
};

// Compiles a GLSL compute shader to SPIR-V with the bundled glslang.
//
// glslang's process-global state is initialised once (R13) and shader
// compilation is serialised, because `TShader::parse` shares global tables.
// The entry point is `entry_point` (default "main" for GLSL when empty).
// `target_env` follows glslang's `EShMsgVulkanRules`/`EShMsgSpvRules` flags, so
// the emitted module uses explicit `layout(set, binding)` decorations.
[[nodiscard]] GlslCompileResult compile_glsl_to_spirv(
    luisa::string_view source,
    luisa::string_view entry_point = {},
    bool optimize = true,
    bool debug = false) noexcept;

}// namespace lc::vk
