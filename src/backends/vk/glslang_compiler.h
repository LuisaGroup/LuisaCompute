#pragma once

#include <cstddef>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace lc::vk {

struct GlslCompileResult {
    luisa::vector<std::byte> spirv;// raw words
    luisa::string error;           // empty == success
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !spirv.empty(); }
};

// The pipeline stage a GLSL source is compiled for. GLSL declares its stage in
// the source (vertex/fragment/compute), but glslang still has to be told which
// `EShLang*` front end to use, so it is an explicit parameter; `Compute` is the
// default because that is the only stage the compute native-shader route
// compiles.
enum class GlslShaderStage : uint8_t {
    Compute,
    Vertex,
    Fragment,
};

// Compiles a GLSL shader stage to SPIR-V with the bundled glslang.
//
// glslang's process-global state is initialised once (R13) and shader
// compilation is serialised, because `TShader::parse` shares global tables.
// The entry point is `entry_point` (default "main" for GLSL when empty).
// `target_env` follows glslang's `EShMsgVulkanRules`/`EShMsgSpvRules` flags, so
// the emitted module uses explicit `layout(set, binding)` decorations.
// `include_dirs` are searched (after the includer's own directory) when the
// source uses `#include` with the `GL_GOOGLE_include_directive` extension.
[[nodiscard]] GlslCompileResult compile_glsl_to_spirv(
    luisa::string_view source,
    luisa::string_view entry_point = {},
    bool optimize = true,
    bool debug = false,
    luisa::span<const luisa::filesystem::path> include_dirs = {},
    GlslShaderStage stage = GlslShaderStage::Compute) noexcept;

}// namespace lc::vk
