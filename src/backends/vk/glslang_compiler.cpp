#include "glslang_compiler.h"

#include <cstring>
#include <mutex>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <luisa/core/logging.h>

namespace lc::vk {

namespace {

// glslang keeps process-global tables, so process initialisation happens once
// and is never torn down while shaders may still be compiled; parsing is
// serialised because the front end shares those tables (R13).
void ensure_glslang_initialized() noexcept {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        if (!glslang::InitializeProcess()) {
            LUISA_WARNING("glslang::InitializeProcess() failed; GLSL native "
                          "shaders will not compile.");
        }
    });
}

std::mutex &glslang_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

uint32_t glsl_version_number(luisa::string_view source) noexcept {
    // glslang wants the client to state the language version; `#version 450`
    // is the Vulkan GLSL default, so it is only overridden when the source
    // asks for more.
    auto version = 450u;
    if (auto at = source.find("#version"); at != luisa::string_view::npos) {
        auto pos = at + 8u;
        while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t')) {
            pos++;
        }
        auto value = 0u;
        while (pos < source.size() && source[pos] >= '0' && source[pos] <= '9') {
            value = value * 10u + static_cast<uint32_t>(source[pos] - '0');
            pos++;
        }
        if (value != 0u) { version = value; }
    }
    return version;
}

}// namespace

GlslCompileResult compile_glsl_to_spirv(
    luisa::string_view source, luisa::string_view entry_point,
    bool optimize, bool debug) noexcept {
    GlslCompileResult result;
    if (source.empty()) {
        result.error = "GLSL source is empty.";
        return result;
    }
    ensure_glslang_initialized();
    std::lock_guard lock{glslang_mutex()};
    // GLSL has exactly one entry point per stage, and it is called `main`;
    // glslang's `setEntryPoint` selects an entry point of that name instead of
    // renaming it.
    if (!entry_point.empty() && entry_point != "main") {
        result.error = luisa::format(
            "GLSL entry points must be named 'main' (requested '{}').",
            entry_point);
        return result;
    }
    std::array<const char *, 1u> strings{source.data()};
    glslang::TShader shader{EShLangCompute};
    shader.setStrings(strings.data(), static_cast<int>(strings.size()));
    auto resources = *GetDefaultResources();
    auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (debug) {
        messages = static_cast<EShMessages>(messages | EShMsgDebugInfo);
    }
    if (!shader.parse(&resources, glsl_version_number(source), false, messages)) {
        result.error = luisa::string{"GLSL parse failed: "};
        result.error.append(shader.getInfoLog() == nullptr ? "unknown error" :
                                                            shader.getInfoLog());
        if (auto *debug_log = shader.getInfoDebugLog(); debug_log != nullptr) {
            result.error.append(" | ");
            result.error.append(debug_log);
        }
        return result;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        result.error = luisa::string{"GLSL link failed: "};
        result.error.append(program.getInfoLog() == nullptr ? "unknown error" :
                                                             program.getInfoLog());
        return result;
    }
    auto *intermediate = program.getIntermediate(EShLangCompute);
    if (intermediate == nullptr) {
        result.error = "glslang produced no compute intermediate.";
        return result;
    }
    glslang::SpvOptions options;
    options.generateDebugInfo = debug;
    options.stripDebugInfo = !debug;
    options.disableOptimizer = !optimize;
    options.optimizeSize = false;
    std::vector<uint32_t> words;
    glslang::GlslangToSpv(*intermediate, words, &options);
    if (words.empty()) {
        result.error = "glslang produced an empty SPIR-V module.";
        return result;
    }
    result.spirv.resize(words.size() * sizeof(uint32_t));
    std::memcpy(result.spirv.data(), words.data(), result.spirv.size());
    return result;
}

}// namespace lc::vk
