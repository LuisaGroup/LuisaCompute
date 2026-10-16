// Unit tests for the shared native-shader reflection helpers
// (src/backends/common/native_shader/native_shader_reflection.h).
//
// These tests need no device: the SPIR-V parser is exercised on (a) synthetic
// modules for the error/edge paths and (b) real modules produced by the bundled
// glslang, which is what the Vulkan native-shader route consumes at runtime.
//
// Covered risks: R2 (SPIR-V decorations are the authoritative binding table),
// R3 (opaque-type reflection), R9 (canonical order), R15 (workgroup size).
#include "ut/ut.hpp"
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/vector.h>
#include "native_shader/native_shader_reflection.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::native_shader_reflection;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Compiles a GLSL compute shader to SPIR-V words with the bundled glslang.
[[nodiscard]] luisa::vector<std::byte> compile_glsl(
    const char *source, luisa::string *error) noexcept {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] { glslang::InitializeProcess(); });
    auto shader = glslang::TShader{EShLangCompute};
    shader.setStrings(&source, 1);
    auto resources = *GetDefaultResources();
    auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(&resources, 450, false, messages)) {
        *error = luisa::string{shader.getInfoLog()};
        return {};
    }
    auto program = glslang::TProgram{};
    program.addShader(&shader);
    if (!program.link(messages)) {
        *error = luisa::string{program.getInfoLog()};
        return {};
    }
    glslang::SpvOptions options;
    options.generateDebugInfo = false;
    options.stripDebugInfo = true;
    options.disableOptimizer = true;
    auto words = std::vector<uint32_t>{};
    glslang::GlslangToSpv(*program.getIntermediate(EShLangCompute), words, &options);
    luisa::vector<std::byte> result(words.size() * sizeof(uint32_t));
    std::memcpy(result.data(), words.data(), result.size());
    return result;
}

[[nodiscard]] luisa::vector<std::byte> make_synthetic_module(
    uint32_t bound, luisa::span<const uint32_t> instructions) noexcept {
    luisa::vector<uint32_t> words;
    words.emplace_back(spirv::magic);
    words.emplace_back(0x00010000u);// version 1.0
    words.emplace_back(0u);         // generator
    words.emplace_back(bound);      // id bound
    words.emplace_back(0u);         // schema
    words.insert(words.end(), instructions.begin(), instructions.end());
    luisa::vector<std::byte> result(words.size() * sizeof(uint32_t));
    std::memcpy(result.data(), words.data(), result.size());
    return result;
}

[[nodiscard]] const NativeShaderResourceBinding *find_binding(
    const SpirvReflection &reflection, uint32_t set, uint32_t binding) noexcept {
    for (auto &&b : reflection.bindings) {
        if (b.space_index == set && b.register_index == binding) { return &b; }
    }
    return nullptr;
}

}// namespace

int main() {

    "spirv_rejects_malformed_modules"_test = [] {
        auto too_small = luisa::vector<std::byte>{};
        expect(!parse_spirv(too_small).ok());
        luisa::vector<std::byte> misaligned(7u);
        expect(!parse_spirv(misaligned).ok());
        // Valid header, wrong magic.
        luisa::vector<uint32_t> words{0xdeadbeefu, 0x00010000u, 0u, 1u, 0u};
        luisa::vector<std::byte> bytes(words.size() * sizeof(uint32_t));
        std::memcpy(bytes.data(), words.data(), bytes.size());
        auto result = parse_spirv(bytes);
        expect(!result.ok());
        expect(!result.error.empty());
        // Byte-swapped magic is reported with its own message.
        words[0] = spirv::magic_reversed;
        std::memcpy(bytes.data(), words.data(), bytes.size());
        expect(!parse_spirv(bytes).ok());
        // Truncated instruction.
        words[0] = spirv::magic;
        words.emplace_back((4u << 16u) | spirv::OpExecutionMode);
        luisa::vector<std::byte> truncated(words.size() * sizeof(uint32_t));
        std::memcpy(truncated.data(), words.data(), truncated.size());
        expect(!parse_spirv(truncated).ok());
    };

    "spirv_execution_mode_local_size"_test = [] {
        // OpExecutionMode %1 LocalSize 8 4 2
        const uint32_t instructions[] = {
            (6u << 16u) | spirv::OpExecutionMode,
            1u, spirv::ExecutionModeLocalSize, 8u, 4u, 2u};
        auto module = make_synthetic_module(2u, instructions);
        auto result = parse_spirv(module);
        expect(result.ok());
        expect(result.block_size.x == 8u);
        expect(result.block_size.y == 4u);
        expect(result.block_size.z == 2u);
        expect(!result.has_push_constant);
        expect(result.bindings.empty());
    };

    "glsl_reflection_covers_opaque_types"_test = [] {
        constexpr auto source = R"(
            #version 450
            layout(local_size_x = 64, local_size_y = 2, local_size_z = 1) in;
            layout(set = 0, binding = 0) readonly buffer A { float a[]; } abuf;
            layout(set = 0, binding = 1) buffer B { float b[]; } bbuf;
            layout(set = 0, binding = 2) uniform U { float scale; } ubuf;
            layout(set = 0, binding = 3) uniform sampler2D tex;
            layout(set = 0, binding = 5, rgba32f) uniform image2D img;
            layout(push_constant) uniform Push { float k; float pad; } pc;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                bbuf.b[i] = abuf.a[i] * ubuf.scale * pc.k;
            }
        )";
        auto error = luisa::string{};
        auto binary = compile_glsl(source, &error);
        expect(binary.empty() == false) << "glslang failed: " << error.c_str();
        auto result = parse_spirv(binary);
        expect(result.ok()) << result.error.c_str();
        // R15: the workgroup size comes from OpExecutionMode LocalSize.
        expect(result.block_size.x == 64u);
        expect(result.block_size.y == 2u);
        expect(result.block_size.z == 1u);
        // R10: the push-constant block is detected and sized.
        expect(result.has_push_constant);
        expect(result.push_constant_size == 8u);
        expect(result.bindings.size() == 5u);
        // R3: opaque types are reflected.
        if (auto binding = find_binding(result, 0u, 0u)) {
            expect(binding->kind == NativeShaderResourceKind::StructuredBuffer);
            expect(binding->usage == Usage::READ);
        } else {
            expect(false) << "missing binding 0";
        }
        if (auto binding = find_binding(result, 0u, 1u)) {
            expect(binding->kind == NativeShaderResourceKind::RWStructuredBuffer);
            expect(binding->usage == Usage::READ_WRITE);
        } else {
            expect(false) << "missing binding 1";
        }
        if (auto binding = find_binding(result, 0u, 2u)) {
            expect(binding->kind == NativeShaderResourceKind::ConstantBuffer);
            expect(binding->usage == Usage::READ);
            expect(binding->size_bytes == 4u);
        } else {
            expect(false) << "missing binding 2";
        }
        if (auto binding = find_binding(result, 0u, 3u)) {
            expect(binding->kind == NativeShaderResourceKind::Texture2D);
        } else {
            expect(false) << "missing binding 3";
        }
        if (auto binding = find_binding(result, 0u, 5u)) {
            expect(binding->kind == NativeShaderResourceKind::RWTexture2D);
            expect(binding->usage == Usage::READ_WRITE);
        } else {
            expect(false) << "missing binding 5";
        }
        // R9: the table is sorted into the canonical (space, register) order.
        for (auto i = 1u; i < result.bindings.size(); i++) {
            auto &&previous = result.bindings[i - 1u];
            auto &&current = result.bindings[i];
            expect(previous.space_index < current.space_index ||
                   (previous.space_index == current.space_index &&
                    previous.register_index < current.register_index));
        }
    };

    "glsl_reflection_multiple_sets"_test = [] {
        constexpr auto source = R"(
            #version 450
            layout(local_size_x = 4) in;
            layout(set = 1, binding = 2) buffer S { uint s[]; } s1;
            layout(set = 0, binding = 7) readonly buffer T { uint t[]; } t0;
            void main() {}
        )";
        auto error = luisa::string{};
        auto binary = compile_glsl(source, &error);
        expect(binary.empty() == false) << "glslang failed: " << error.c_str();
        auto result = parse_spirv(binary);
        expect(result.ok()) << result.error.c_str();
        expect(result.bindings.size() == 2u);
        // Canonical order sorts set 0 before set 1.
        expect(result.bindings[0].space_index == 0u);
        expect(result.bindings[0].register_index == 7u);
        expect(result.bindings[1].space_index == 1u);
        expect(result.bindings[1].register_index == 2u);
    };

    "kind_usage_and_canonical_helpers"_test = [] {
        expect(native_shader_default_usage(NativeShaderResourceKind::StructuredBuffer) ==
               Usage::READ);
        expect(native_shader_default_usage(NativeShaderResourceKind::RWByteAddressBuffer) ==
               Usage::READ_WRITE);
        expect(native_shader_default_usage(NativeShaderResourceKind::ConstantBuffer) ==
               Usage::READ);
        luisa::vector<NativeShaderResourceBinding> bindings;
        auto add = [&](uint32_t space, uint32_t reg, NativeShaderResourceKind kind) {
            NativeShaderResourceBinding b;
            b.space_index = space;
            b.register_index = reg;
            b.kind = kind;
            b.usage = native_shader_default_usage(kind);
            bindings.emplace_back(b);
        };
        add(1u, 0u, NativeShaderResourceKind::StructuredBuffer);
        add(0u, 3u, NativeShaderResourceKind::RWStructuredBuffer);
        add(0u, 1u, NativeShaderResourceKind::Texture2D);
        luisa::compute::detail::native_shader_canonicalize_bindings(bindings);
        expect(bindings[0].space_index == 0u && bindings[0].register_index == 1u);
        expect(bindings[1].space_index == 0u && bindings[1].register_index == 3u);
        expect(bindings[2].space_index == 1u && bindings[2].register_index == 0u);
    };

    "spirv_feature_requirements_are_scanned"_test = [] {
        // R12: `OpCapability` operands are mapped onto the runtime's persistent
        // artifact requirement mask so that a module needing an optional feature
        // can fail closed before pipeline creation.
        auto make_module = [](uint32_t capability) noexcept {
            const uint32_t instructions[] = {
                (2u << 16u) | spirv::OpCapability, capability};
            return make_synthetic_module(1u, instructions);
        };
        auto int64_module = make_module(11u);// OpCapability Int64
        auto result = parse_spirv(int64_module);
        expect(result.ok()) << result.error.c_str();
        expect(result.capabilities.size() == 1u);
        expect(result.required_features == lc::spirv::target_feature::shader_int64);
        // A device without the feature fails closed with a named requirement.
        auto missing = lc::spirv::check_spirv_target_feature_requirements(
            result.required_features, 0u);
        expect(!static_cast<bool>(missing));
        expect(missing.missing_required_bits ==
               lc::spirv::target_feature::shader_int64);
        auto names = native_shader_reflection::detail::describe_required_features(
            missing.missing_required_bits);
        expect(names.find("shaderInt64") != luisa::string::npos) << names.c_str();
        // ... and passes when the device enabled it.
        expect(static_cast<bool>(lc::spirv::check_spirv_target_feature_requirements(
            result.required_features, lc::spirv::target_feature::shader_int64)));
        // Capabilities that are core for compute shaders require nothing.
        auto shader_module = make_module(1u);// OpCapability Shader
        expect(parse_spirv(shader_module).required_features == 0u);
    };

    "glsl_compilation_is_thread_safe"_test = [] {
        // R13: glslang's global tables are shared, so compilation is serialised
        // behind a process-wide guard.
        constexpr auto source = R"(
            #version 450
            layout(local_size_x = 8) in;
            layout(set = 0, binding = 0) buffer B { uint b[]; } buf;
            void main() { buf.b[gl_GlobalInvocationID.x] = 1u; }
        )";
        auto results = std::array<luisa::vector<std::byte>, 2u>{};
        auto errors = std::array<luisa::string, 2u>{};
        {
            auto threads = std::array<std::thread, 2u>{};
            for (auto i = 0u; i < threads.size(); i++) {
                threads[i] = std::thread{[&, i] {
                    results[i] = compile_glsl(source, &errors[i]);
                }};
            }
            for (auto &thread : threads) { thread.join(); }
        }
        for (auto i = 0u; i < results.size(); i++) {
            expect(!results[i].empty()) << errors[i].c_str();
            auto reflection = parse_spirv(results[i]);
            expect(reflection.ok()) << reflection.error.c_str();
            expect(reflection.bindings.size() == 1u);
            expect(reflection.block_size.x == 8u);
        }
    };

    return 0;
}
