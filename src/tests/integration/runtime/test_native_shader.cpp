// Device tests for the native shader injection extension
// (NativeShaderExt / NativeShaderLauncher / NativeShaderDispatchCommand).
//
// Run as: test_native_shader <dx|vk>
//
// Covered risks:
//  * R14 - a non-`main` entry point compiles.
//  * R15 - the reflected workgroup size drives the dispatch.
//  * R9  - reflection order / binding-aware launcher arguments.
//  * R6  - producer(WRITE)/consumer(READ) ordering with command reordering both
//          enabled and disabled.
//  * R16 - load/destroy does not leak.
//  * R2/R3 - the Vulkan HLSL/GLSL binding tables come from the SPIR-V module.
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/backends/ext/command_reorder_ext.h>
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// `dst[i] = src[i] * k + c` with an HLSL entry point named CSMain (R14) and a
// uniform block fed by root 32-bit constants (dx) / push constants (vk).
constexpr auto hlsl_source = R"(
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * k + c;
}
)";

// Vulkan HLSL uses the SPIR-V facing attributes so that the declared
// (set, binding) pairs are explicit (R2).
constexpr auto vk_hlsl_source = R"(
struct Uniforms { float k; float c; };
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;
[[vk::push_constant]] ConstantBuffer<Uniforms> uniforms;
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * uniforms.k + uniforms.c;
}
)";

// GLSL equivalent with a declared layout (R3) and a push-constant block (R10).
constexpr auto glsl_source = R"(
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float k; float c; } uniforms;
void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = src.a[i] * uniforms.k + uniforms.c;
}
)";

constexpr auto element_count = 1024u;
constexpr auto block_size = uint3{64u, 1u, 1u};

[[nodiscard]] const NativeShaderResourceBinding *find_binding(
    luisa::span<const NativeShaderResourceBinding> bindings,
    NativeShaderResourceKind kind) noexcept {
    for (auto &&binding : bindings) {
        if (binding.kind == kind) { return &binding; }
    }
    return nullptr;
}

// Builds a launcher for the `dst[i] = src[i] * k + c` shaders, binding by the
// reflected binding index (unambiguous even when HLSL register namespaces
// collide, which they do for `register(t0)`/`register(b0)`).
[[nodiscard]] luisa::unique_ptr<NativeShaderDispatchCommand> make_dispatch(
    const NativeShader &shader, const BufferView<float> &src,
    const BufferView<float> &dst, float k, float c, uint count) noexcept {
    auto launcher = shader.launcher();
    auto src_binding = find_binding(shader.bindings(),
                                    NativeShaderResourceKind::StructuredBuffer);
    auto dst_binding = find_binding(shader.bindings(),
                                    NativeShaderResourceKind::RWStructuredBuffer);
    expect(src_binding != nullptr);
    expect(dst_binding != nullptr);
    auto *bindings = shader.bindings().data();
    auto src_index = static_cast<uint32_t>(src_binding - bindings);
    auto dst_index = static_cast<uint32_t>(dst_binding - bindings);
    launcher.add_buffer_by_index(src_index, src, Usage::READ)
        .add_buffer_by_index(dst_index, dst, Usage::WRITE)
        .add_uniform(k)
        .add_uniform(c);
    return std::move(launcher).build(uint3{count, 1u, 1u});
}

}// namespace

static void test_native_shader(Device &device) {
    auto backend = device.backend_name();
    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        LUISA_WARNING("Backend '{}' has no NativeShaderExt; skipping.", backend);
        return;
    }
    auto is_dx = backend == "dx";
    auto is_vk = backend == "vk";

    // ---- compile + reflection -------------------------------------------
    if (is_dx) {
        "native_shader_dx_hlsl_reflection"_test = [&] {
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::HLSL;
            info.source = hlsl_source;
            info.entry_point = "CSMain";
            info.push_constant_size = 2u * sizeof(float);
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            expect(result.block_size.x == 64u && result.block_size.y == 1u &&
                   result.block_size.z == 1u);
            // `compile` reports the raw DXC reflection: the SRV, the UAV and the
            // uniform cbuffer.
            expect(result.bindings.size() == 3u) << result.bindings.size();
            // `load` replaces the uniform cbuffer with the launcher's uniform
            // block, so only the two buffers remain to be bound.
            auto metadata = ext->load(result);
            expect(metadata.valid());
            expect(metadata.bindings.size() == 2u) << metadata.bindings.size();
            if (auto *srv = find_binding(metadata.bindings,
                                         NativeShaderResourceKind::StructuredBuffer)) {
                expect(srv->register_index == 0u);
                expect(srv->space_index == 0u);
                expect(srv->usage == Usage::READ);
            } else {
                expect(false) << "missing SRV reflection";
            }
            if (auto *uav = find_binding(metadata.bindings,
                                         NativeShaderResourceKind::RWStructuredBuffer)) {
                expect(uav->register_index == 0u);
                expect(uav->usage == Usage::READ_WRITE);
            } else {
                expect(false) << "missing UAV reflection";
            }
            ext->destroy_shader(metadata.handle);
        };
        "native_shader_dx_rejects_glsl"_test = [&] {
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::GLSL;
            info.source = glsl_source;
            auto result = ext->compile(info);
            expect(!result.ok());
            expect(!result.error.empty());
            expect(result.error.find("GLSL") != luisa::string::npos)
                << result.error.c_str();
        };
    }
    if (is_vk) {
        "native_shader_vk_glsl_reflection"_test = [&] {
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::GLSL;
            info.source = glsl_source;
            info.push_constant_size = 2u * sizeof(float);
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            expect(!result.binary.empty());
            expect(result.block_size.x == 64u && result.block_size.y == 1u &&
                   result.block_size.z == 1u);
            expect(result.bindings.size() == 2u) << result.bindings.size();
            if (result.bindings.size() == 2u) {
                expect(result.bindings[0].kind ==
                       NativeShaderResourceKind::StructuredBuffer);
                expect(result.bindings[0].register_index == 0u);
                expect(result.bindings[0].usage == Usage::READ);
                expect(result.bindings[1].kind ==
                       NativeShaderResourceKind::RWStructuredBuffer);
                expect(result.bindings[1].register_index == 1u);
                expect(result.bindings[1].usage == Usage::READ_WRITE);
            }
        };
        "native_shader_vk_hlsl_register_mapping"_test = [&] {
            // R2 gate: with plain HLSL registers, the SPIR-V decorations DXC
            // emits are the authoritative (set, binding) table.
            constexpr auto source = R"(
StructuredBuffer<float> src : register(t1);
RWStructuredBuffer<float> dst : register(u2);
[numthreads(32, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x];
}
)";
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::HLSL;
            info.source = source;
            info.entry_point = "CSMain";
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            expect(result.block_size.x == 32u && result.block_size.y == 1u &&
                   result.block_size.z == 1u);
            expect(result.bindings.size() == 2u) << result.bindings.size();
            if (result.bindings.size() == 2u) {
                LUISA_INFO("vk HLSL register mapping: t1 -> set {} binding {}, "
                           "u2 -> set {} binding {}",
                           result.bindings[0].space_index,
                           result.bindings[0].register_index,
                           result.bindings[1].space_index,
                           result.bindings[1].register_index);
                expect(result.bindings[0].space_index == 0u);
                expect(result.bindings[0].register_index == 1u);
                expect(result.bindings[0].usage == Usage::READ);
                expect(result.bindings[1].space_index == 0u);
                expect(result.bindings[1].register_index == 2u);
                expect(result.bindings[1].usage == Usage::READ_WRITE);
            }
        };
        "native_shader_vk_binding_beyond_canonical_set"_test = [&] {
            // R21: the Vulkan native-shader route does not go through the DSL
            // descriptor plan, so a `set` beyond the canonical
            // `descriptor_interface_max_set_count` (5) is representable: the
            // module's set index maps onto a pipeline-layout index, and the
            // unused sets in between get empty layouts.
            constexpr auto source = R"(
                #version 450
                layout(local_size_x = 32) in;
                layout(set = 9, binding = 3) buffer B { uint b[]; } buf;
                void main() { buf.b[gl_GlobalInvocationID.x] = 42u; }
            )";
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::GLSL;
            info.source = source;
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            expect(result.bindings.size() == 1u);
            if (result.bindings.size() == 1u) {
                expect(result.bindings[0].space_index == 9u);
                expect(result.bindings[0].register_index == 3u);
            }
            auto metadata = ext->load(result);
            expect(metadata.valid());
            expect(metadata.block_size.x == 32u);
            NativeShader shader{*ext, std::move(metadata)};
            Stream stream = device.create_stream();
            auto buffer = device.create_buffer<uint>(64u);
            auto launcher = shader.launcher();
            launcher.add_buffer_by_index(0u, buffer.view(), Usage::WRITE);
            stream << std::move(launcher).build(uint3{64u, 1u, 1u})
                   << synchronize();
            luisa::vector<uint> host(64u, 0u);
            stream << buffer.copy_to(luisa::span{host}) << synchronize();
            for (auto i = 0u; i < host.size(); i++) {
                expect(host[i] == 42u) << "set-9 binding dispatch mismatch at " << i;
            }
        };
        "native_shader_vk_feature_requirements"_test = [&] {
            // R12: a module that needs an optional feature either loads (the
            // device enabled it) or fails closed at compile() naming it.
            constexpr auto source = R"(
RWStructuredBuffer<uint64_t> b : register(u0);
[numthreads(32, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    b[tid.x] = 0x123456789ull;
}
)";
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::HLSL;
            info.source = source;
            info.entry_point = "CSMain";
            auto result = ext->compile(info);
            if (!result.ok()) {
                expect(result.error.find("shaderInt64") != luisa::string::npos)
                    << result.error.c_str();
            } else {
                // The device enabled shaderInt64, so the module must load; the
                // instance is destroyed before the test (and the device) goes
                // away, as the API contract requires.
                auto metadata = ext->load(result);
                expect(metadata.valid());
                if (metadata.valid()) {
                    NativeShader temporary{*ext, std::move(metadata)};
                    expect(static_cast<bool>(temporary));
                }
            }
        };
        "native_shader_vk_hlsl_reflection"_test = [&] {
            NativeShaderCompileInfo info;
            info.language = NativeShaderLanguage::HLSL;
            info.source = vk_hlsl_source;
            info.entry_point = "CSMain";
            info.push_constant_size = 2u * sizeof(float);
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            expect(!result.binary.empty());
            expect(result.bindings.size() == 2u) << result.bindings.size();
            if (result.bindings.size() == 2u) {
                // The SPIR-V decorations are authoritative (R2).
                expect(result.bindings[0].space_index == 0u);
                expect(result.bindings[0].register_index == 0u);
                expect(result.bindings[1].space_index == 0u);
                expect(result.bindings[1].register_index == 1u);
            }
        };
    }

    // ---- dispatch --------------------------------------------------------
    {
        NativeShaderCompileInfo info;
        info.language = is_dx ? NativeShaderLanguage::HLSL : NativeShaderLanguage::GLSL;
        info.source = is_dx ? hlsl_source :
                      (is_vk ? glsl_source : luisa::string_view{});
        // The HLSL sources use a non-`main` entry point (R14); GLSL has exactly
        // one entry point per stage and it is called `main`.
        info.entry_point = is_dx ? "CSMain" : "main";
        info.push_constant_size = 2u * sizeof(float);
        if (is_dx || is_vk) {
            auto result = ext->compile(info);
            expect(result.ok()) << result.error.c_str();
            auto metadata = ext->load(result);
            expect(metadata.valid());
            if (!metadata.valid()) { return; }
            NativeShader shader{*ext, metadata};
            expect(static_cast<bool>(shader));
            Stream stream = device.create_stream();

            luisa::vector<float> host_a(element_count);
            for (auto i = 0u; i < element_count; i++) {
                host_a[i] = static_cast<float>(i);
            }
            Buffer<float> a = device.create_buffer<float>(element_count);
            Buffer<float> b = device.create_buffer<float>(element_count);
            stream << a.copy_from(luisa::span{host_a});

            "native_shader_dispatch"_test = [&] {
                auto cmd = make_dispatch(shader, a.view(), b.view(), 2.0f, 1.0f,
                                         element_count);
                expect(cmd->custom_cmd_uuid() ==
                       luisa::to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH));
                expect(cmd->max_dispatch_size().x == element_count);
                stream << std::move(cmd) << synchronize();
                luisa::vector<float> host_b(element_count);
                stream << b.copy_to(luisa::span{host_b}) << synchronize();
                for (auto i = 0u; i < element_count; i++) {
                    expect(host_b[i] == host_a[i] * 2.0f + 1.0f)
                        << "dispatch mismatch at " << i << ": " << host_b[i]
                        << " vs " << host_a[i] * 2.0f + 1.0f;
                }
            };

            // R6: producer(WRITE) -> consumer(READ) with an independent
            // dispatch interleaved, reordering enabled and disabled.
            auto reorder_ext = device.extension<CommandReorderExt>();
            "native_shader_reorder_chain"_test = [&] {
                for (auto reorder_enabled : {true, false}) {
                    if (reorder_ext != nullptr) {
                        reorder_ext->set_command_reorder_enabled(reorder_enabled);
                    }
                    Buffer<float> p = device.create_buffer<float>(element_count);
                    Buffer<float> q = device.create_buffer<float>(element_count);
                    Buffer<float> r = device.create_buffer<float>(element_count);
                    Kernel1D independent = [](BufferFloat r) noexcept {
                        r->write(dispatch_id().x,
                                 cast<float>(dispatch_id().x) + 7.0f);
                    };
                    auto independent_shader = device.compile(independent);
                    CommandList cmdlist;
                    cmdlist << make_dispatch(shader, a.view(), p.view(), 2.0f, 0.0f,
                                             element_count);
                    cmdlist << independent_shader(r.view()).dispatch(element_count);
                    cmdlist << make_dispatch(shader, p.view(), q.view(), 3.0f, 0.0f,
                                             element_count);
                    stream << cmdlist.commit() << synchronize();
                    luisa::vector<float> host_q(element_count);
                    luisa::vector<float> host_r(element_count);
                    stream << q.copy_to(luisa::span{host_q})
                           << r.copy_to(luisa::span{host_r})
                           << synchronize();
                    for (auto i = 0u; i < element_count; i++) {
                        auto expected_q = host_a[i] * 2.0f * 3.0f;
                        expect(host_q[i] == expected_q)
                            << "reorder mismatch (reorder=" << reorder_enabled
                            << ") at " << i << ": " << host_q[i] << " vs "
                            << expected_q;
                        expect(host_r[i] == static_cast<float>(i) + 7.0f)
                            << "independent dispatch mismatch at " << i;
                    }
                }
                if (reorder_ext != nullptr) {
                    reorder_ext->set_command_reorder_enabled(true);
                }
            };

            // R16: repeated compile/load/destroy must not leak (run under
            // `scripts/mem_monitor.py` to check the process's private memory).
            "native_shader_load_destroy_loop"_test = [&] {
                for (auto i = 0; i < 32; i++) {
                    auto m = ext->load(result);
                    expect(m.valid());
                    NativeShader temporary{*ext, std::move(m)};
                    expect(static_cast<bool>(temporary));
                }
                // A deliberately failing compile must not allocate a shader and
                // must still be reported as an error.
                NativeShaderCompileInfo broken;
                broken.language = NativeShaderLanguage::HLSL;
                broken.source = "this is not valid HLSL";
                auto failed = ext->compile(broken);
                expect(!failed.ok());
                expect(!failed.error.empty());
            };
            // R16: an application that forgets to destroy a native shader must
            // not crash the device teardown - the extension releases the
            // remaining instances while the backend device is still alive and
            // reports them. This test intentionally leaves one handle behind
            // (the warning it produces is the expected outcome); if the
            // teardown ordering regresses, the process crashes here.
            "native_shader_leftover_handle_at_teardown"_test = [&] {
                auto m = ext->load(result);
                expect(m.valid());
                LUISA_INFO("leaving native shader instance {:#x} for the device "
                           "teardown to release (expected warning)",
                           m.handle);
            };
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    log_level_verbose();
    test_native_shader(dc->device);
    return 0;
}
