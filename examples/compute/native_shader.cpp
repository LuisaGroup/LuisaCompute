// Native shader injection (HLSL on dx, GLSL on vk) through NativeShaderExt.
//
//   example_native_shader dx
//   example_native_shader vk
//
// The example compiles a native compute shader at runtime, reflects its
// bindings, creates a backend shader instance from the compiled bytecode and
// dispatches it - twice, with an independent DSL kernel in between - to show
// that the declared per-argument usages keep the command-reorder pass correct.
// Buffers are read back and verified on the host; the process exits non-zero
// when a check fails.
#include <luisa/backends/ext/command_reorder_ext.h>
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

// `dst[i] = src[i] * k + c`, with a non-`main` entry point, a structured buffer
// pair and a uniform block.
//
//  * dx: the uniform block is the shader's `cbuffer ... : register(b0)`, fed by
//        root 32-bit constants built from the launcher's `add_uniform` values.
//  * vk: the GLSL source below spells out the (set, binding) pairs, and its
//        `layout(push_constant)` block is fed by the same `add_uniform` values.
constexpr auto hlsl_source = R"(
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * k + c;
}
)";

// The same shader in GLSL (Vulkan): the reflected SPIR-V of this source is what
// the pipeline layout and the dispatch bindings are built from.
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

constexpr auto element_count = 1u << 16u;

[[nodiscard]] int fail(const char *message) noexcept {
    LUISA_ERROR("native shader example FAILED: {}", message);
    return 1;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <dx|vk>",
                   argc > 0 ? argv[0] : "example_native_shader");
        return 1;
    }
    auto backend = luisa::string_view{argv[1]};
    if (backend != "dx" && backend != "vk") {
        LUISA_INFO("This example demonstrates the dx and vk backends only.");
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        LUISA_WARNING("Backend '{}' has no NativeShaderExt.", backend);
        return 1;
    }
    if (backend == "dx") {
        // dx has no GLSL front end: the extension refuses GLSL explicitly.
        NativeShaderCompileInfo info;
        info.language = NativeShaderLanguage::GLSL;
        info.source = glsl_source;
        auto rejected = ext->compile(info);
        if (rejected.ok() || rejected.error.empty()) {
            return fail("dx did not reject GLSL");
        }
        LUISA_INFO("dx rejects GLSL as expected: {}", rejected.error);
    }

    NativeShaderCompileInfo info;
    info.language = backend == "dx" ? NativeShaderLanguage::HLSL :
                                      NativeShaderLanguage::GLSL;
    info.source = backend == "dx" ? luisa::string_view{hlsl_source} :
                                    luisa::string_view{glsl_source};
    info.entry_point = backend == "dx" ? "CSMain" : "main";
    info.push_constant_size = 2u * sizeof(float);
    auto compiled = ext->compile(info);
    if (!compiled.ok()) { return fail(compiled.error.c_str()); }
    LUISA_INFO("compiled a native {} shader: {} bytes, workgroup size ({} {} {}), "
               "{} reflected binding(s)",
               backend == "dx" ? "HLSL" : "GLSL", compiled.binary.size(),
               compiled.block_size.x, compiled.block_size.y, compiled.block_size.z,
               compiled.bindings.size());
    for (auto i = 0u; i < compiled.bindings.size(); i++) {
        auto &&binding = compiled.bindings[i];
        LUISA_INFO("  binding {}: set {} register {} array {} usage {}",
                   i, binding.space_index, binding.register_index,
                   binding.array_size, luisa::to_underlying(binding.usage));
    }
    auto metadata = ext->load(compiled);
    if (!metadata.valid()) { return fail("load() returned an invalid shader"); }
    NativeShader shader{*ext, std::move(metadata)};
    Stream stream = device.create_stream();

    luisa::vector<float> host_src(element_count);
    for (auto i = 0u; i < element_count; i++) {
        host_src[i] = static_cast<float>(i & 0xffffu) * 0.5f;
    }
    Buffer<float> src = device.create_buffer<float>(element_count);
    Buffer<float> intermediate = device.create_buffer<float>(element_count);
    Buffer<float> dst = device.create_buffer<float>(element_count);
    Buffer<float> independent = device.create_buffer<float>(element_count);
    stream << src.copy_from(luisa::span{host_src}) << synchronize();

    // Binding by index is unambiguous on both backends: on dx the HLSL register
    // namespaces make `register(t0)` and `register(b0)` the same bind point.
    auto make_dispatch = [&](const BufferView<float> &from,
                             const BufferView<float> &to,
                             float k, float c) noexcept {
        auto launcher = shader.launcher();
        launcher.add_buffer_by_index(0u, from, Usage::READ)
            .add_buffer_by_index(1u, to, Usage::WRITE)
            .add_uniform(k)
            .add_uniform(c);
        // `validate()` reports the same contract checks `build()` asserts, so a
        // caller can fail softly before committing to a dispatch.
        if (auto error = launcher.validate(); !error.empty()) {
            LUISA_ERROR("native shader launcher is not dispatchable: {}", error);
        }
        return std::move(launcher).build(uint3{element_count, 1u, 1u});
    };

    Kernel1D mark = [](BufferFloat out) noexcept {
        out->write(dispatch_id().x, cast<float>(dispatch_id().x) + 7.0f);
    };
    auto mark_shader = device.compile(mark);

    CommandList cmdlist;
    // producer: src -> intermediate (WRITE on intermediate)
    cmdlist << make_dispatch(src.view(), intermediate.view(), 2.0f, 1.0f);
    // an unrelated DSL dispatch that must not be ordered against the native ones
    cmdlist << mark_shader(independent.view()).dispatch(element_count);
    // consumer: intermediate -> dst (READ on intermediate, WRITE on dst)
    cmdlist << make_dispatch(intermediate.view(), dst.view(), 3.0f, 0.5f);
    Clock clock;
    stream << cmdlist.commit() << synchronize();
    LUISA_INFO("dispatched the reorder chain in {:.3f} ms", clock.toc());

    luisa::vector<float> host_dst(element_count);
    luisa::vector<float> host_independent(element_count);
    stream << dst.copy_to(luisa::span{host_dst})
           << independent.copy_to(luisa::span{host_independent})
           << synchronize();
    for (auto i = 0u; i < element_count; i++) {
        auto expected = (host_src[i] * 2.0f + 1.0f) * 3.0f + 0.5f;
        if (host_dst[i] != expected) {
            LUISA_ERROR("mismatch at {}: {} != {}", i, host_dst[i], expected);
            return 1;
        }
        if (host_independent[i] != static_cast<float>(i) + 7.0f) {
            LUISA_ERROR("independent dispatch mismatch at {}", i);
            return 1;
        }
    }
    LUISA_INFO("reorder chain verified ({} elements)", element_count);

    // Reordering off must produce the same results: the declared usages are the
    // synchronization contract, and this example declares them correctly.
    if (auto reorder = device.extension<CommandReorderExt>(); reorder != nullptr) {
        reorder->set_command_reorder_enabled(false);
        CommandList second;
        second << make_dispatch(src.view(), dst.view(), 4.0f, 0.0f);
        stream << second.commit() << synchronize();
        stream << dst.copy_to(luisa::span{host_dst}) << synchronize();
        for (auto i = 0u; i < element_count; i++) {
            if (host_dst[i] != host_src[i] * 4.0f) {
                LUISA_ERROR("reorder-off mismatch at {}: {} != {}",
                            i, host_dst[i], host_src[i] * 4.0f);
                return 1;
            }
        }
        reorder->set_command_reorder_enabled(true);
        LUISA_INFO("reordering disabled: results identical");
    }
    LUISA_INFO("native shader example PASSED on '{}'", backend);
    return 0;
}
