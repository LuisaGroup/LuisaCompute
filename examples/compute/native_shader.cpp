// Native shader injection (HLSL on dx, GLSL on vk, CUDA C++ on cuda) through
// NativeShaderExt.
//
//   example_native_shader dx
//   example_native_shader vk
//   example_native_shader cuda
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
#include <luisa/core/stl/filesystem.h>
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

// The same shader again, this time in CUDA C++ for the CUDA backend. NVRTC
// compiles it and the extension reflects the kernel signature: the pointer
// parameters are the buffer bindings, in declaration order (`const float *src`
// is read-only, `float *dst` is writable), and the non-pointer parameters are
// the launcher's `add_uniform` values, in declaration order - exactly the two
// buffers and the two floats the other routes bind. The kernel is declared
// `extern "C"` so that its name reaches the PTX unmangled.
constexpr auto cuda_source = R"(
extern "C" __global__ void scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] * k + c;
}
)";

// The include-directory demonstration below compiles these sources, which
// `#include` a header from a scratch directory passed via `include_dirs` (and,
// for the file-source route, resolved relative to the source file itself).
// The helper is a plain free function - valid HLSL, GLSL 450 and CUDA C++.
// NVRTC's JIT mode only allows execution-space-annotated functions, so under
// `__CUDACC__` the helper is marked `__host__ __device__ inline`.
constexpr auto math_header = R"(
#if defined(__CUDACC__)
#define LUISA_EXAMPLE_TRANSFORM __host__ __device__ inline
#else
#define LUISA_EXAMPLE_TRANSFORM
#endif
LUISA_EXAMPLE_TRANSFORM float luisa_example_transform(float x, float k, float c) { return x * k + c; }
#undef LUISA_EXAMPLE_TRANSFORM
)";

constexpr auto hlsl_include_source = R"(
#include "native_shader_math.h"
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = luisa_example_transform(src[tid.x], k, c);
}
)";

// GLSL requires `#extension GL_GOOGLE_include_directive` (after `#version`,
// before any `#include`) to enable the include directive.
constexpr auto glsl_include_source = R"(
#version 450
#extension GL_GOOGLE_include_directive : require
#include "native_shader_math.h"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float k; float c; } uniforms;
void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = luisa_example_transform(src.a[i], uniforms.k, uniforms.c);
}
)";

constexpr auto cuda_include_source = R"(
#include "native_shader_math.h"
extern "C" __global__ void include_scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = luisa_example_transform(src[i], k, c);
}
)";

[[nodiscard]] int fail(const char *message) noexcept {
    LUISA_ERROR("native shader example FAILED: {}", message);
    return 1;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <dx|vk|cuda>",
                   argc > 0 ? argv[0] : "example_native_shader");
        return 1;
    }
    auto backend = luisa::string_view{argv[1]};
    if (backend != "dx" && backend != "vk" && backend != "cuda") {
        LUISA_INFO("This example demonstrates the dx, vk and cuda backends only.");
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        LUISA_WARNING("Backend '{}' has no NativeShaderExt.", backend);
        return 1;
    }
    auto language_name = [&] {
        return backend == "dx" ? "HLSL" : backend == "vk" ? "GLSL" : "CUDA";
    };
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
    } else if (backend == "cuda") {
        // cuda has no HLSL/GLSL front end either: the extension refuses both and
        // compiles CUDA C++ (NativeShaderLanguage::CUDA_NVRTC) instead.
        NativeShaderCompileInfo info;
        info.language = NativeShaderLanguage::HLSL;
        info.source = hlsl_source;
        auto rejected = ext->compile(info);
        if (rejected.ok() || rejected.error.empty()) {
            return fail("cuda did not reject HLSL");
        }
        LUISA_INFO("cuda rejects HLSL as expected: {}", rejected.error);
    }

    NativeShaderCompileInfo info;
    info.language = backend == "dx" ? NativeShaderLanguage::HLSL :
                    backend == "vk" ? NativeShaderLanguage::GLSL :
                                       NativeShaderLanguage::CUDA_NVRTC;
    info.source = backend == "dx" ? luisa::string_view{hlsl_source} :
                  backend == "vk" ? luisa::string_view{glsl_source} :
                                     luisa::string_view{cuda_source};
    info.entry_point = backend == "dx" ? "CSMain" :
                       backend == "vk" ? "main" : "scale";
    // The uniform bytes: a `cbuffer`/push-constant block on dx/vk, the two
    // scalar kernel parameters on cuda (where the reflection also derives it, so
    // this line is a cross-check rather than a requirement).
    info.push_constant_size = 2u * sizeof(float);
    if (backend == "cuda") {
        // CUDA kernels declare no workgroup size; the launcher's block size must
        // be given (or derived from `__launch_bounds__(N)`).
        info.block_size = uint3{64u, 1u, 1u};
    }
    auto compiled = ext->compile(info);
    if (!compiled.ok()) { return fail(compiled.error.c_str()); }
    LUISA_INFO("compiled a native {} shader: {} bytes, workgroup size ({} {} {}), "
               "{} reflected binding(s)",
               language_name(), compiled.binary.size(),
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

    // Binding by index is unambiguous on every route: on dx the HLSL register
    // namespaces make `register(t0)` and `register(b0)` the same bind point.
    auto make_dispatch = [&](const NativeShader &shader,
                             const BufferView<float> &from,
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
    cmdlist << make_dispatch(shader, src.view(), intermediate.view(), 2.0f, 1.0f);
    // an unrelated DSL dispatch that must not be ordered against the native ones
    cmdlist << mark_shader(independent.view()).dispatch(element_count);
    // consumer: intermediate -> dst (READ on intermediate, WRITE on dst)
    cmdlist << make_dispatch(shader, intermediate.view(), dst.view(), 3.0f, 0.5f);
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
        second << make_dispatch(shader, src.view(), dst.view(), 4.0f, 0.0f);
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

    // ---- include directories & file sources ------------------------------
    // The same `dst[i] = src[i] * k + c` shader, but its body comes from a
    // header resolved through `NativeShaderCompileInfo::include_dirs` (and,
    // for a `FilePath` source, relative to the source file's own directory).
    {
        namespace fs = luisa::filesystem;
        std::error_code ec;
        auto scratch_dir = fs::temp_directory_path(ec) / "luisa_native_shader_example";
        if (ec) { return fail("cannot locate the temp directory"); }
        fs::create_directories(scratch_dir, ec);
        if (ec) { return fail("cannot create the scratch directory"); }
        {
            std::ofstream out{scratch_dir / "native_shader_math.h",
                              std::ios::out | std::ios::trunc};
            if (!out.is_open()) { return fail("cannot write native_shader_math.h"); }
            out << math_header;
        }
        luisa::string_view include_source = backend == "dx" ? hlsl_include_source :
                                            backend == "vk" ? glsl_include_source :
                                                              cuda_include_source;
        auto source_name = backend == "dx" ? "native_shader_include.hlsl" :
                           backend == "vk" ? "native_shader_include.glsl" :
                                             "native_shader_include.cu";
        // `info.source` is a string_view: the path string must outlive the
        // compile call.
        auto source_path = luisa::to_string(scratch_dir / source_name);
        {
            std::ofstream out{fs::path{source_path},
                              std::ios::out | std::ios::trunc};
            if (!out.is_open()) { return fail("cannot write the include shader source"); }
            out.write(include_source.data(),
                      static_cast<std::streamsize>(include_source.size()));
        }

        NativeShaderCompileInfo include_info;
        include_info.language = info.language;
        include_info.entry_point = backend == "dx" ? "CSMain" :
                                   backend == "vk" ? "main" : "include_scale";
        include_info.push_constant_size = 2u * sizeof(float);
        if (backend == "cuda") {
            include_info.block_size = uint3{64u, 1u, 1u};
        }
        // Route 1: a source *file*; headers resolve through `include_dirs`
        // and relative to the file's own directory.
        include_info.source_type = NativeShaderSourceType::FilePath;
        include_info.source = luisa::string_view{source_path};
        include_info.include_dirs = {scratch_dir};
        auto from_file = ext->compile(include_info);
        if (!from_file.ok()) { return fail(from_file.error.c_str()); }
        LUISA_INFO("compiled {} from a file with an include directory",
                   language_name());
        // Route 2: the same source in memory, still using `include_dirs`.
        include_info.source_type = NativeShaderSourceType::SourceCode;
        include_info.source = include_source;
        auto from_source = ext->compile(include_info);
        if (!from_source.ok()) { return fail(from_source.error.c_str()); }
        LUISA_INFO("compiled {} from memory with an include directory",
                   language_name());
        // A directory that does not exist must fail closed with an error. The
        // CUDA route cannot exercise this: the standalone NVRTC compiler
        // process aborts on any compile error (that is how the CUDA backend
        // has always reported NVRTC failures), so error-message validation is
        // only meaningful on the DXC (dx/vk) and glslang (vk) routes.
        if (backend != "cuda") {
            include_info.include_dirs = {scratch_dir / "no_such_directory"};
            auto bogus = ext->compile(include_info);
            if (bogus.ok() || bogus.error.empty()) {
                return fail("a bogus include directory was not reported");
            }
            LUISA_INFO("bogus include directory rejected as expected: {}", bogus.error);
        }

        auto metadata = ext->load(from_file);
        if (!metadata.valid()) { return fail("load() returned an invalid shader"); }
        NativeShader include_shader{*ext, std::move(metadata)};
        Buffer<float> include_dst = device.create_buffer<float>(element_count);
        stream << make_dispatch(include_shader, src.view(), include_dst.view(),
                                5.0f, 2.0f)
               << synchronize();
        stream << include_dst.copy_to(luisa::span{host_dst}) << synchronize();
        for (auto i = 0u; i < element_count; i++) {
            auto expected = host_src[i] * 5.0f + 2.0f;
            if (host_dst[i] != expected) {
                LUISA_ERROR("include-directory dispatch mismatch at {}: {} != {}",
                            i, host_dst[i], expected);
                return 1;
            }
        }
        LUISA_INFO("include-directory dispatch verified ({} elements)", element_count);
    }
    LUISA_INFO("native shader example PASSED on '{}'", backend);
    return 0;
}
