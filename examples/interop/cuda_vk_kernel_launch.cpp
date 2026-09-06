// Example: compile CUDA kernels with NVRTC and dispatch them *inside a Vulkan
// backend stream* through the VK_NV_cuda_kernel_launch path exposed by
// luisa::compute::VkCudaInterop.
//
// This is the documentation-grade counterpart of
// src/tests/integration/runtime/test_vk_cuda_kernel_launch.cpp:
//   1. run_saxpy                  - single CUDA launch streamed via `stream << ...`
//   2. run_block_reduce           - multi-block reduction with shared memory
//   3. run_chained_command_list   - two chained CUDA launches in one CommandList
//   4. run_dsl_interop            - DSL dispatches before/after a CUDA launch
//                                   on the same VK stream (barrier correctness)
//   5. run_histogram              - integer output buffer + atomicAdd interop,
//                                   using the DSL-style typed CudaKernelT API
//   6. run_ptx_roundtrip          - compile to PTX, recreate the shader from PTX
//
// Argument packing convention (see include/luisa/backends/ext/vk_cuda_interop.h):
//   - buffer arguments arrive as raw 64-bit device addresses, so CUDA kernel
//     parameters must be plain pointers (e.g. `float *data`);
//   - uniform arguments are passed by value in argument order;
//   - kernels must be `extern "C"` (NVRTC otherwise mangles the entry name).
//
// Two call styles are demonstrated:
//   - manual:  KernelLauncher{}.add_buffer(...).add_uniform(...)
//                .build(shader, grid, block, shared)
//   - DSL-like typed API (scenarios 1 & 5): argument usages are declared
//     once in the signature (CudaArg<T, Usage>) and baked into the kernel
//     instance, so the dispatch site passes bare views only:
//       auto saxpy = ext->create_cuda_kernel(
//                        {.source = ..., .kernel_name = "saxpy"})
//                        .kernel<CudaArg<Buffer<float>, Usage::READ>,
//                                CudaArg<Buffer<float>, Usage::READ_WRITE>,
//                                float, uint32_t>();
//       stream << saxpy(x.view(), y.view(), a, n)
//                   .dispatch(n, uint3{k_block_size, 1u, 1u});
//
// Usage: cuda_vk_kernel_launch [backend]   (backend defaults to "vk")
// Skips gracefully (exit code 0) when VK_NV_cuda_kernel_launch is unavailable.

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/vk_cuda_interop.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr uint32_t k_block_size = 256u;

[[nodiscard]] constexpr uint3 grid_for(uint32_t n) noexcept {
    return uint3{(n + k_block_size - 1u) / k_block_size, 1u, 1u};
}

// ---------------------------------------------------------------------------
// CUDA kernel sources (NVRTC-compiled at runtime)
// ---------------------------------------------------------------------------

constexpr luisa::string_view saxpy_source = R"(
extern "C" __global__ void saxpy(const float *x, float *y, float a, unsigned int n) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
)";

constexpr luisa::string_view scale_source = R"(
extern "C" __global__ void scale(float *data, float factor, unsigned int n) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * factor;
    }
}
)";

// Per-block partial sums with a static shared-memory tree reduction.
// Launch with exactly ceil(n / block_elems) blocks.
constexpr luisa::string_view reduce_sum_source = R"(
extern "C" __global__ void reduce_sum(const float *data, float *partial_sums,
                                      unsigned int n, unsigned int block_elems) {
    __shared__ float tile[256];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * block_elems + tid;
    auto v = (i < n) ? data[i] : 0.0f;
    tile[tid] = v;
    __syncthreads();
    for (auto s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        partial_sums[blockIdx.x] = tile[0];
    }
}
)";

constexpr luisa::string_view histogram_source = R"(
extern "C" __global__ void histogram(const float *data, unsigned int *bins,
                                     unsigned int n, unsigned int bin_count) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto v = data[i];
        auto bin = (unsigned int)(v < 0.0f ? 0.0f : v);
        if (bin >= bin_count) { bin = bin_count - 1u; }
        atomicAdd(&bins[bin], 1u);
    }
}
)";

constexpr luisa::string_view fill_source = R"(
extern "C" __global__ void fill(float *data, float value, unsigned int n) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}
)";

// ---------------------------------------------------------------------------
// Host verification helpers (examples use logging, not Boost.UT)
// ---------------------------------------------------------------------------

[[nodiscard]] bool check_span_near(luisa::span<const float> got,
                                   luisa::span<const float> expected,
                                   float eps,
                                   luisa::string_view what) noexcept {
    if (got.size() != expected.size()) {
        LUISA_ERROR("FAIL [{}]: size mismatch: got {} want {}",
                    what, got.size(), expected.size());
        return false;
    }
    for (auto i = 0u; i < got.size(); ++i) {
        if (std::abs(got[i] - expected[i]) > eps) {
            LUISA_ERROR("FAIL [{}]: index {} got {} want {}",
                        what, i, got[i], expected[i]);
            return false;
        }
    }
    LUISA_INFO("PASS [{}]: {} elements verified (eps={}), sample [0]={} [last]={}",
               what, got.size(), eps, got[0], got[got.size() - 1u]);
    return true;
}

[[nodiscard]] bool check_near(float got, float expected, float eps,
                              luisa::string_view what) noexcept {
    if (std::abs(got - expected) > eps) {
        LUISA_ERROR("FAIL [{}]: got {} want {}", what, got, expected);
        return false;
    }
    LUISA_INFO("PASS [{}]: got {} want {} (eps={})", what, got, expected, eps);
    return true;
}

[[nodiscard]] bool check_span_eq(luisa::span<const uint> got,
                                 luisa::span<const uint> expected,
                                 luisa::string_view what) noexcept {
    if (got.size() != expected.size()) {
        LUISA_ERROR("FAIL [{}]: size mismatch", what);
        return false;
    }
    for (auto i = 0u; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            LUISA_ERROR("FAIL [{}]: bin {} got {} want {}",
                        what, i, got[i], expected[i]);
            return false;
        }
    }
    LUISA_INFO("PASS [{}]: {} bins verified exactly", what, got.size());
    return true;
}

// ---------------------------------------------------------------------------
// Scenario 1: single SAXPY launch streamed directly into the VK stream
// (DSL-style typed API: create_cuda_kernel + kernel<Args...>() + .dispatch())
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_saxpy(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;

    auto stream = device.create_stream();
    auto x = ext->create_buffer<float>(n);
    auto y = ext->create_buffer<float>(n);
    luisa::vector<float> host_x(n);
    luisa::vector<float> host_y(n, 1.0f);
    luisa::vector<float> host_out(n, -1.0f);
    for (auto i = 0u; i < n; ++i) { host_x[i] = static_cast<float>(i); }
    stream << x.view().copy_from(luisa::span{host_x})
           << y.view().copy_from(luisa::span{host_y})
           << synchronize();

    auto shader = ext->create_cuda_kernel(
        {.source = saxpy_source, .kernel_name = "saxpy"});
    if (!shader) {
        LUISA_WARNING("[saxpy] create_cuda_kernel failed; skipping.");
        return false;
    }
    // Typed DSL-style invocation: signature checked at compile time; the
    // per-argument usage is baked into the kernel instance at creation, so
    // the dispatch site passes bare views, uniforms by value.
    auto saxpy = shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                               vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                               float, uint32_t>();
    stream << saxpy(x.view(), y.view(), a, n)
                 .dispatch(n, uint3{k_block_size, 1u, 1u})
           << y.view().copy_to(luisa::span{host_out})
           << synchronize();
    // No manual destroy_cuda_kernel_shader: CudaShader is RAII.

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) { expected[i] = a * host_x[i] + host_y[i]; }
    return check_span_near(host_out, expected, 1e-5f, "saxpy");
}

// ---------------------------------------------------------------------------
// Scenario 2: multi-block shared-memory reduction (saxpy -> partial sums ->
// host reduce)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_block_reduce(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 4096u;
    // One float per thread per block => block_elems == k_block_size.
    auto num_blocks = (n + k_block_size - 1u) / k_block_size;

    auto stream = device.create_stream();
    auto data = ext->create_buffer<float>(n);
    auto partials = ext->create_buffer<float>(num_blocks);
    luisa::vector<float> host_data(n);
    luisa::vector<float> host_partials(num_blocks, -1.0f);
    for (auto i = 0u; i < n; ++i) {
        host_data[i] = static_cast<float>(i % 97) * 0.25f;
    }
    stream << data.view().copy_from(luisa::span{host_data})
           << synchronize();

    auto reduce_shader = ext->create_cuda_kernel_shader(
        {.source = reduce_sum_source, .kernel_name = "reduce_sum"});
    if (reduce_shader == 0u) {
        LUISA_WARNING("[block_reduce] shader compilation failed; skipping.");
        return false;
    }

    vk_cuda_interop::KernelLauncher launcher;
    launcher.add_buffer(data.view(), Usage::READ)
        .add_buffer(partials.view(), Usage::WRITE)
        .add_uniform(n)
        .add_uniform(k_block_size);
    stream << std::move(launcher).build(
                  reduce_shader, uint3{num_blocks, 1u, 1u},
                  uint3{k_block_size, 1u, 1u}, 0u)
           << partials.view().copy_to(luisa::span{host_partials})
           << synchronize();
    ext->destroy_cuda_kernel_shader(reduce_shader);

    auto gpu_total = 0.0;
    for (auto p : host_partials) { gpu_total += static_cast<double>(p); }
    auto cpu_total = 0.0;
    for (auto v : host_data) { cpu_total += static_cast<double>(v); }
    // Scale epsilon by the magnitude of the sum (float accumulation error).
    auto eps = static_cast<float>(std::abs(cpu_total) * 1e-4 + 1e-3);
    auto ok = check_near(static_cast<float>(gpu_total),
                         static_cast<float>(cpu_total), eps, "block_reduce");
    LUISA_INFO("[block_reduce] {} partial sums, gpu_total={}, cpu_total={}",
               num_blocks, gpu_total, cpu_total);
    return ok;
}

// ---------------------------------------------------------------------------
// Scenario 3: two chained CUDA launches in one CommandList (saxpy -> scale)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_chained_command_list(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto factor = 3.0f;

    auto saxpy_shader = ext->create_cuda_kernel_shader(
        {.source = saxpy_source, .kernel_name = "saxpy"});
    auto scale_shader = ext->create_cuda_kernel_shader(
        {.source = scale_source, .kernel_name = "scale"});
    if (saxpy_shader == 0u || scale_shader == 0u) {
        LUISA_WARNING("[chained] shader compilation failed; skipping.");
        if (saxpy_shader != 0u) { ext->destroy_cuda_kernel_shader(saxpy_shader); }
        if (scale_shader != 0u) { ext->destroy_cuda_kernel_shader(scale_shader); }
        return false;
    }

    auto stream = device.create_stream();
    auto x = ext->create_buffer<float>(n);
    auto y = ext->create_buffer<float>(n);
    luisa::vector<float> host_x(n);
    luisa::vector<float> host_y(n, 1.0f);
    luisa::vector<float> host_out(n, 0.0f);
    for (auto i = 0u; i < n; ++i) { host_x[i] = static_cast<float>(i); }

    // First launch: y = a * x + y. Second launch consumes its output:
    // y = y * factor.
    vk_cuda_interop::KernelLauncher saxpy_launcher;
    saxpy_launcher.add_buffer(x.view(), Usage::READ)
        .add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(a)
        .add_uniform(n);
    vk_cuda_interop::KernelLauncher scale_launcher;
    scale_launcher.add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(factor)
        .add_uniform(n);

    CommandList cmdlist;
    cmdlist << x.view().copy_from(luisa::span{host_x})
            << y.view().copy_from(luisa::span{host_y});
    cmdlist << std::move(saxpy_launcher).build(
        saxpy_shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << std::move(scale_launcher).build(
        scale_shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << y.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();
    ext->destroy_cuda_kernel_shader(saxpy_shader);
    ext->destroy_cuda_kernel_shader(scale_shader);

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = factor * (a * host_x[i] + host_y[i]);
    }
    return check_span_near(host_out, expected, 1e-5f, "chained_command_list");
}

// ---------------------------------------------------------------------------
// Scenario 4: DSL kernels and a CUDA launch interleaved on the same VK stream
// (validates usage/barrier tracking across DSL and CUDA commands)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_dsl_interop(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto y_init = 1.0f;
    constexpr auto y_post = 10.0f;

    auto shader = ext->create_cuda_kernel_shader(
        {.source = saxpy_source, .kernel_name = "saxpy"});
    if (shader == 0u) {
        LUISA_WARNING("[dsl_interop] shader compilation failed; skipping.");
        return false;
    }

    // DSL kernels running on the same Vulkan stream, before and after the
    // CUDA launch.
    Kernel1D index_kernel = [](BufferFloat buffer) {
        auto i = dispatch_x();
        buffer->write(i, cast<float>(i));
    };
    Kernel1D fill_kernel = [](BufferFloat buffer, Float value) {
        buffer->write(dispatch_x(), value);
    };
    Kernel1D add_kernel = [](BufferFloat buffer, Float value) {
        auto i = dispatch_x();
        buffer->write(i, buffer->read(i) + value);
    };
    auto index_shader = device.compile(index_kernel);
    auto fill_shader = device.compile(fill_kernel);
    auto add_shader = device.compile(add_kernel);

    auto stream = device.create_stream();
    auto x = ext->create_buffer<float>(n);
    auto y = ext->create_buffer<float>(n);
    luisa::vector<float> host_out(n, 0.0f);

    vk_cuda_interop::KernelLauncher launcher;
    launcher.add_buffer(x.view(), Usage::READ)
        .add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(a)
        .add_uniform(n);

    CommandList cmdlist;
    cmdlist << index_shader(x.view()).dispatch(n)
            << fill_shader(y.view(), y_init).dispatch(n);
    cmdlist << std::move(launcher).build(
        shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << add_shader(y.view(), y_post).dispatch(n)
            << y.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();
    ext->destroy_cuda_kernel_shader(shader);

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = a * static_cast<float>(i) + y_init + y_post;
    }
    return check_span_near(host_out, expected, 1e-5f, "dsl_interop");
}

// ---------------------------------------------------------------------------
// Scenario 5: histogram with atomicAdd into an integer output buffer
// (DSL-style typed API with a uint buffer argument)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_histogram(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    static constexpr uint32_t bin_count = 16u;

    auto shader = ext->create_cuda_kernel(
        {.source = histogram_source, .kernel_name = "histogram"});
    if (!shader) {
        LUISA_WARNING("[histogram] shader compilation failed; skipping.");
        return false;
    }
    auto histogram = shader.kernel<
        vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
        uint32_t, uint32_t>();

    auto stream = device.create_stream();
    auto data = ext->create_buffer<float>(n);
    auto bins = ext->create_buffer<uint>(bin_count);
    luisa::vector<float> host_data(n);
    luisa::vector<uint> host_bins(bin_count, 0u);
    luisa::vector<uint> zero_bins(bin_count, 0u);
    for (auto i = 0u; i < n; ++i) {
        // Values in [0, bin_count); floor(v) is the bin index.
        host_data[i] = static_cast<float>(i % bin_count) + 0.5f;
    }

    CommandList cmdlist;
    cmdlist << data.view().copy_from(luisa::span{host_data})
            << bins.view().copy_from(luisa::span{zero_bins});
    cmdlist << histogram(data.view(), bins.view(), n, bin_count)
                 .dispatch(grid_for(n), uint3{k_block_size, 1u, 1u});
    cmdlist << bins.view().copy_to(luisa::span{host_bins});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<uint> expected(bin_count, 0u);
    for (auto i = 0u; i < n; ++i) {
        expected[static_cast<uint>(std::floor(host_data[i]))] += 1u;
    }
    return check_span_eq(host_bins, expected, "histogram");
}

// ---------------------------------------------------------------------------
// Scenario 6: PTX roundtrip — compile to PTX, recreate the shader from PTX
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_ptx_roundtrip(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 512u;
    constexpr auto value = 42.5f;

    luisa::vector<char> ptx;
    auto shader_from_source = ext->create_cuda_kernel_shader(
        {.source = fill_source, .kernel_name = "fill", .ptx_output = &ptx});
    if (shader_from_source == 0u || ptx.empty()) {
        LUISA_WARNING("[ptx_roundtrip] PTX compilation failed; skipping.");
        if (shader_from_source != 0u) {
            ext->destroy_cuda_kernel_shader(shader_from_source);
        }
        return false;
    }
    LUISA_INFO("[ptx_roundtrip] NVRTC produced {} bytes of PTX", ptx.size());

    // Rebuild the shader from the PTX text.
    auto shader_from_ptx = ext->create_cuda_kernel_shader(
        {.source = luisa::string_view{ptx.data(), ptx.size()},
         .kernel_name = "fill",
         .source_is_ptx = true});
    if (shader_from_ptx == 0u) {
        LUISA_WARNING("[ptx_roundtrip] shader creation from PTX failed; skipping.");
        ext->destroy_cuda_kernel_shader(shader_from_source);
        return false;
    }

    auto stream = device.create_stream();
    auto buffer = ext->create_buffer<float>(n);
    luisa::vector<float> host_out(n, 0.0f);

    vk_cuda_interop::KernelLauncher launcher;
    launcher.add_buffer(buffer.view(), Usage::WRITE)
        .add_uniform(value)
        .add_uniform(n);
    stream << std::move(launcher).build(
                  shader_from_ptx, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u)
           << buffer.view().copy_to(luisa::span{host_out})
           << synchronize();
    ext->destroy_cuda_kernel_shader(shader_from_ptx);
    ext->destroy_cuda_kernel_shader(shader_from_source);

    luisa::vector<float> expected(n, value);
    return check_span_near(host_out, expected, 0.0f, "ptx_roundtrip");
}

}// namespace

int main(int argc, char *argv[]) {
    auto backend = luisa::string_view{"vk"};
    if (argc > 1 && argv[1] != nullptr) { backend = argv[1]; }

    Context context{argv[0]};
    Device device = context.create_device(backend);
    auto ext = device.extension<VkCudaInterop>();
    if (ext == nullptr) {
        LUISA_WARNING("VkCudaInterop extension is not available; skipping example.");
        return 0;
    }
    if (!ext->cuda_kernel_launch_supported()) {
        LUISA_WARNING("VK_NV_cuda_kernel_launch is not supported on this device; skipping example.");
        return 0;
    }

    // Non-short-circuit aggregation: run every scenario, report all failures.
    auto ok = true;
    ok = run_saxpy(device, ext) && ok;
    ok = run_block_reduce(device, ext) && ok;
    ok = run_chained_command_list(device, ext) && ok;
    ok = run_dsl_interop(device, ext) && ok;
    ok = run_histogram(device, ext) && ok;
    ok = run_ptx_roundtrip(device, ext) && ok;

    if (ok) {
        LUISA_INFO("All CUDA-in-VK-stream dispatches verified.");
    } else {
        LUISA_ERROR("Verification FAILED.");
    }
    return ok ? 0 : 1;
}
