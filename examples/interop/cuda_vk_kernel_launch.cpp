// Example: compile DSL kernels with the *cuda backend* and dispatch them
// *inside a Vulkan backend stream* through VK_NV_cuda_kernel_launch, exposed
// by luisa::compute::VkCudaInterop.
//
// Flow: create the vk device, query the LUID-matched cuda device index via
// ext->cuda_device_index(), create a cuda device on the same physical GPU,
// compile DSL kernels with it (cuda_device.compile(kernel)), and import them
// into the vk backend with ext->create_cuda_kernel(shader). Both the
// luisa-backend-vk and luisa-backend-cuda plugins must be present at runtime.
//
// Launch ABI (see include/luisa/backends/ext/vk_cuda_interop.h): at encode
// time the vk backend packs the arguments into the DSL kernel ABI — a single
// by-value `struct Params` blob with 16-byte-aligned slots (buffers as
// LCBuffer { ptr, size_bytes } bindings, uniforms as raw values) and an
// `ls_kid` uint4 trailer carrying the exact dispatch size. Dispatch with the
// exact-thread-count overloads (a bare uint3 thread count) so `ls_kid`
// matches the DSL launch semantics. Imported kernels use static __shared__
// memory only, and the launch block dimension defaults to the compiled
// kernel's set_block_size().
//
// This is the documentation-grade counterpart of
// src/tests/integration/runtime/test_vk_cuda_kernel_launch.cpp:
//   1. run_saxpy                - single CUDA launch streamed via `stream << ...`
//   2. run_block_reduce         - multi-block reduction with shared memory
//   3. run_chained_command_list - two chained CUDA launches in one CommandList
//   4. run_dsl_interop          - DSL dispatches before/after an imported
//                                 launch on the same VK stream (barrier
//                                 correctness)
//   5. run_histogram            - integer output buffer + atomics, using the
//                                 DSL-style typed CudaKernelT API
//   6. run_reimport             - import the same cuda shader twice and
//                                 dispatch after the source shader's
//                                 destruction (import self-containedness)
//
// Usage: cuda_vk_kernel_launch [backend]   (backend defaults to "vk")
// Skips gracefully (exit code 0) when VK_NV_cuda_kernel_launch or the cuda
// backend is unavailable.

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

// ---------------------------------------------------------------------------
// DSL kernels (compiled on the cuda device, imported into the vk backend)
// ---------------------------------------------------------------------------
// Every kernel calls set_block_size(k_block_size) so the imported shader
// carries a deterministic block dimension for the thread-count dispatch
// overloads.

[[nodiscard]] auto make_saxpy_kernel() noexcept {
    return Kernel1D{[](BufferFloat x, BufferFloat y, Float a, UInt n) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        if_(i < n, [&] {
            y->write(i, a * x->read(i) + y->read(i));
        });
    }};
}

[[nodiscard]] auto make_scale_kernel() noexcept {
    return Kernel1D{[](BufferFloat data, Float factor, UInt n) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        if_(i < n, [&] {
            data->write(i, data->read(i) * factor);
        });
    }};
}

// Per-block partial sums with a static shared-memory tree reduction.
// The halving strides are compile-time constants, so the loop is unrolled on
// the host into eight device iterations.
[[nodiscard]] auto make_reduce_sum_kernel() noexcept {
    return Kernel1D{[](BufferFloat data, BufferFloat partials, UInt n) {
        set_block_size(k_block_size);
        Shared<float> tile{k_block_size};
        auto tid = thread_id().x;
        auto bid = block_id().x;
        auto i = bid * k_block_size + tid;
        auto v = def(0.0f);
        if_(i < n, [&] { v = data->read(i); });
        tile[tid] = v;
        sync_block();
        for (auto s = k_block_size / 2u; s > 0u; s >>= 1u) {
            if_(tid < s, [&] { tile[tid] = tile[tid] + tile[tid + s]; });
            sync_block();
        }
        if_(tid == 0u, [&] { partials->write(bid, tile[0]); });
    }};
}

[[nodiscard]] auto make_histogram_kernel() noexcept {
    return Kernel1D{[](BufferFloat data, BufferUInt bins, UInt n, UInt bin_count) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        if_(i < n, [&] {
            auto v = data->read(i);
            auto bin = def(min(cast<uint>(max(v, 0.0f)), bin_count - 1u));
            bins->atomic(bin).fetch_add(1u);
        });
    }};
}

[[nodiscard]] auto make_fill_kernel() noexcept {
    return Kernel1D{[](BufferFloat data, Float value, UInt n) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        if_(i < n, [&] { data->write(i, value); });
    }};
}

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
// (DSL-style typed API: create_cuda_kernel + kernel<Args...>() + .dispatch()
// with the imported shader's compiled block size)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_saxpy(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;

    auto cuda_shader = cuda_device.compile(make_saxpy_kernel());
    auto shader = ext->create_cuda_kernel(cuda_shader);
    if (!shader) {
        LUISA_WARNING("[saxpy] create_cuda_kernel failed; skipping.");
        return false;
    }

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

    // Typed DSL-style invocation: signature checked at compile time; the
    // per-argument usage is baked into the kernel instance at creation, so
    // the dispatch site passes bare views, uniforms by value. The block
    // dimension defaults to the imported shader's compiled block size.
    auto saxpy = shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                               vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                               float, uint32_t>();
    stream << saxpy(x.view(), y.view(), a, n)
                  .dispatch(n)
           << y.view().copy_to(luisa::span{host_out})
           << synchronize();
    // No manual destroy_cuda_kernel_shader: CudaShader is RAII.

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) { expected[i] = a * host_x[i] + host_y[i]; }
    return check_span_near(host_out, expected, 1e-5f, "saxpy");
}

// ---------------------------------------------------------------------------
// Scenario 2: multi-block shared-memory reduction (partial sums -> host
// reduce), launched through the DSL-style typed API with an exact thread
// count
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_block_reduce(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 4096u;
    // One float per thread per block => one partial per block.
    auto num_blocks = (n + k_block_size - 1u) / k_block_size;

    auto cuda_shader = cuda_device.compile(make_reduce_sum_kernel());
    auto shader = ext->create_cuda_kernel(cuda_shader);
    if (!shader) {
        LUISA_WARNING("[block_reduce] shader import failed; skipping.");
        return false;
    }

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

    // Typed DSL-style invocation: usages are baked into the kernel instance;
    // the block dimension defaults to the imported shader's compiled block
    // size. Exact-thread-count launch: grid = ceil(n / block), ls_kid = n.
    auto reduce_sum = shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                                    vk_cuda_interop::CudaArg<Buffer<float>, Usage::WRITE>,
                                    uint32_t>();
    stream << reduce_sum(data.view(), partials.view(), n)
                  .dispatch(n)
           << partials.view().copy_to(luisa::span{host_partials})
           << synchronize();

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
// Scenario 3: two chained imported launches in one CommandList (saxpy ->
// scale)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_chained_command_list(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto factor = 3.0f;

    auto saxpy_shader = ext->create_cuda_kernel(cuda_device.compile(make_saxpy_kernel()));
    auto scale_shader = ext->create_cuda_kernel(cuda_device.compile(make_scale_kernel()));
    if (!saxpy_shader || !scale_shader) {
        LUISA_WARNING("[chained] shader import failed; skipping.");
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
    // y = y * factor. Both go through the DSL-style typed API; per-argument
    // usages are baked into the kernel instances at creation.
    auto saxpy = saxpy_shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                                     float, uint32_t>();
    auto scale = scale_shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                                     float, uint32_t>();

    CommandList cmdlist;
    cmdlist << x.view().copy_from(luisa::span{host_x})
            << y.view().copy_from(luisa::span{host_y});
    cmdlist << saxpy(x.view(), y.view(), a, n).dispatch(n);
    cmdlist << scale(y.view(), factor, n).dispatch(n);
    cmdlist << y.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = factor * (a * host_x[i] + host_y[i]);
    }
    return check_span_near(host_out, expected, 1e-5f, "chained_command_list");
}

// ---------------------------------------------------------------------------
// Scenario 4: vk-compiled DSL kernels and an imported CUDA launch interleaved
// on the same VK stream (validates usage/barrier tracking across DSL and
// CUDA commands)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_dsl_interop(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto y_init = 1.0f;
    constexpr auto y_post = 10.0f;

    auto shader = ext->create_cuda_kernel(cuda_device.compile(make_saxpy_kernel()));
    if (!shader) {
        LUISA_WARNING("[dsl_interop] shader import failed; skipping.");
        return false;
    }

    // DSL kernels running on the same Vulkan stream, before and after the
    // imported CUDA launch.
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

    // The imported CUDA kernel uses the same DSL-style typed API as the
    // vk-compiled kernels around it.
    auto saxpy = shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                               vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                               float, uint32_t>();

    CommandList cmdlist;
    cmdlist << index_shader(x.view()).dispatch(n)
            << fill_shader(y.view(), y_init).dispatch(n);
    cmdlist << saxpy(x.view(), y.view(), a, n).dispatch(n);
    cmdlist << add_shader(y.view(), y_post).dispatch(n)
            << y.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = a * static_cast<float>(i) + y_init + y_post;
    }
    return check_span_near(host_out, expected, 1e-5f, "dsl_interop");
}

// ---------------------------------------------------------------------------
// Scenario 5: histogram with atomics into an integer output buffer
// (DSL-style typed API with a uint buffer argument)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_histogram(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    static constexpr uint32_t bin_count = 16u;

    auto shader = ext->create_cuda_kernel(cuda_device.compile(make_histogram_kernel()));
    if (!shader) {
        LUISA_WARNING("[histogram] shader import failed; skipping.");
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
                   .dispatch(n);
    cmdlist << bins.view().copy_to(luisa::span{host_bins});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<uint> expected(bin_count, 0u);
    for (auto i = 0u; i < n; ++i) {
        expected[static_cast<uint>(std::floor(host_data[i]))] += 1u;
    }
    return check_span_eq(host_bins, expected, "histogram");
}

// ---------------------------------------------------------------------------
// Scenario 6: re-import — import the same cuda shader twice into two
// CudaShader objects, dispatch both on the same stream into separate buffers,
// and verify both after the source Shader has been destroyed (the import is
// self-contained)
// ---------------------------------------------------------------------------

[[nodiscard]] bool run_reimport(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 512u;
    constexpr auto value = 42.5f;

    vk_cuda_interop::CudaShader shader_a;
    vk_cuda_interop::CudaShader shader_b;
    {
        auto cuda_shader = cuda_device.compile(make_fill_kernel());
        shader_a = ext->create_cuda_kernel(cuda_shader);
        shader_b = ext->create_cuda_kernel(cuda_shader);
        // The source DSL shader is destroyed here; the imported vk shaders
        // must remain usable.
    }
    if (!shader_a || !shader_b) {
        LUISA_WARNING("[reimport] shader import failed; skipping.");
        return false;
    }

    auto stream = device.create_stream();
    auto buffer_a = ext->create_buffer<float>(n);
    auto buffer_b = ext->create_buffer<float>(n);
    luisa::vector<float> host_a(n, 0.0f);
    luisa::vector<float> host_b(n, 0.0f);

    auto fill_a = shader_a.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::WRITE>,
                                  float, uint32_t>();
    auto fill_b = shader_b.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::WRITE>,
                                  float, uint32_t>();
    CommandList cmdlist;
    cmdlist << fill_a(buffer_a.view(), value, n).dispatch(n);
    cmdlist << fill_b(buffer_b.view(), value + 1.0f, n).dispatch(n);
    cmdlist << buffer_a.view().copy_to(luisa::span{host_a})
            << buffer_b.view().copy_to(luisa::span{host_b});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<float> expected_a(n, value);
    luisa::vector<float> expected_b(n, value + 1.0f);
    return check_span_near(host_a, expected_a, 0.0f, "reimport_a") &&
           check_span_near(host_b, expected_b, 0.0f, "reimport_b");
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
    if (ext->cuda_device_index() < 0) {
        LUISA_WARNING("No CUDA device matches this Vulkan device; skipping example.");
        return 0;
    }
    auto cuda_backend_installed = false;
    for (auto &&name : context.installed_backends()) {
        if (name == "cuda") { cuda_backend_installed = true; }
    }
    if (!cuda_backend_installed) {
        LUISA_WARNING("The cuda backend is not installed; skipping example.");
        return 0;
    }
    // Pair the cuda device with the same physical GPU (LUID-matched): the
    // compiled module image targets that GPU's compute capability.
    DeviceConfig cuda_config{.device_index = static_cast<size_t>(ext->cuda_device_index())};
    Device cuda_device = context.create_device("cuda", &cuda_config);

    // Non-short-circuit aggregation: run every scenario, report all failures.
    auto ok = true;
    ok = run_saxpy(device, ext, cuda_device) && ok;
    ok = run_block_reduce(device, ext, cuda_device) && ok;
    ok = run_chained_command_list(device, ext, cuda_device) && ok;
    ok = run_dsl_interop(device, ext, cuda_device) && ok;
    ok = run_histogram(device, ext, cuda_device) && ok;
    ok = run_reimport(device, ext, cuda_device) && ok;

    if (ok) {
        LUISA_INFO("All CUDA-in-VK-stream dispatches verified.");
    } else {
        LUISA_ERROR("Verification FAILED.");
    }
    return ok ? 0 : 1;
}
