// Test for VK_NV_cuda_kernel_launch: dispatching DSL kernels compiled by the
// cuda backend inside a Vulkan command buffer through the VkCudaInterop
// extension.
// - dsl_kernel_vector_op: imported DSL axpy kernel (typed API) on interop
//   buffers with upload/verify
// - reimport_twice: import the same cuda shader twice and dispatch both after
//   the source shader's destruction (import self-containedness)
// - multi_launch_one_command_list: two chained CUDA launches in one CommandList
// - interop_with_dsl_kernel: DSL Kernel1D dispatches on the same stream before
//   and after a CUDA launch (barrier correctness)
// - reject_printing_kernel: shaders using device printing must not import
//
// Skips gracefully unless run as: test_vk_cuda_kernel_launch vk
// Also skips when the cuda backend is not installed or no CUDA device matches
// the Vulkan device.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/vk_cuda_interop.h>

#include <cmath>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t k_block_size = 256u;

[[nodiscard]] auto make_axpy_kernel() noexcept {
    return Kernel1D{[](BufferFloat x, BufferFloat y, Float a, UInt n) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        if_(i < n, [&] {
            y->write(i, a * x->read(i) + y->read(i));
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

void expect_span_near(luisa::span<const float> result,
                      luisa::span<const float> expected) noexcept {
    expect(result.size() == expected.size());
    for (auto i = 0u; i < result.size(); ++i) {
        expect(std::abs(result[i] - expected[i]) < 1e-5f)
            << "mismatch at index " << i << ": " << result[i]
            << " vs " << expected[i];
    }
}

void test_dsl_kernel_vector_op(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;

    auto shader = ext->create_cuda_kernel(cuda_device.compile(make_axpy_kernel()));
    expect(static_cast<bool>(shader)) << "create_cuda_kernel should succeed";
    if (!shader) { return; }

    auto stream = device.create_stream();
    auto x = ext->create_buffer<float>(n);
    auto y = ext->create_buffer<float>(n);
    luisa::vector<float> host_x(n);
    luisa::vector<float> host_y(n, 1.0f);
    luisa::vector<float> host_out(n, -1.0f);
    for (auto i = 0u; i < n; ++i) {
        host_x[i] = static_cast<float>(i);
    }
    stream << x.view().copy_from(luisa::span{host_x})
           << y.view().copy_from(luisa::span{host_y})
           << synchronize();

    // Typed DSL-style invocation with the imported shader's compiled block
    // size as the default block dimension.
    auto axpy = shader.kernel<vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
                              vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ_WRITE>,
                              float, uint32_t>();
    stream << axpy(x.view(), y.view(), a, n)
                  .dispatch(n)
           << y.view().copy_to(luisa::span{host_out})
           << synchronize();

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = a * host_x[i] + host_y[i];
    }
    expect_span_near(host_out, expected);
    LUISA_INFO("dsl_kernel_vector_op passed.");
}

void test_reimport_twice(Device &device, VkCudaInterop *ext, Device &cuda_device) {
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
    expect(static_cast<bool>(shader_a)) << "first import should succeed";
    expect(static_cast<bool>(shader_b)) << "second import should succeed";
    if (!shader_a || !shader_b) { return; }

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

    for (auto i = 0u; i < n; ++i) {
        expect(host_a[i] == value) << "mismatch at index " << i;
        expect(host_b[i] == value + 1.0f) << "mismatch at index " << i;
    }
    LUISA_INFO("reimport_twice passed.");
}

void test_multi_launch_one_command_list(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto b = 3.0f;

    auto shader = ext->create_cuda_kernel(cuda_device.compile(make_axpy_kernel()));
    expect(static_cast<bool>(shader));
    if (!shader) { return; }

    auto stream = device.create_stream();
    auto x = ext->create_buffer<float>(n);
    auto y = ext->create_buffer<float>(n);
    auto z = ext->create_buffer<float>(n);
    luisa::vector<float> host_x(n);
    luisa::vector<float> host_y(n, 1.0f);
    luisa::vector<float> host_z(n, 0.5f);
    luisa::vector<float> host_out(n, 0.0f);
    for (auto i = 0u; i < n; ++i) {
        host_x[i] = static_cast<float>(i);
    }

    // First launch: y = a * x + y. Second launch (chained in the same command
    // list) consumes the first launch's output: z = b * y + z.
    vk_cuda_interop::KernelLauncher first_launcher;
    first_launcher.add_buffer(x.view(), Usage::READ)
        .add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(a)
        .add_uniform(n);
    vk_cuda_interop::KernelLauncher second_launcher;
    second_launcher.add_buffer(y.view(), Usage::READ)
        .add_buffer(z.view(), Usage::READ_WRITE)
        .add_uniform(b)
        .add_uniform(n);

    auto block = uint3{k_block_size, 1u, 1u};
    CommandList cmdlist;
    cmdlist << x.view().copy_from(luisa::span{host_x})
            << y.view().copy_from(luisa::span{host_y})
            << z.view().copy_from(luisa::span{host_z});
    cmdlist << std::move(first_launcher).build(
        shader.handle(), uint3{n, 1u, 1u}, block, 0u);
    cmdlist << std::move(second_launcher).build(
        shader.handle(), uint3{n, 1u, 1u}, block, 0u);
    cmdlist << z.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = b * (a * host_x[i] + host_y[i]) + host_z[i];
    }
    expect_span_near(host_out, expected);
    LUISA_INFO("multi_launch_one_command_list passed.");
}

void test_interop_with_dsl_kernel(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto y_init = 1.0f;
    constexpr auto y_post = 10.0f;

    auto shader = ext->create_cuda_kernel(cuda_device.compile(make_axpy_kernel()));
    expect(static_cast<bool>(shader));
    if (!shader) { return; }

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

    vk_cuda_interop::KernelLauncher launcher;
    launcher.add_buffer(x.view(), Usage::READ)
        .add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(a)
        .add_uniform(n);

    CommandList cmdlist;
    cmdlist << index_shader(x.view()).dispatch(n)
            << fill_shader(y.view(), y_init).dispatch(n);
    cmdlist << std::move(launcher).build(
        shader.handle(), uint3{n, 1u, 1u},
        uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << add_shader(y.view(), y_post).dispatch(n)
            << y.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = a * static_cast<float>(i) + y_init + y_post;
    }
    expect_span_near(host_out, expected);
    LUISA_INFO("interop_with_dsl_kernel passed.");
}

void test_reject_printing_kernel(Device &device, VkCudaInterop *ext, Device &cuda_device) {
    static_cast<void>(device);
    // Kernels using device printing gain an LCPrintBuffer Params member that
    // the vk side cannot encode; import must fail closed.
    Kernel1D print_kernel = [](BufferFloat buffer) {
        set_block_size(k_block_size);
        auto i = dispatch_x();
        device_log("value: {}", buffer->read(i));
    };
    auto cuda_shader = cuda_device.compile(print_kernel);
    expect(static_cast<bool>(cuda_shader)) << "printing kernel should compile on the cuda backend";
    if (!cuda_shader) { return; }
    auto imported = ext->create_cuda_kernel(cuda_shader);
    expect(!static_cast<bool>(imported))
        << "kernels using device printing must not be importable";
    LUISA_INFO("reject_printing_kernel passed.");
}

}// namespace

int main(int argc, char *argv[]) {
    // Note: keep the backend check on raw argv (strcmp) instead of
    // luisa::string_view — the string_view construction in the || chain is
    // miscompiled in this unity-build TU (speculative strlen on argv[1]).
    if (argc <= 1 || argv[1] == nullptr || std::strcmp(argv[1], "vk") != 0) {
        LUISA_INFO("test_vk_cuda_kernel_launch requires the vk backend; skipping.");
        return 0;
    }
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    auto ext = device.extension<VkCudaInterop>();
    if (ext == nullptr) {
        LUISA_WARNING("VkCudaInterop extension is not available; skipping test.");
        return 0;
    }
    if (!ext->cuda_kernel_launch_supported()) {
        LUISA_WARNING("VK_NV_cuda_kernel_launch is not supported on this device; skipping test.");
        return 0;
    }
    if (ext->cuda_device_index() < 0) {
        LUISA_WARNING("No CUDA device matches this Vulkan device; skipping test.");
        return 0;
    }
    auto cuda_backend_installed = false;
    for (auto &&name : dc->context.installed_backends()) {
        if (name == "cuda") { cuda_backend_installed = true; }
    }
    if (!cuda_backend_installed) {
        LUISA_WARNING("The cuda backend is not installed; skipping test.");
        return 0;
    }
    // Pair the cuda device with the same physical GPU (LUID-matched): the
    // compiled module image targets that GPU's compute capability.
    DeviceConfig cuda_config{.device_index = static_cast<size_t>(ext->cuda_device_index())};
    Device cuda_device = dc->context.create_device("cuda", &cuda_config);
    log_level_verbose();

    test_dsl_kernel_vector_op(device, ext, cuda_device);
    test_reimport_twice(device, ext, cuda_device);
    test_multi_launch_one_command_list(device, ext, cuda_device);
    test_interop_with_dsl_kernel(device, ext, cuda_device);
    test_reject_printing_kernel(device, ext, cuda_device);
    return 0;
}
