// Test for VK_NV_cuda_kernel_launch: dispatching CUDA kernels inside a Vulkan
// command buffer through the VkCudaInterop extension.
// - cuda_source_vector_op: NVRTC-compiled CUDA C++ source kernel (axpy) on
//   interop buffers with upload/verify
// - ptx_roundtrip: compile with ptx_output, rebuild a shader from the PTX and
//   run it
// - multi_launch_one_command_list: two chained CUDA launches in one CommandList
// - interop_with_dsl_kernel: DSL Kernel1D dispatches on the same stream before
//   and after a CUDA launch (barrier correctness)
//
// Skips gracefully unless run as: test_vk_cuda_kernel_launch vk

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

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t k_block_size = 256u;

[[nodiscard]] constexpr uint3 grid_for(uint32_t n) noexcept {
    return uint3{(n + k_block_size - 1u) / k_block_size, 1u, 1u};
}

constexpr luisa::string_view axpy_source = R"(
extern "C" __global__ void axpy(const float *x, float *y, float a, unsigned int n) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
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

void expect_span_near(luisa::span<const float> result,
                      luisa::span<const float> expected) noexcept {
    expect(result.size() == expected.size());
    for (auto i = 0u; i < result.size(); ++i) {
        expect(std::abs(result[i] - expected[i]) < 1e-5f)
            << "mismatch at index " << i << ": " << result[i]
            << " vs " << expected[i];
    }
}

void test_cuda_source_vector_op(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;

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

    auto shader = ext->create_cuda_kernel_shader(
        {.source = axpy_source, .kernel_name = "axpy"});
    expect(shader != 0u) << "create_cuda_kernel_shader should succeed";
    if (shader == 0u) { return; }

    vk_cuda_interop::KernelLauncher launcher;
    launcher.add_buffer(x.view(), Usage::READ)
        .add_buffer(y.view(), Usage::READ_WRITE)
        .add_uniform(a)
        .add_uniform(n);
    stream << std::move(launcher).build(
                  shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u)
           << y.view().copy_to(luisa::span{host_out})
           << synchronize();
    ext->destroy_cuda_kernel_shader(shader);

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = a * host_x[i] + host_y[i];
    }
    expect_span_near(host_out, expected);
    LUISA_INFO("cuda_source_vector_op passed.");
}

void test_ptx_roundtrip(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 512u;
    constexpr auto value = 42.5f;

    luisa::vector<char> ptx;
    auto shader_from_source = ext->create_cuda_kernel_shader(
        {.source = fill_source, .kernel_name = "fill", .ptx_output = &ptx});
    expect(shader_from_source != 0u);
    expect(!ptx.empty()) << "ptx_output should receive the compiled PTX";
    if (shader_from_source == 0u || ptx.empty()) { return; }

    // Rebuild the shader from the PTX text.
    auto shader_from_ptx = ext->create_cuda_kernel_shader(
        {.source = luisa::string_view{ptx.data(), ptx.size()},
         .kernel_name = "fill",
         .source_is_ptx = true});
    expect(shader_from_ptx != 0u) << "PTX roundtrip shader creation should succeed";
    if (shader_from_ptx == 0u) {
        ext->destroy_cuda_kernel_shader(shader_from_source);
        return;
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

    for (auto i = 0u; i < n; ++i) {
        expect(host_out[i] == value) << "mismatch at index " << i;
    }
    LUISA_INFO("ptx_roundtrip passed.");
}

void test_multi_launch_one_command_list(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto b = 3.0f;

    auto shader = ext->create_cuda_kernel_shader(
        {.source = axpy_source, .kernel_name = "axpy"});
    expect(shader != 0u);
    if (shader == 0u) { return; }

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

    CommandList cmdlist;
    cmdlist << x.view().copy_from(luisa::span{host_x})
            << y.view().copy_from(luisa::span{host_y})
            << z.view().copy_from(luisa::span{host_z});
    cmdlist << std::move(first_launcher).build(
        shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << std::move(second_launcher).build(
        shader, grid_for(n), uint3{k_block_size, 1u, 1u}, 0u);
    cmdlist << z.view().copy_to(luisa::span{host_out});
    stream << cmdlist.commit() << synchronize();
    ext->destroy_cuda_kernel_shader(shader);

    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; ++i) {
        expected[i] = b * (a * host_x[i] + host_y[i]) + host_z[i];
    }
    expect_span_near(host_out, expected);
    LUISA_INFO("multi_launch_one_command_list passed.");
}

void test_interop_with_dsl_kernel(Device &device, VkCudaInterop *ext) {
    static constexpr uint32_t n = 1024u;
    constexpr auto a = 2.0f;
    constexpr auto y_init = 1.0f;
    constexpr auto y_post = 10.0f;

    auto shader = ext->create_cuda_kernel_shader(
        {.source = axpy_source, .kernel_name = "axpy"});
    expect(shader != 0u);
    if (shader == 0u) { return; }

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
    expect_span_near(host_out, expected);
    LUISA_INFO("interop_with_dsl_kernel passed.");
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1 || argv[1] == nullptr || luisa::string_view{argv[1]} != "vk") {
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
    log_level_verbose();

    test_cuda_source_vector_op(device, ext);
    test_ptx_roundtrip(device, ext);
    test_multi_launch_one_command_list(device, ext);
    test_interop_with_dsl_kernel(device, ext);
    return 0;
}
