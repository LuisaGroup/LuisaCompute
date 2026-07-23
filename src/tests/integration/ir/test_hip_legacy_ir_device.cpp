// Test for the HIP backend's legacy-IR DeviceInterface entry points.
// This test covers:
// - Creating a buffer from an ir::Type.
// - Compiling an ir::KernelModule through the backend overload.
// - Dispatching the resulting shader against the IR-typed buffer.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/ir/ast2ir.h>

#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_hip_legacy_ir_device(Device &device) {
    static constexpr auto element_count = 256u;
    log_level_verbose();

    auto ir_type = AST2IR::build_type(Type::of<uint>());
    auto buffer_info = device.impl()->create_buffer(
        &ir_type, element_count, nullptr);
    expect(buffer_info.valid()) << "legacy-IR buffer creation should succeed";
    if (!buffer_info.valid()) { return; }
    expect(buffer_info.element_stride == sizeof(uint));
    expect(buffer_info.total_size_bytes == element_count * sizeof(uint));

    auto buffer = BufferView<uint>{
        buffer_info.native_handle,
        buffer_info.handle,
        buffer_info.element_stride,
        0u,
        element_count,
        element_count};

    Kernel1D kernel = [](BufferUInt output) noexcept {
        set_block_size(64u);
        auto i = dispatch_id().x;
        output.write(i, i * 17u + 5u);
    };
    auto ir_kernel = AST2IR::build_kernel(kernel.function()->function());
    auto shader_info = device.impl()->create_shader(
        ShaderOption{.enable_cache = false}, ir_kernel->get());
    expect(shader_info.valid()) << "legacy-IR shader creation should succeed";
    if (!shader_info.valid()) {
        device.impl()->destroy_buffer(buffer_info.handle);
        return;
    }

    luisa::vector<uint> output(element_count);
    luisa::compute::detail::ShaderInvoke<1u> invocation{
        shader_info.handle, 1u, 0u};
    invocation << buffer;
    auto stream = device.create_stream();
    stream << std::move(invocation).dispatch(element_count)
           << buffer.copy_to(luisa::span{output})
           << synchronize();

    auto all_correct = true;
    for (auto i = 0u; i < element_count; i++) {
        auto expected = i * 17u + 5u;
        if (output[i] != expected) {
            LUISA_WARNING("Legacy-IR result mismatch at {}: got {}, expected {}.",
                          i, output[i], expected);
            all_correct = false;
        }
    }
    expect(all_correct) << "legacy-IR shader should produce the expected output";

    device.impl()->destroy_shader(shader_info.handle);
    device.impl()->destroy_buffer(buffer_info.handle);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "hip_legacy_ir_device_entry_points"_test = [&] {
        test_hip_legacy_ir_device(dc->device);
    };
}
