#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/dispatch_buffer.h>
#include <dsl/syntax.h>
#include <dsl/dispatch_indirect.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    Kernel1D clear_kernel = [](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.clear();
    };
    constexpr auto kernel_block_size = make_uint3(64, 1, 1);
    constexpr auto dispatch_count = 16u;
    Kernel1D emplace_kernel = [&](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.dispatch_kernel(kernel_block_size, make_uint3(dispatch_id().x, 1u, 1u), dispatch_id().x);
    };
    Kernel1D dispatch_kernel = [&](BufferVar<uint> buffer) {
        set_block_size(kernel_block_size.x, kernel_block_size.y, kernel_block_size.z);
        buffer.atomic(kernel_id()).fetch_add(dispatch_size().x);
    };
    Shader1D<IndirectDispatchBuffer> clear_shader = device.compile(clear_kernel);
    Shader1D<IndirectDispatchBuffer> emplace_shader = device.compile(emplace_kernel);
    Shader1D<Buffer<uint>> dispatch_shader = device.compile(dispatch_kernel);

    IndirectDispatchBuffer dispatch_buffer = device.create_indirect_dispatch_buffer(dispatch_count);
    Buffer<uint> buffer = device.create_buffer<uint>(dispatch_count);
    std::array<uint, dispatch_count> buffer_data{};
    stream << buffer.copy_from(buffer_data.data())
           << clear_shader(dispatch_buffer).dispatch(1)
           << emplace_shader(dispatch_buffer).dispatch(dispatch_count)
           << dispatch_shader(buffer).dispatch(dispatch_buffer)
           << buffer.copy_to(buffer_data.data())
           << synchronize();
    luisa::string result;
    for (auto& i : buffer_data){
        result += std::to_string(i) + " ";
    }
    LUISA_INFO("Result should be: 0 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225");
    LUISA_INFO("Result: {}", result);
}
