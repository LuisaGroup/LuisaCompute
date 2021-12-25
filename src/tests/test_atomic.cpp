//
// Created by Mike Smith on 2021/6/23.
//

#include <iostream>
#include <variant>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    luisa::variant<luisa::monostate> a;

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    auto buffer = device.create_buffer<uint>(1u);
    Kernel1D count_kernel = [](BufferUInt buffer) noexcept {
        Constant<uint> constant{1u};
        Var x = buffer.atomic(0).fetch_add(constant[0]);
    };
    auto count = device.compile(count_kernel);

    auto host_buffer = 0u;
    auto stream = device.create_stream();

    Clock clock;
    clock.tic();
    stream << buffer.copy_from(&host_buffer)
           << count(buffer).dispatch(102400u)
           << buffer.copy_to(&host_buffer)
           << synchronize();
    auto time = clock.toc();

    LUISA_INFO("Count: {}, Time: {} ms", host_buffer, time);
}
