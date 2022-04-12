//
// Created by Mike Smith on 2021/6/23.
//

#include <iostream>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    auto device = context.create_device("ispc");

    auto buffer = device.create_buffer<uint>(4u);
    Kernel1D count_kernel = [&]() noexcept {
        Constant<uint> constant{1u};
        Var x = buffer.atomic(3u).fetch_add(constant[0]);
        if_(x == 0u, [&] {
            buffer.write(0u, 1u);
        });
    };
    auto count = device.compile(count_kernel);

    auto host_buffer = make_uint4(0u);
    auto stream = device.create_stream();

    Clock clock;
    clock.tic();
    stream << buffer.copy_from(&host_buffer)
           << count().dispatch(102400u)
           << buffer.copy_to(&host_buffer)
           << synchronize();
    auto time = clock.toc();

    LUISA_INFO("Count: {} {}, Time: {} ms", host_buffer.x, host_buffer.w, time);

    auto atomic_float_buffer = device.create_buffer<uint>(1u);
    Callable atomic_float_add = [](BufferUInt buffer) noexcept {
        auto gid = dispatch_x();
        auto x = 0.1f * cast<float>(gid);
        $loop {
            auto old_value = buffer.read(0u);
            auto new_value = as<uint>(x + as<float>(old_value));
            $if (buffer.atomic(0u).compare_exchange(old_value, new_value) == old_value) {
                $break;
            };
        };
    };
    Kernel1D add_kernel = [&](BufferUInt buffer) noexcept {
        atomic_float_add(buffer);
    };
    auto add_shader = device.compile(add_kernel);

    auto result = 0.f;
    stream << atomic_float_buffer.copy_from(&result)
           << add_shader(atomic_float_buffer).dispatch(1024u)
           << atomic_float_buffer.copy_to(&result)
           << synchronize();
    LUISA_INFO("Result: {}.", result);
}
