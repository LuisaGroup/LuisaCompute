//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>
#include <iostream>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/image.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    float a;
    float b;
    float array[16];
};

LUISA_STRUCT(Test, a, b, array) {};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    Callable load = [](BufferVar<float> buffer, Var<uint> index) noexcept {
        return buffer.read(index);
    };

    Callable store = [](BufferVar<float> buffer, Var<uint> index, Var<float> value) noexcept {
        buffer.write(index, value);
    };

    Callable add = [](Var<float> a, Var<float> b) noexcept {
        return a + b;
    };

    Kernel1D kernel_def = [&](BufferVar<float> source, BufferVar<float> result, Var<float> x) noexcept {
        set_block_size(256u);
        auto index = dispatch_id().x;
        store(result, index, add(load(source, index), x));
    };
    auto kernel = device.compile(kernel_def);

    static constexpr auto n = 1024u * 1024u;

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<float>(n);
    auto result_buffer = device.create_buffer<float>(n);

    std::vector<float> data(n);
    std::vector<float> results(n);
    std::iota(data.begin(), data.end(), 1.0f);

    Clock clock;
    stream << buffer.copy_from(data.data());
    auto command_buffer = stream.command_buffer();
    for (auto i = 0; i < 10; i++) {
        command_buffer << kernel(buffer, result_buffer, 3).dispatch(n);
    }
    command_buffer << commit();
    stream << result_buffer.copy_to(results.data());
    auto t1 = clock.toc();
    stream << synchronize();
    auto t2 = clock.toc();

    LUISA_INFO("Dispatched in {} ms. Finished in {} ms.", t1, t2);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[n - 2u], results[n - 1u]);
}
