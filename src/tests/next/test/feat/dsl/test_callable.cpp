//
// Created by Mike Smith on 2021/2/27.
//

#include "common/config.h"

#include <numeric>
#include <iostream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {
struct CallableTest {
    float a;
    float b;
    float array[16];
};
}// namespace luisa::test
LUISA_STRUCT(luisa::test::CallableTest, a, b, array) {};

namespace luisa::test {

int test_callable(Device &device) {
    log_level_verbose();

    static constexpr uint n = 10u;

    Buffer<float> buffer = device.create_buffer<float>(n);

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
        UInt index = dispatch_id().x;
        store(result, index, add(load(source, index), x));
    };
    Shader1D<Buffer<float>, Buffer<float>, float> kernel = device.compile(kernel_def);
    Stream stream = device.create_stream();
    Buffer<float> result_buffer = device.create_buffer<float>(n);

    std::vector<float> data(n);
    std::vector<float> results(n);
    std::iota(data.begin(), data.end(), 1.0f);

    Clock clock;
    stream << buffer.copy_from(data.data());
    CommandList command_list = CommandList::create();
    command_list << kernel(buffer, result_buffer, 3.0f).dispatch(n);

    stream << command_list.commit()
           << result_buffer.copy_to(results.data());

    double t1 = clock.toc();
    stream << synchronize();
    double t2 = clock.toc();

    LUISA_INFO("Dispatch in {} ms. Finished in {} ms", t1, t2);
    LUISA_INFO("Data: {}, {}, {}, {}, ..., {}, {}.",
               data[0], data[1], data[2], data[3],
               data[n - 2u], data[n - 1u]);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[n - 2u], results[n - 1u]);

    for (unsigned int i = 0u; i < n; i++) {
        // LUISA_INFO("Result: {} || {}", i, results[i] - data[i]);
        CHECK(results[i] - data[i] == doctest::Approx(3.0f).epsilon(0.001f));
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("dsl") {
    LUISA_TEST_CASE_WITH_DEVICE("callable", luisa::test::test_callable(device) == 0);
}