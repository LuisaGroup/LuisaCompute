//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>

#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/stream.h>
#include <dsl/buffer_view.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

struct Test {
    float a;
    float b;
    float array[16];
};

LUISA_STRUCT(Test, a, b, array);

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = std::make_unique<FakeDevice>(context);
#endif

    auto load = LUISA_CALLABLE(BufferView<float> buffer, Var<uint> index) noexcept {
        return buffer[index];
    };

    auto store = LUISA_CALLABLE(BufferView<float> buffer, Var<uint> index, Var<float> value) noexcept {
        buffer[index] = value;
    };

    auto add = LUISA_CALLABLE(Var<float> a, Var<float> b) noexcept {
        return a + b;
    };

    auto kernel = LUISA_KERNEL1D(BufferView<float> source, BufferView<float> result, Var<Test> x) noexcept {
        set_block_size(256u);
        auto index = dispatch_id().x;
        store(result, index, add(load(source, index), x.a));
    };
    device->prepare(kernel);

    static constexpr auto n = 1024u * 1024u;

    auto stream = device->create_stream();
    auto buffer = device->create_buffer<float>(n);
    auto result_buffer = device->create_buffer<float>(n);

    std::vector<float> data(n);
    std::vector<float> results(n);
    std::iota(data.begin(), data.end(), 1.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    stream << buffer.copy_from(data.data());
    {
        auto s = stream << kernel(buffer, result_buffer, Test{1.0f, 0.0f}).launch(n);
        for (auto i = 0; i < 10; i++) {
            s << kernel(buffer, result_buffer, Test{2.0f + i, 0.0f}).launch(n);
        }
    }
    stream << result_buffer.copy_to(results.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    stream << synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    LUISA_INFO("Dispatched in {} ms. Finished in {} ms.",
               (t1 - t0) / 1ns * 1e-6, (t2 - t0) / 1ns * 1e-6);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[n - 2u], results[n - 1u]);
}
