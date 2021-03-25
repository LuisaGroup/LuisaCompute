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

int main(int argc, char *argv[]) {

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = std::make_unique<FakeDevice>();
#endif

    auto add = LUISA_CALLABLE(Var<float> a, Var<float> b) noexcept {
        return a + b;
    };

    auto kernel = LUISA_KERNEL(BufferView<float> source, BufferView<float> result, Var<float> x) noexcept {
        auto index = dispatch_id().x;
        result[index] = add(source[index], x);
    };
    device->prepare(kernel);

    auto stream = device->create_stream();
    auto buffer = device->create_buffer<float>(16384u);
    auto result_buffer = device->create_buffer<float>(16384u);

    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::iota(data.begin(), data.end(), 1.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    stream << buffer.upload(data.data())
           << kernel(buffer, result_buffer, 2.0f).parallelize(16384u)
           << result_buffer.download(results.data())
           << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    LUISA_INFO("Finished in {} ms.", (t1 - t0) / 1ns * 1e-6);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);
}
