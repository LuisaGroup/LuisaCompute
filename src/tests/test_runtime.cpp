//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>

#include <core/dynamic_module.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/buffer.h>
#include <runtime/stream.h>
#include <dsl/buffer_view.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

struct Base {
    float a;
};

struct Derived : Base {
    float b;
    constexpr Derived(float a, float b) noexcept : Base{a}, b{b} {}
};

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();
    
    Arena arena;
    Pool<Derived> pool{arena};
    {
        auto p = pool.create(1.0f, 2.0f);
        LUISA_INFO("Pool object: ({}, {}).", p->a, p->b);
    }
    
    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = std::make_unique<FakeDevice>();
#endif

    auto buffer = device->create_buffer<float>(16384u);
    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::iota(data.begin(), data.end(), 1.0f);
    
    auto kernel = LUISA_KERNEL(Var<float> a) noexcept {
    
    };
    kernel.prepare(*device);
    
    auto stream = device->create_stream();

    auto t0 = std::chrono::high_resolution_clock::now();
    stream
        << [] { LUISA_INFO("Hello!"); }
        << buffer.upload(data.data())
        << buffer.download(results.data())
        << [] { LUISA_INFO("Bye!"); }
        << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    LUISA_INFO("Finished in {} ms.", (t1 - t0) / 1ns * 1e-6);

    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);
}
