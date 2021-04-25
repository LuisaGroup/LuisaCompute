//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>

#include <core/clock.h>
#include <core/dynamic_module.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;

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
    auto device = FakeDevice::create(context);
#endif

    auto buffer = device.create<Buffer<float>>(16384u);
    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::iota(data.begin(), data.end(), 1.0f);

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    auto add_mul = LUISA_CALLABLE(Var<int> a, Var<int> b) noexcept {
        return std::make_tuple(a + b, a * b);
    };

    auto callable = LUISA_CALLABLE(Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    auto kernel = LUISA_KERNEL1D(BufferVar<float> buffer_float, Var<uint> count) noexcept {
        Shared<float4> shared_floats{16};

        Var v_int = 10;

        auto [a, m] = add_mul(v_int, v_int);
        Var a_copy = a;
        Var m_copy = m;

        for (auto v : range(v_int)) {
            v_int += v;
        }

        Var vv_int = int_consts[v_int];
        vv_int = 0;
        Var v_float = buffer_float[count + thread_id().x];
        Var vv_float = float_consts[vv_int];
        Var call_ret = callable(10, v_int, v_float);

        Var v_float_copy = v_float;

        Var z = -1 + v_int * v_float + 1.0f;
        z += 1;
        static_assert(std::is_same_v<decltype(z), Var<float>>);
        for (uint i = 0; i < 3; ++i) {
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            Var<float2> w{v_int, v_float};
            w *= float2{1.2f};

            if_(1 + 1 == 2, [] {
                Var a = 0.0f;
            }).elif (1 + 2 == 3, [] {
                  Var b = 1.0f;
              }).else_([] {
                Var c = 2.0f;
            });

            switch_(123)
                .case_(1, [] {

                })
                .case_(2, [] {

                })
                .default_([] {

                });

            Var x = w.x;
        }

        Var vec4 = buffer[10];           // indexing into captured buffer (with literal)
        Var another_vec4 = buffer[v_int];// indexing into captured buffer (with Var)
        buffer[v_int + 1] = 123.0f;
    };
    device.compile(kernel);

    auto stream = device.create<Stream>();

    Clock clock;
    stream << buffer.copy_from(data.data())
           << buffer.copy_to(results.data());
    stream.synchronize();   
    LUISA_INFO("Finished in {} ms.", clock.toc());

    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);
}
