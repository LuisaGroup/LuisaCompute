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
#include <runtime/heap.h>
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

    Buffer<float> buffer;

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    auto buffer2 = device.create_buffer<float>(16384u);
    buffer = std::move(buffer2);
    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::iota(data.begin(), data.end(), 1.0f);

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    Callable add_mul = [](Var<int> a, Var<int> b) noexcept {
        return std::make_tuple(a + b, a * b);
    };

    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    Callable add = [](Float a, Float b) noexcept { return a + b; };
    Callable sub = [](Float a, Float b) noexcept { return a - b; };
    Callable mul = [](Float a, Float b) noexcept { return a * b; };
    std::vector ftab{add, sub, mul};
    Kernel1D kernel = [&](BufferVar<float> buffer_float, Var<uint> count, HeapVar heap) noexcept {
        Var tag = 114514;
        match({123, 6666, 114514}, tag, [&](auto i) noexcept {
            Var result = ftab[i](float_consts[0], float_consts[1]);
        });

        Var v_int = 10;
        Shared<float4> shared_floats{16};
        Var color = heap.tex2d(v_int).sample(float2(0.0f));

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

            Var<float2> w{v_int.cast<float>(), v_float};
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
    auto compiled_kernel = device.compile(kernel);
    auto stream = device.create_stream();

    Clock clock;
    stream << buffer.copy_from(data.data())
           << buffer.copy_to(results.data())
           << synchronize();
    LUISA_INFO("Finished in {} ms.", clock.toc());

    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);

    auto heap = device.create_heap(1_gb);
    for (auto i = 0u; i < 10u; i++) {
        static_cast<void>(heap.create_buffer<float>(i, 1024u));
        static_cast<void>(heap.create_texture(i, PixelStorage::FLOAT4, uint2(1024u)));
        LUISA_INFO("Used size: {}", heap.allocated_size());
    }
    for (auto i = 0u; i < 10u; i++) {
        static_cast<void>(heap.destroy_buffer(i));
        LUISA_INFO("Used size: {}", heap.allocated_size());
    }
}
