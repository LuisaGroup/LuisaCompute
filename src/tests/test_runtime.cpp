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
#include <runtime/bindless_array.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>

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

    Context context{argv[0]};

    Buffer<float> buffer;
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    auto buffer2 = device.create_buffer<float>(16384u);
    buffer = std::move(buffer2);
    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::vector<float> volume_data(64u * 64u * 64u * 4u);
    std::vector<float> volume_data_download(64u * 64u * 64u * 4u);
    std::iota(data.begin(), data.end(), 1.0f);
    std::iota(volume_data.begin(), volume_data.end(), 1.0f);
    auto volume_buffer = device.create_buffer<float>(64u * 64u * 64u * 4u);
    auto volume = device.create_volume<float>(PixelStorage::FLOAT4, make_uint3(64u), 1u);

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    Callable add_mul = [](Var<int> a, Var<int> b) noexcept {
        return compose(a + b, a * b);
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
    std::vector func_table{add, sub, mul};
    Callable dynamic_dispatch = [&](UInt tag, Float a, Float b) noexcept {
        Float result;
        $switch(tag) {
            for (auto i = 0u; i < func_table.size(); i++) {
                $case(i) { result = func_table[i](a, b); };
            }
        };
        return result;
    };

    Kernel1D kernel = [&](BufferVar<float> buffer_float, Var<uint> count, BindlessVar heap) noexcept {
        Var tag = 114514;
        match({123, 6666, 114514}, tag, [&](auto i) noexcept {
            Var result = func_table[i](float_consts[0], float_consts[1]);
        });

        Var v_int = 10;
        Shared<float4> shared_floats{16};
        Var color = heap.tex2d(v_int).sample(float2(0.0f));

        auto am = add_mul(v_int, v_int);
        Var a_copy = am.get<0>();
        Var m_copy = am.get<1>();

        for (auto v : range(v_int)) {
            v_int += v;
        }

        Var vv_int = int_consts[v_int];
        vv_int = 0;
        Var v_float = buffer_float.read(count + thread_id().x);
        Var vv_float = float_consts[vv_int];
        Var call_ret = callable(10, v_int, v_float);

        Var v_float_copy = v_float;

        Var z = -1 + v_int * v_float + 1.0f;
        z += 1;
        for (uint i = 0; i < 3; ++i) {
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            Var<float2> w{cast<float>(v_int), v_float};
            w *= float2{1.2f};

            if_(1 + 1 == 2, [] {
                Var a = 0.0f;
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

        Var vec4 = buffer.read(10);           // indexing into captured buffer (with literal)
        Var another_vec4 = buffer.read(v_int);// indexing into captured buffer (with Var)
        buffer.write(v_int + 1, 123.0f);
    };
    auto compiled_kernel = device.compile(kernel);
    auto stream = device.create_stream();

    Clock clock;
    stream << buffer.copy_from(data.data())
           << buffer.copy_to(results.data())
           << volume.view(0).copy_from(volume_data.data())
           << volume.view(0).copy_to(volume_data_download.data())
           << synchronize();
    LUISA_INFO("Finished in {} ms.", clock.toc());

    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);

    for (auto i = 0u; i < volume_data.size(); i++) {
        if (volume_data[i] != volume_data_download[i]) {
            LUISA_ERROR_WITH_LOCATION(
                "Bad: i = {}, origin = {}, download = {}.",
                i, volume_data[i], volume_data_download[i]);
        }
    }
}
