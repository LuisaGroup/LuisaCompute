#include <iostream>
#include <chrono>
#include <numeric>
#include <thread>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/context.h>
using namespace luisa;
using namespace luisa::compute;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a){};

int main(int argc, char *argv[]) {

    Context ctx{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    Buffer<float4> buffer = device.create_buffer<float4>(1024u);
    Buffer<float> float_buffer = device.create_buffer<float>(1024u);

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    std::vector<std::thread> threads;
    threads.reserve(8u);

    for (size_t i = 0u; i < 8u; i++) {
        threads.emplace_back([&, worker = i] {
            Clock clock;
            Constant float_consts = {1.0f, 2.0f};
            Constant int_consts = const_vector;
            Kernel1D kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count) noexcept {
                Shared<float4> shared_floats{16};

                Var v_int = 10;
                Var vv_int = int_consts[v_int];
                Var v_float = buffer_float.read(count + thread_id().x);
                Var vv_float = float_consts[vv_int];
                Var call_ret = callable(10, v_int, v_float);

                Var v_float_copy = v_float;

                Var z = -1 + v_int * v_float + 1.0f;
                z += 1;
                static_assert(std::is_same_v<decltype(z), Var<float>>);

                for (size_t i = 0u; i < 3u; i++) {
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

                    loop([&] {
                        z += 1;
                        if_(true, break_);
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

                Var<int3> s;
                Var<Test> vvt{s, v_float_copy};
                Var<Test> vt{vvt};

                Var vt_copy = vt;
                Var c = 0.5f + vt.a * 1.0f;

                Var vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
                Var another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
            };
            double t1 = clock.toc();

            Shader1D<Buffer<float>, uint> kernel = device.compile(kernel_def);
            luisa::unique_ptr<Command> command = kernel(float_buffer, 12u).dispatch(1024u);

            clock.tic();
            Shader1D<Buffer<float>, uint> shader = device.compile<1>(kernel_def);
            double t2 = clock.toc();
            LUISA_INFO("Thread: {}, AST: {:.3f} ms, Codegen & Compile: {:.3f} ms",
                       worker, t1, t2);
        });
    }

    for (std::thread &t : threads) { t.join(); }
}

