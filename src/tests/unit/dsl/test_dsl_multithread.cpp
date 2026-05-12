// Test for multi-threaded DSL kernel compilation
// This test verifies that the DSL can safely compile kernels
// from multiple threads concurrently without race conditions.
//
// Features tested:
// - Thread-safe kernel compilation
// - Callable usage across threads
// - Buffer and constant access from multiple threads
// - DSL syntax operations in multi-threaded context

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
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure for DSL struct handling
struct Test {
    int3 something;
    float a;
};

// Register the structure with the DSL
LUISA_STRUCT(Test, something, a) {};

int test_dsl_multithread(Device &device) {
    // Create buffers for kernel operations
    Buffer<float4> buffer = device.create_buffer<float4>(1024u);
    Buffer<float> float_buffer = device.create_buffer<float>(1024u);

    // Create constant vector
    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    // Callable function that performs arithmetic operations
    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    // Create worker threads for concurrent kernel compilation
    std::vector<std::thread> threads;
    threads.reserve(8u);

    for (size_t i = 0u; i < 8u; i++) {
        threads.emplace_back([&, worker = i] {
            Clock clock;

            // Define constants for kernel
            Constant float_consts = {1.0f, 2.0f};
            Constant int_consts = const_vector;

            // Define kernel with various DSL operations
            Kernel1D kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count) noexcept {
                // Shared memory allocation
                Shared<float4> shared_floats{16};

                // Variable declarations and operations
                Var v_int = 10;
                Var vv_int = int_consts[v_int];
                Var v_float = buffer_float.read(count + thread_id().x);
                Var vv_float = float_consts[vv_int];
                Var call_ret = callable(10, v_int, v_float);

                Var v_float_copy = v_float;

                // Arithmetic operations
                Var z = -1 + v_int * v_float + 1.0f;
                z += 1;
                static_assert(std::is_same_v<decltype(z), Var<float>>);

                // Loop with various DSL constructs
                for (size_t i = 0u; i < 3u; i++) {
                    Var v_vec = float3{1.0f};
                    Var v2 = float3{2.0f} - v_vec * 2.0f;
                    v2 *= 5.0f + v_float;

                    Var<float2> w{cast<float>(v_int), v_float};
                    w *= float2{1.2f};

                    // Conditional statements
                    if_(1 + 1 == 2, [] {
                        Var a = 0.0f;
                    }).else_([] {
                        Var c = 2.0f;
                    });

                    // Loop with break
                    loop([&] {
                        z += 1;
                        if_(true, break_);
                    });

                    // Switch statement
                    switch_(123)
                        .case_(1, [] {

                        })
                        .case_(2, [] {

                        })
                        .default_([] {

                        });

                    Var x = w.x;
                }

                // Struct variable usage
                Var<int3> s;
                Var<Test> vvt{s, v_float_copy};
                Var<Test> vt{vvt};

                Var vt_copy = vt;
                Var c = 0.5f + vt.a * 1.0f;

                // Buffer access operations
                Var vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
                Var another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
            };
            double t1 = clock.toc();

            // Compile and dispatch kernel
            auto kernel = device.compile(kernel_def);
            luisa::unique_ptr<Command> command = kernel(float_buffer, 12u).dispatch(1024u);

            clock.tic();
            auto shader = device.compile<1>(kernel_def);
            double t2 = clock.toc();
            LUISA_INFO("Thread: {}, AST: {:.3f} ms, Codegen & Compile: {:.3f} ms",
                       worker, t1, t2);
        });
    }

    // Wait for all threads to complete
    for (std::thread &t : threads) { t.join(); }

    expect(true) << "multithreaded compilation completed";
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_dsl_multithread(device);
}
