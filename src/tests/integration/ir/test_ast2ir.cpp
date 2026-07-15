// Test for AST (Abstract Syntax Tree) to IR (Intermediate Representation) conversion.
// This comprehensive test verifies the conversion of various DSL constructs including:
// - Callables with capture propagation
// - Kernel definitions with template parameters
// - Constants and shared memory
// - Matrix operations and vector math
// - Control flow (loops, conditionals, switches)
// - JSON and binary serialization of IR

#include "ut/ut.hpp"
#include "test_device.h"

#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>

#include <luisa/core/clock.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>

#include <luisa/ir/ast2ir.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure for struct type testing in kernels
struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a) {};

void test_ast2ir(Device &device) {
    constexpr auto f = 10;

    // Enable verbose logging for debugging
    luisa::log_level_verbose();

    // Create test buffers for kernel capture testing
    auto buffer = device.create_buffer<float4>(1024u);
    auto float_buffer = device.create_buffer<float>(1024u);

    // Define callable c1 that captures buffer and reads from it
    Callable c1 = [&](UInt a) noexcept {
        return buffer->read(a + thread_x());// captures buffer
    };

    // Define callable c2 that captures buffer (from c1) and float_buffer
    Callable c2 = [&](UInt b) noexcept {
        // captures buffer (propagated from c1) and float_buffer
        return c1(b) + make_float4(float_buffer->read(b));
    };

    // Define kernel k1 that captures buffer and float_buffer (from c2)
    Kernel1D k1 = [&] {
        // captures buffer and float_buffer (propagated from c2)
        auto v = c2(dispatch_x());
        float_buffer->write(dispatch_x(), v.x + v.y + v.z);
    };

    // Create constant vector data
    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    // Define callable that demonstrates tuple return with compose
    Callable add_mul = [&](Var<int> a, Var<int> b) noexcept {
        return compose(cast<int>(float_buffer->read(a + b)), a * b);
    };

    // Define callable with constant array access
    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    // Define template callable demonstrating generic programming
    Callable<int(int, int)> add = [&]<typename T>(Var<T> a, Var<T> b) noexcept {
        return cast<int>(c1(cast<uint>(a + b)).x);
    };

    // Start timing for kernel definition parsing
    Clock clock;
    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    // Define comprehensive kernel with various DSL constructs
    Kernel1D<Buffer<float>, uint> kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count) noexcept -> void {
        for (auto n = 0u; n < 1u; n++) {
            // Test shared memory allocation
            Shared<float4> shared_floats{16};

            count += 1u;

            // Test constant array access
            Constant float_consts = {1.0f, 2.0f};
            auto ff = float_consts.read(0);

            // Test matrix operations
            Var mat = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat3 = mat * mat2;
            Var mat4 = mat3 * make_float2(2.f);

            // Test variable creation and vector operations
            Var v_int = 10;
            Var t = make_int3(1, 2, 3);
            Var vv = ite(t == 10, 1, 2);

            // Test min operations with different types
            Var vvv = min(vv, 10);
            Var xxx = make_uint4(5);
            Var vvvv = min(xxx, 1u);

            // Test tuple unpacking from callable
            Var am = add_mul(v_int, v_int);
            Var a_copy = am.get<0>();
            Var m_copy = am.get<1>();

            // Test loop with break
            loop([] {
                if_(true, break_);
            });

            // Test dynamic range iteration
            for (auto v : dynamic_range(v_int)) {
                v_int += v;
            }

            // Test template callable invocation
            Var v_int_add_one = add(v_int, 1);
            Var vv_int = int_consts[v_int];
            Var v_float = buffer_float.read(count + thread_id().x);
            Var vv_float = float_consts[0];
            Var call_ret = callable(10, v_int, v_float);

            Var v_float_copy = v_float;

            // Test arithmetic operations
            Var z = -1 + v_int * v_float + 1.0f;
            z += 1;
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            // Test vector construction and operations
            Var<float2> w{cast<float>(v_int), v_float};
            w *= float2{1.2f};

            // Test if-else statement
            if_(v_int == v_int, [] {
                Var a = 0.0f;
            }).else_([] {
                Var c = 2.0f;
            });

            // Test switch statement
            switch_(123)
                .case_(1, [] {

                })
                .case_(2, [] {

                })
                .default_([] {

                });

            Var x = w.x;

            // Test struct construction and member access
            Var<int3> s;
            Var<Test> vvt{s, v_float_copy};
            Var<Test> vt{vvt};

            Var vt_copy = vt;
            Var c = 0.5f + vt.a * 1.0f;

            // Test buffer read/write operations
            Var vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
            Var another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
            buffer->write(v_int + 1, float4(123.0f));
        }
    };
    LUISA_INFO("Kernel definition parsed in {} ms.", clock.toc());

    // Convert AST to IR
    clock.tic();
    auto ir = AST2IR::build_kernel(kernel_def.function()->function());
    LUISA_INFO("AST2IR done in {} ms.", clock.toc());

    // Dump IR to JSON format
    {
        clock.tic();
        auto dump = ir::luisa_compute_ir_dump_json(&ir->get()->module);
        LUISA_INFO("IR json dump done in {} ms.", clock.toc());
        std::ofstream out{"test_ast2ir.json"};
        out << luisa::string_view{reinterpret_cast<const char *>(dump.ptr), dump.len};
    }

    // Dump IR to binary format
    {
        clock.tic();
        auto binary = ir::luisa_compute_ir_dump_binary(&ir->get()->module);
        LUISA_INFO("IR binary dump done in {} ms.", clock.toc());

        std::ofstream bin_out{"test_ast2ir.bin", std::ios::binary};
        bin_out.write(reinterpret_cast<const char *>(binary.ptr),
                      static_cast<std::streamsize>(binary.len));
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_ast2ir(device);
}
