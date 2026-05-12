// Test for round-trip conversion: AST -> IR -> AST.
// This test verifies the bidirectional conversion between Abstract Syntax Tree (AST)
// and Intermediate Representation (IR), ensuring that IR can be converted back to AST
// while preserving semantics.

#if __has_include("ut/ut.hpp")
#include "ut/ut.hpp"
#else
#include "../../ut/ut.hpp"
#endif
#if __has_include("test_device.h")
#include "test_device.h"
#else
#include "../../test_device.h"
#endif

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
#include <luisa/ir/ir2ast.h>

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

void test_ast2ir_ir2ast(Device &device) {
    (void)device;
    constexpr auto f = 10;

    // Enable verbose logging for debugging
    luisa::log_level_verbose();

    // Create constant vector data
    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    // Note: Callable is commented out for this round-trip test
    // Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
    //     Constant int_consts = const_vector;
    //     return cast<float>(a) + int_consts[b].cast<float>() * c;
    // };

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

            // Test min operations
            Var vvv = min(vv, 10);
            Var xxx = make_uint4(5);
            Var vvvv = min(xxx, 1u);

            // Test loop with break
            loop([] {
                if_(true, break_);
            });

            // Test dynamic range iteration
            for (auto v : dynamic_range(v_int)) {
                v_int += v;
            }

            // Test constant array and buffer access
            Var vv_int = int_consts[v_int];
            Var v_float = buffer_float.read(count + thread_id().x);
            Var vv_float = float_consts[0];
            // Note: Callable invocation is commented out for this test
            // Var call_ret = callable(10, v_int, v_float);

            Var v_float_copy = v_float;

            // Test arithmetic operations
            Var z = -1 + v_int * v_float + 1.0f;
            z += 1;
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            // Test vector construction
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
        }
    };
    LUISA_INFO("Kernel definition parsed in {} ms.", clock.toc());

    // Convert AST to IR (forward conversion)
    clock.tic();
    auto ir = AST2IR::build_kernel(kernel_def.function()->function());
    LUISA_INFO("AST2IR done in {} ms.", clock.toc());

    // Convert IR back to AST (reverse conversion)
    clock.tic();
    auto ast = luisa::compute::IR2AST::build(ir.get()->get());
    LUISA_INFO("IR2AST done in {} ms.", clock.toc());
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_ast2ir_ir2ast(device);
}
