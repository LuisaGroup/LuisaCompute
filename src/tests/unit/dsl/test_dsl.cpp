// Test for Domain-Specific Language (DSL) features including:
// - Struct definitions with LUISA_STRUCT macro
// - Template struct definitions with LUISA_TEMPLATE_STRUCT macro
// - Callable functions and kernel definitions
// - Buffer operations (read, write, volatile operations)
// - Control flow constructs (if, switch, loop, for)
// - Variable declarations and type casting
// - Constant values and shared memory
// - Bindless array access

#include "luisa/dsl/struct.h"
#include "ut/ut.hpp"
#include "test_device.h"
#include <iostream>
#include <chrono>
#include <numeric>

#include <luisa/core/clock.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/bindless_array.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure with int3 vector and float member
struct Test1 {
    int3 something;
    float a;
};

// Test structure with int3 vector and bool member
struct Test2 {
    int3 a;
    bool b;
};

// Test structure with mixed scalar and vector types
struct Test3 {
    int a;
    bool2 b;
    bool c;
};

// Simple 3D point structure wrapping a float3
struct Point3D {
    float3 v;
};

// Structure containing a fixed-size array
struct TriArray {
    int v[3];
};

// Structure containing a multi-dimensional array
struct MDArray {
    int v[2][3][4];
};

// Register structures with the DSL using LUISA_STRUCT macro
LUISA_STRUCT(TriArray, v) {};
LUISA_STRUCT(Test1, something, a) {};
LUISA_STRUCT(Test2, a, b) {};
LUISA_STRUCT(Test3, a, b, c) {};
LUISA_STRUCT(Point3D, v) {};
LUISA_STRUCT(MDArray, v) {};

// Template structure for key-value pairs
template<typename IndexType, typename ValueType>
struct KeyValuePair {
    IndexType key;
    ValueType value;
};

// Define template macros for LUISA_TEMPLATE_STRUCT
#define LUISA_KEY_VALUE_PAIR_TEMPLATE() \
    template<typename IndexType, typename ValueType>
#define LUISA_KEY_VALUE_PAIR() KeyValuePair<IndexType, ValueType>

// Register template structure with the DSL
LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value){};

void test_dsl(Device &device) {

    constexpr auto f = 10;

    luisa::log_level_verbose();

    // Create buffers for testing
    auto buffer = device.create_buffer<float4>(1024u);
    auto float_buffer = device.create_buffer<float>(1024u);

    // Callable that accesses bindless array buffer
    Callable bb = [&](BindlessVar a) noexcept {
        return a.buffer<TriArray>(0u).read(0u);
    };
    bb.function_builder()->set_name("my_function_bb");

    // Callable demonstrating struct variables and buffer capture
    Callable c1 = [&](UInt a) noexcept {
        Var<Point3D> p1;
        Var<Point3D> p2{make_float3(1.f)};
        Var<Test1> t1;
        Var<Test2> t2;
        Var<Test3> t3;
        Var<KeyValuePair<int, float>> kvp1;
        return buffer->read(a + thread_x());// captures buffer
    };
    c1.function_builder()->set_name("my_function_c1");

    // Callable that calls another callable (c1) and captures additional buffer
    Callable c2 = [&](UInt b) noexcept {
        // captures buffer (propagated from c1) and float_buffer
        return c1(b) + make_float4(float_buffer->read(b));
    };
    c2.function_builder()->set_name("my_function_c2");

    // Kernel that uses callable c2 and writes to buffer
    Kernel1D k1 = [&] {
        // captures buffer and float_buffer (propagated from c2)
        auto v = c2(dispatch_x());
        auto z = sign(v);
        float_buffer->write(dispatch_x(), z.x + z.y + z.z);
    };

    // Create constant vector for testing
    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    // Callable demonstrating type casting and tuple composition
    Callable add_mul = [&](Var<int> a, Var<int> b) noexcept {
        a.set_name("a");
        b.set_name("b");
        return compose(cast<int>(float_buffer->read(a + b)), a * b);
    };
    add_mul.function_builder()->set_name("my_function_add_mul");

    // Callable with constant array access and type casting
    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        a.set_name("a");
        b.set_name("b");
        c.set_name("c");
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };
    callable.function_builder()->set_name("my_function_callable");

    // Callable binding to template lambdas
    Callable<int(int, int)> add = [&]<typename T>(Var<T> a, Var<T> b) noexcept {
        a.set_name("a");
        b.set_name("b");
        return cast<int>(c1(cast<uint>(a + b)).x);
    };
    add.function_builder()->set_name("my_function_add");

    Clock clock;
    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    // Main kernel definition demonstrating various DSL features
    auto kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count, Var<BindlessArray> heap, BufferVar<int3> b0, BufferVar<float4x4> b1, Var<ByteBuffer> bb) noexcept -> void {
        using namespace dsl_literals;

        // Test volatile read/write operations on different buffer types
        b0.volatile_write(1, b0.volatile_read(0));
        b1.volatile_write(1, b1.volatile_read(0));
        bb.volatile_write(16, bb.volatile_read<float3>(1));
        bb.volatile_write(16, bb.volatile_read<float3x3>(1));

        // Test literal suffixes for type specification
        auto lx = 0._half;
        auto ly = 0._float;
        auto lz = 0_ulong2;
        lx.set_name("lx");
        ly.set_name("ly");
        lz.set_name("lz");

        // Shared memory allocation
        Shared<float4> shared_floats{16};

        // Constant array access
        Constant float_consts = {1.0f, 2.0f};
        auto ff = float_consts.read(0);

        // Variable declarations with different initialization methods
        Var v_int = 10;
        Var t = make_int3(1, 2, 3);
        Var vv = ite(t == 10, 1, 2);

        Var vvv = min(vv, 10);
        Var xxx = make_uint4(5);
        Var vvvv = min(xxx, 1u);

        // Tuple unpacking from callable result
        Var am = add_mul(v_int, v_int);
        Var a_copy = am.get<0>();
        Var m_copy = am.get<1>();

        // Loop with break statement
        loop([] {
            if_(true, break_);
        });

        // Dynamic range loop
        for (auto v : dynamic_range(v_int)) {
            v_int += v;
        }

        // Various operations testing
        Var v_int_add_one = add(v_int, 1);
        Var vv_int = int_consts[v_int];
        Var v_float = buffer_float.read(count + thread_id().x);
        Var vv_float = float_consts[0];
        Var call_ret = callable(10, v_int, v_float);

        Var v_float_copy = v_float;

        // Arithmetic operations with automatic type promotion
        Var z = -1 + v_int * v_float + 1.0f;
        z += 1;
        Var v_vec = float3{1.0f};
        Var v2 = float3{2.0f} - v_vec * 2.0f;
        v2 *= 5.0f + v_float;

        Var<float2> w{cast<float>(v_int), v_float};
        w *= float2{1.2f};

        // Conditional statement
        if_(v_int == v_int, [] {
            Var a = 0.0f;
        }).else_([] {
            Var c = 2.0f;
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

        // Struct variable initialization and member access
        Var<int3> s;
        Var<Test1> vvt{s, v_float_copy};
        Var<Test1> vt{vvt};

        Var vt_copy = vt;
        Var c = 0.5f + vt.a * 1.0f;

        // Template struct variable
        Var<KeyValuePair<int, float>> kvp2{vt.a, 0.5f};

        // Buffer operations: indexing with literal and variable
        Var vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
        Var another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
        another_vec4 += heap.buffer<float4>(0).read(0);
        buffer->write(v_int + 1, float4(123.0f));

        // Volatile buffer operations
        auto test_volatile = buffer_float.volatile_read(v_int + 1);
        buffer_float.volatile_write(v_int + 1, test_volatile);
        buffer->volatile_write(v_int + 1, float4(123.0f));
    };
    auto t1 = clock.toc();

    auto kernel = device.compile<2>(kernel_def);
    expect(true) << "DSL kernel compiled successfully";
    // auto command = kernel(float_buffer, 12u).dispatch(1024u);
    // auto launch_command = static_cast<ShaderDispatchCommand *>(command.get());
}
// TODO Change 'static inline const auto reg" to: 
// int main(int argc, char *argv[]) {
//     auto dc = luisa::test::create_device_from_ut(argc, argv);
//     if (!dc) {
//         return 0;
//     }
//     auto &device = dc->device;
//     test_dsl(device);
// }

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_dsl(device);
}
