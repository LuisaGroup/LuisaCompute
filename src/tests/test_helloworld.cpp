#include <iostream>
#include <chrono>
#include <numeric>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>

#include "../backends/common/c_codegen/codegen_utils.h"
using namespace luisa;
using namespace luisa::compute;
// struct Test1 {
//     int3 something;
//     float a;
// };

// struct Test2 {
//     int3 a;
//     bool b;
// };

// struct Test3 {
//     int a;
//     bool2 b;
//     bool c;
// };

// struct Point3D {
//     float3 v;
// };

// struct TriArray {
//     int v[3];
// };

// struct MDArray {
//     int v[2][3][4];
// };

// LUISA_STRUCT(TriArray, v) {};

// LUISA_STRUCT(Test1, something, a) {};
// LUISA_STRUCT(Test2, a, b) {};
// LUISA_STRUCT(Test3, a, b, c) {};
// LUISA_STRUCT(Point3D, v) {};
// LUISA_STRUCT(MDArray, v) {};

int main(int argc, char *argv[]) {
    luisa::fiber::scheduler sc;
    Context ctx{argv[0]};
    auto device = ctx.create_device("toy-c");
    auto stream = device.create_stream();
    auto shader = device.load_shader<1, Buffer<float4>, Buffer<float>, float4>("kernel");
    
    auto buffer_float4 = device.create_buffer<float4>(1);
    auto buffer_float = device.create_buffer<float>(4);
    float4 &result = *(float4*)buffer_float4.native_handle();
    auto input  = (float*)buffer_float.native_handle();
    for(auto i : vstd::range(4)){
        input[i] = 66 + i;
    }
    stream << shader(buffer_float4, buffer_float, float4( 114514.0f)).dispatch(1) << synchronize();
    LUISA_INFO("{}, {}, {}, {}", result.x, result.y, result.z, result.w);

    // Kernel1D kernel_def = [&](BufferVar<float4> buffer, BufferVar<float> buffer_float, Var<float4> count) noexcept -> void {
    //     auto var = make_float4(
    //         buffer_float.read(0),
    //         buffer_float.read(1),
    //         buffer_float.read(2),
    //         buffer_float.read(3)
    //     );
    //     var += count;
    //     buffer.write(0, var);
    // };
    // Clanguage_CodegenUtils utils;
    // utils.codegen(
    //     "D:/compute/src/backends/common/c_codegen/builtin/test.c",
    //     "kernel",
    //     kernel_def.function()->function());
}
