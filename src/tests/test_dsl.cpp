//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a)

int main() {
    
    FunctionBuilder{Function::Tag::KERNEL}.define([] {
        
        Var v_int = 10;
        Var v_float = 1.0f;
        Var v_float_copy = v_float;

        Var z = -1 + v_int * v_float + 1.0f;
        z += 1;
        static_assert(std::is_same_v<decltype(z), Var<float>>);

        Var v_vec = float3{1.0f};
        Var v2 = float3{2.0f} - v_vec * 2.0f;
        v2 *= 5.0f + v_float;
        
        Var<float2> w{v_int, v_float};
        w *= float2{1.2f};
        
        Var x = w.x;
        Var<int3> s;
        Var<Test> vt{s, v_float_copy};
        Var c = 0.5f + vt.a * 1.0f;
    });
}
