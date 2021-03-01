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

class FakeDevice : public Device {

    void _dispose_buffer(uint64_t handle) noexcept override {}

    uint64_t _create_buffer(size_t byte_size) noexcept override {
        return 0;
    }

    uint64_t _create_buffer_with_data(size_t size_bytes, const void *data) noexcept override {
        return 0;
    }
};

int main() {

    FakeDevice device;
    Buffer<float4> buffer{&device, 1024u};
    
    auto callable = LUISA_CALLABLE(Var<int> a, Var<int> b, Var<float> c) noexcept {
      return a + b * c;
    };

    auto kernel = LUISA_KERNEL(Var<Buffer<float>> buffer_float, Var<uint> count) noexcept {

        Var v_int = 10;
        Var v_float = buffer_float[count];
        Var call_ret = callable(10, v_int, v_float);

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
        Var vt_copy = vt;
        Var c = 0.5f + vt.a * 1.0f;

        Var vec4 = buffer[10];           // indexing into captured buffer (with literal)
        Var another_vec4 = buffer[v_int];// indexing into captured buffer (with Var)

        Var<Buffer<int>> b{dsl::detail::ArgumentCreation{}};
        Var bb = b[12];
    };
}
