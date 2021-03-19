//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>

#include <runtime/device.h>
#include <dsl/sugar.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a)

int main() {
    
    FakeDevice device;
    auto buffer = device.create_buffer<float4>(1024u);
    auto float_buffer = device.create_buffer<float>(1024u);

    std::vector<int> const_vector{1, 2, 3, 4};

    auto callable = LUISA_CALLABLE($int a, $int b, $float c) noexcept {
        $constant int_consts = const_vector;
        return cast<float>(int_consts[a]) + b.cast<float>() * c;
    };

    auto kernel = LUISA_KERNEL($buffer<float> buffer_float, $uint count) noexcept {

        $constant float_consts = {1.0f, 2.0f};
        $constant int_consts = const_vector;

        $shared<float4> shared_floats{16};

        $array<float, 5> array;

        $ v_int = 10;
        static_assert(std::is_same_v<decltype(v_int), $int>);

        $for(x) : $range(v_int / 2) {
            array[x] = v_int.cast<float>();
        };

        $ v_float = buffer_float[count];
        $ call_ret = callable(10, v_int, v_float);

        $ v_float_copy = v_float;

        $ z = -1 + v_int * v_float + 1.0f;
        z += 1;
        static_assert(std::is_same_v<decltype(z), $float>);

        $ v_vec = float3{1.0f};
        $ v2 = float3{2.0f} - v_vec * 2.0f;
        v2 *= 5.0f + v_float;

        $float2 w{v_int, v_float};
        w *= float2{1.2f};

        $if(w.x < 5) {
        }
        $elif(w.x > 0) {
        }
        $else{

        };

        $while(true){

        };

        $switch(123) {
            $case(1){

            };
            $default{

            };
        };

        $int x = cast<int>(w.x);
        $int3 s{x, x, x};

        $Test vvt{s, v_float_copy};
        $Test vt{vvt};

        $ xx = 1.0f;

        $ vt_copy = vt;
        $ c = 0.5f + vt.a * 1.0f;

        $ vec4 = buffer[10];           // indexing into captured buffer (with literal)
        $ another_vec4 = buffer[v_int];// indexing into captured buffer (with Var)
    };
    
    auto command = kernel(float_buffer, 12u).parallelize(1024u);
    auto launch_command = static_cast<KernelLaunchCommand *>(command.get());
    LUISA_INFO("Command: kernel = {}, args = {}", launch_command->kernel_uid(), launch_command->argument_count());
    auto function = Function::kernel(launch_command->kernel_uid());
}
