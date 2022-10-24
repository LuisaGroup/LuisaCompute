//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>

#include <runtime/device.h>
#include <dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a) {};
using $Test = Var<Test>;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto buffer = device.create_buffer<float4>(1024u);
    auto float_buffer = device.create_buffer<float>(1024u);

    std::vector<int> const_vector{1, 2, 3, 4};

    Callable callable = [&]($int a, $int b, $float c) noexcept {
        $constant int_consts = const_vector;
        return cast<float>(int_consts[a]) + b.cast<float>() * c;
    };

    Kernel1D kernel = [&]($buffer<float> buffer_float, $uint count) noexcept {
        $constant float_consts = {1.0f, 2.0f};
        $constant int_consts = const_vector;

        $shared<float4> shared_floats{16};

        $array<float, 5> array;

        $ v_int = 10;
        static_assert(std::is_same_v<decltype(v_int), $int>);

        $for(x, 1) {
            array[x] = cast<float>(v_int);
        };

        $ v_float = buffer_float.read(count);
        $ call_ret = callable(10, v_int, v_float);

        $ v_float_copy = v_float;

        $ z = -1 + v_int * v_float + 1.0f;
        z += 1;

        $ v_vec = make_float3(1.0f);
        $ v2 = make_float3(2.0f) - v_vec * 2.0f;
        v2 *= 5.0f + v_float;

        $float2 w{cast<float>(v_int), v_float};
        w *= float2{1.2f};

        $if(w.x < 5) {
        }
        $elif(w.x > 0) {
        }
        $else{

        };

        $loop {
            $break;
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

        $ vec4 = buffer.read(10);           // indexing into captured buffer (with literal)
        $ another_vec4 = buffer.read(v_int);// indexing into captured buffer (with Var)
    };

    auto shader = device.compile(kernel);
    auto command = shader(float_buffer, 12u).dispatch(1024u);
    auto launch_command = static_cast<ShaderDispatchCommand *>(command.get());
    LUISA_INFO("Command: kernel = {}, args = {}", hash_to_string(launch_command->kernel().hash()), launch_command->argument_count());
}
