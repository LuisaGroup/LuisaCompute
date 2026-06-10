#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_state_machine(char *argv[]) {

    "state_machine_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        StateMachineCoroScheduler<Buffer<int>>>);
        expect(true);
    };

    "state_machine_compiles_and_runs"_test = [argv] {
        constexpr uint N = 256u;

        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        {
            auto k = Kernel1D{[]() noexcept {}};
            auto s = device.compile(k);
            stream << s().dispatch(N) << synchronize();
            LUISA_INFO("Basic kernel dispatch OK");
        }

        Buffer<int> output = device.create_buffer<int>(N);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            buf.write(0, 42);
            $suspend("a");
            buf.write(1, 99);
        });

        LUISA_INFO("Coroutine created, subroutine_count={}", coro.subroutine_count());

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");

        std::vector<int> host(N, -1);
        stream << output.copy_to(host.data()) << synchronize();
        LUISA_INFO("buf[0]={} buf[1]={}", host[0], host[1]);
        expect(host[0] == 42);
        expect(host[1] == 99);
    };

    "state_machine_local_var_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> x = 42;
            $suspend("a");
            Var<int> y = x + 57;
            buf.write(0, y);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 99);
    };

    "state_machine_branch_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> x = 0;
            $if (dispatch_id().x == 0u) {
                x = 10;
            } $else {
                x = 20;
            };
            $suspend("a");
            buf.write(0, x);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 10);
    };

    "state_machine_float3_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<float> output = device.create_buffer<float>(16);

        auto coro = Coroutine<void(Buffer<float>)>([](Var<Buffer<float>> buf) {
            Float3 v = make_float3(1.f, 2.f, 3.f);
            $suspend("a");
            Float sum = v.x + v.y + v.z;
            buf.write(0, sum);
        });

        StateMachineCoroScheduler<Buffer<float>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<float> host(16, -1.f);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 6.f);
    };

    "state_machine_bool_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Bool flag = (dispatch_id().x < 1u);
            $suspend("a");
            $if (flag) {
                buf.write(0, 77);
            } $else {
                buf.write(0, 88);
            };
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 77);
    };

    "state_machine_multi_var_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> a = 10, b = 20;
            Var<int> c = a + b;
            $suspend("a");
            Var<int> d = c + 5;
            buf.write(0, a);
            buf.write(1, b);
            buf.write(2, c);
            buf.write(3, d);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 10);
        expect(host[1] == 20);
        expect(host[2] == 30);
        expect(host[3] == 35);
    };

    "state_machine_dispatch_id_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x;
            $suspend("a");
            buf.write(tid, tid.cast<int>() * 10);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(16)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        for (int i = 0; i < 16; i++) {
            expect(host[i] == i * 10);
        }
    };

    "state_machine_nested_if_chain"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x;
            Var<int> val = 0;

            $if (tid < 3u) {
                val = 10;
            } $elif (tid < 6u) {
                val = 20;
            } $elif (tid < 9u) {
                val = 30;
            } $else {
                val = 40;
            };

            $suspend("chain");
            buf.write(tid, val);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(12)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 10);
        expect(host[3] == 20);
        expect(host[6] == 30);
        expect(host[10] == 40);
    };

    "state_machine_nested_if_inside_if"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x;
            Var<int> val = 0;

            $if (tid < 6u) {
                $if (tid < 3u) {
                    val = 11;
                } $else {
                    val = 22;
                };
            } $else {
                val = 33;
            };

            $suspend("nested");
            buf.write(tid, val);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(12)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 11);
        expect(host[3] == 22);
        expect(host[8] == 33);
    };

    "state_machine_switch_across_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x % 4u;
            Var<int> val = 0;
            $switch (tid) {
                $case (0u) { val = 100; };
                $case (1u) { val = 200; };
                $case (2u) { val = 300; };
                $default   { val = 400; };
            };
            $suspend("switch");
            buf.write(dispatch_id().x, val);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(8)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 100);
        expect(host[1] == 200);
        expect(host[2] == 300);
        expect(host[3] == 400);
        expect(host[4] == 100);
    };

    "state_machine_multiple_suspend_points"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> a = 1;
            $suspend("s1");
            Var<int> b = a + 1;
            $suspend("s2");
            Var<int> c = b + 1;
            $suspend("s3");
            buf.write(0, a + b + c);
        });

        LUISA_INFO("multi_suspend: scope_count={}", coro.subroutine_count());

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        LUISA_INFO("multi_suspend host[0]={}", host[0]);
        expect(host[0] == 6);
    };

    "state_machine_all_pixels_active"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        constexpr uint N = 256u;
        Buffer<int> output = device.create_buffer<int>(N);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x;
            Var<int> val = tid.cast<int>() * 7;
            $suspend("mult");
            buf.write(tid, val + 1);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(N)(stream);
        stream << synchronize();

        std::vector<int> host(N, -1);
        stream << output.copy_to(host.data()) << synchronize();
        for (uint i = 0u; i < N; i++) {
            expect(host[i] == static_cast<int>(i * 7 + 1));
        }
    };

#if 0 // $for/$while+suspend, $suspend-inside-$if: needs per-scope cfg-distill + phi→alloca
    "state_machine_for_with_suspend_single"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 100;
            $for (i, 2) {
                acc = acc + 1;
                $suspend("iter");
            };
            buf.write(0, acc);
        });

        LUISA_INFO("for_suspend scope_count={}", coro.subroutine_count());

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 102);
    };

    "state_machine_while_with_suspend"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 0;
            Var<int> i = 0;
            $while (i < 3) {
                acc = acc + i;
                i = i + 1;
                $suspend("iter");
            };
            buf.write(0, acc);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 3);
    };

    "state_machine_suspend_inside_nested_if"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device(argv[1]);
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);

        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            UInt tid = dispatch_id().x;
            Var<int> val = 0;
            $if (tid < 4u) {
                $if (tid < 2u) {
                    val = 111;
                } $else {
                    val = 222;
                };
                $suspend("inner");
            };
            buf.write(tid, val);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(6)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(host.data()) << synchronize();
        expect(host[0] == 111);
        expect(host[2] == 222);
        expect(host[4] == 0);
    };
#endif

}

int main(int argc, char *argv[]) {
    reg_coro_state_machine(argv);
    return 0;
}
