#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_state_machine(luisa::test::coro_test::Options options) {

    "state_machine_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        StateMachineCoroScheduler<Buffer<int>>>);
        expect(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                 StateMachineCoroScheduler<Buffer<int>>>);
    };

    "state_machine_compiles_and_runs"_test = [options] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("buf[0]={} buf[1]={}", host[0], host[1]);
        expect(host[0] == 42);
        expect(host[1] == 99);
    };

    "state_machine_local_var_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 99);
    };

    "state_machine_branch_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 10);
    };

    "state_machine_float3_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 6.f);
    };

    "state_machine_bool_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 77);
    };

    "state_machine_multi_var_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 10);
        expect(host[1] == 20);
        expect(host[2] == 30);
        expect(host[3] == 35);
    };

    "state_machine_ref_callable_update_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        Buffer<int> output = device.create_buffer<int>(16);
        Callable bump = [](Int &x) noexcept {
            x = x * 3 + 7;
            return x;
        };

        auto coro = Coroutine<void(Buffer<int>)>([&](Var<Buffer<int>> buf) {
            Int state = 2;
            $suspend("first");
            Int a = bump(state);
            $suspend("second");
            Int b = bump(state);
            buf.write(0, a);
            buf.write(1, b);
            buf.write(2, state);
        });

        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();

        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("ref_callable host[0..2]={}, {}, {}", host[0], host[1], host[2]);
        expect(host[0] == 13);
        expect(host[1] == 46);
        expect(host[2] == 46);
    };

    "state_machine_dispatch_id_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        for (int i = 0; i < 16; i++) {
            expect(host[i] == i * 10);
        }
    };

    "state_machine_nested_if_chain"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 10);
        expect(host[3] == 20);
        expect(host[6] == 30);
        expect(host[10] == 40);
    };

    "state_machine_nested_if_inside_if"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 11);
        expect(host[3] == 22);
        expect(host[8] == 33);
    };

    "state_machine_switch_across_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 100);
        expect(host[1] == 200);
        expect(host[2] == 300);
        expect(host[3] == 400);
        expect(host[4] == 100);
    };

    "state_machine_multiple_suspend_points"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("multi_suspend host[0]={}", host[0]);
        expect(host[0] == 6);
    };

    "state_machine_all_pixels_active"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        for (uint i = 0u; i < N; i++) {
            expect(host[i] == static_cast<int>(i * 7 + 1));
        }
    };

    "state_machine_for_with_suspend_single"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("for_suspend host[0]={}", host[0]);
        expect(host[0] == 102);

    };

    "state_machine_for_if_suspend_branch"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 0;
            $for (i, 4u) {
                acc = acc + 1;
                $if ((i & 1u) == 0u) {
                    acc = acc + 10;
                    $suspend("even");
                } $else {
                    acc = acc + 100;
                };
            };
            buf.write(0, acc);
        });
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 224);
    };

    "state_machine_while_with_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 3);
    };


    "state_machine_suspend_inside_nested_if"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        stream << output.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 111);
        expect(host[2] == 222);
        expect(host[4] == 0);
    };

    "state_machine_double_suspend_linear"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> x = 10;
            $suspend("a");
            x = x + 30;
            $suspend("b");
            buf.write(0, x);
        });
        LUISA_INFO("double_suspend scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("double_suspend host[0]={}", host[0]);
        expect(host[0] == 40);
    };

    "state_machine_triple_suspend_with_loop"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 10;
            $suspend("setup");
            $for (i, 2) {
                acc = acc + 1;
                $suspend("step");
            };
            $suspend("accumulate");
            buf.write(0, acc);
        });
        LUISA_INFO("triple_suspend scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("triple_suspend host[0]={}", host[0]);
        expect(host[0] == 12);
    };

    "state_machine_for_suspend_plus_after_suspend"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 100;
            $for (i, 2) {
                acc = acc + 1;
                $suspend("iter");
            };
            $suspend("after");
            buf.write(0, acc);
        });
        LUISA_INFO("for_plus_after scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("for_plus_after host[0]={}", host[0]);
        expect(host[0] == 102);
    };

    "state_machine_sdf_pattern_5_scopes"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            $suspend("setup");
            $for (i, 2) {
                $suspend("step");
            };
            $suspend("accumulate");
            buf.write(0, 42);
            $suspend("done");
        });
        LUISA_INFO("sdf_5scope scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("sdf_5scope host[0]={}", host[0]);
        expect(host[0] == 42);
    };

    "state_machine_5scope_with_var_in_loop"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            $suspend("setup");
            $for (i, 2) {
                Var<int> v = 0;
                $suspend("step");
            };
            $suspend("accumulate");
            buf.write(0, 42);
            $suspend("done");
        });
        LUISA_INFO("var_in_loop scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("var_in_loop host[0]={}", host[0]);
        expect(host[0] == 42);
    };

    "state_machine_5scope_with_if_read_i"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            $suspend("setup");
            $for (i, 2) {
                Var<int> v = 0;
                $if (i == 0u) { v = 1; };
                $suspend("step");
            };
            $suspend("accumulate");
            buf.write(0, 42);
            $suspend("done");
        });
        LUISA_INFO("if_read_i scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("if_read_i host[0]={}", host[0]);
        expect(host[0] == 42);
    };

    "state_machine_nested_break_preserves_loop_update_count"_test =
        [options] {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();
            Buffer<uint> output = device.create_buffer<uint>(1u);
            auto coro = Coroutine<void(Buffer<uint>)>(
                [](Var<Buffer<uint>> buf) {
                    $suspend("before_search");
                    UInt found = ~0u;
                    $for (i, 16u) {
                        $if (i >= 4u) {
                            $if (i == 5u) {
                                found = i;
                                $break;
                            };
                        };
                    };
                    buf.write(0u, found);
                });
            StateMachineCoroScheduler<Buffer<uint>> scheduler{
                device, coro};
            scheduler(output).dispatch(1u)(stream);
            std::array<uint, 1u> host{};
            stream << output.copy_to(luisa::span{host})
                   << synchronize();
            expect(host[0] == 5u)
                << "an enclosing loop update must execute exactly once "
                   "when a nested selection exits through it";
        };

    "state_machine_path_tracing_nested_break_pattern"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        Buffer<int> output = device.create_buffer<int>(16);
        auto coro = Coroutine<void(Buffer<int>)>([](Var<Buffer<int>> buf) {
            Var<int> acc = 0;
            $suspend("per_spp");
            $for (i, 4u) {
                acc = acc + 1;
                $suspend("per_depth");
                $for (depth, 3u) {
                    $suspend("before");
                    acc = acc + 10;
                    $if (depth == 1u) {
                        $break;
                    };
                    $suspend("after");
                    acc = acc + 100;
                };
            };
            $suspend("write");
            buf.write(0, acc);
        });
        LUISA_INFO("nested_break scope_count={}", coro.subroutine_count());
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(1)(stream);
        stream << synchronize();
        std::vector<int> host(16, -1);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("nested_break host[0]={}", host[0]);
        expect(host[0] == 484);
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_state_machine(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
