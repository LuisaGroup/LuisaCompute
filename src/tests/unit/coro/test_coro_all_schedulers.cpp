#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <array>
#include <cstdlib>

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/rtx/accel.h>
#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/stream.h>
#include <luisa/xir/function.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_all_schedulers(luisa::test::coro_test::Options options) {

    auto expect_filled = [](luisa::span<const uint> host, uint base, luisa::string_view label) noexcept {
        auto ok = true;
        for (auto i = 0u; i < host.size(); i++) {
            auto expected = base + i;
            if (host[i] != expected) {
                LUISA_WARNING("{} mismatch at {}: got {}, expected {}",
                              label, i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "all coroutine instances should write expected values";
    };

    // ══════════════════════════════════════════════════════════════════
    // 1-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_1suspend_state_machine"_test = [options, expect_filled] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 11u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect_filled(host, 11u, "cross_1suspend_state_machine");
    };

    "cross_1suspend_wavefront"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 12u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect_filled(host, 12u, "cross_1suspend_wavefront");
    };

    "cross_1suspend_persistent"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 13u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = N, .block_size = N}};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching {} logical instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect_filled(host, 13u, "cross_1suspend_persistent");
    };

    // ══════════════════════════════════════════════════════════════════
    // 3-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_3suspend_state_machine"_test = [options, expect_filled] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 31u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect_filled(host, 31u, "cross_3suspend_state_machine");
    };

    "cross_3suspend_wavefront"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 32u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect_filled(host, 32u, "cross_3suspend_wavefront");
    };

    "cross_3suspend_persistent"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 33u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = N, .block_size = N}};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching {} logical instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect_filled(host, 33u, "cross_3suspend_persistent");
    };

    "ray_query_loop_is_normalized_before_coroutine_destructuring"_test =
        [options] {
            constexpr uint N = 32u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();
            auto output = device.create_buffer<uint>(N);

            constexpr std::array vertices{
                make_float3(-1.0f, -1.0f, 0.0f),
                make_float3(1.0f, -1.0f, 0.0f),
                make_float3(0.0f, 1.0f, 0.0f)};
            constexpr std::array triangles{
                Triangle{0u, 1u, 2u}};
            auto vertex_buffer =
                device.create_buffer<float3>(vertices.size());
            auto triangle_buffer =
                device.create_buffer<Triangle>(triangles.size());
            auto mesh =
                device.create_mesh(vertex_buffer, triangle_buffer);
            auto accel = device.create_accel();
            // Non-opaque geometry makes the surface callback observable.
            accel.emplace_back(
                mesh, make_float4x4(1.0f), 0xffu, false);

            auto coro = Coroutine<void(Buffer<uint>)>(
                [&accel](BufferUInt result) noexcept {
                    auto tid = dispatch_x();
                    $suspend("before-ray-query");
                    auto ray = make_ray(
                        make_float3(0.0f, 0.0f, 1.0f),
                        make_float3(0.0f, 0.0f, -1.0f));
                    UInt callback_count = 0u;
                    auto hit =
                        accel->traverse(ray, {})
                            .on_surface_candidate(
                                [&](SurfaceCandidate &candidate) noexcept {
                                    callback_count += 1u;
                                    candidate.commit();
                                })
                            .on_procedural_candidate(
                                [](ProceduralCandidate &) noexcept {})
                            .trace();
                    result.write(
                        tid,
                        callback_count +
                            select(0u, 10u, !hit->miss()));
                });

            expect(coro.subroutine_count() == 2u)
                << "ray-query lowering must preserve the suspend boundary";

            stream << vertex_buffer.copy_from(luisa::span{vertices})
                   << triangle_buffer.copy_from(luisa::span{triangles})
                   << mesh.build()
                   << accel.build()
                   << synchronize();

            auto clear_and_check =
                [&](auto &&dispatch, luisa::string_view label) {
                    luisa::vector<uint> zero(N);
                    stream << output.copy_from(luisa::span{zero});
                    dispatch();
                    luisa::vector<uint> host(N);
                    stream << output.copy_to(luisa::span{host})
                           << synchronize();
                    auto valid = true;
                    for (auto value : host) {
                        valid &= value == 11u;
                    }
                    expect(valid) << label;
                };

            StateMachineCoroScheduler<Buffer<uint>> state_machine{
                device, coro};
            clear_and_check(
                [&] { state_machine(output).dispatch(N)(stream); },
                "ray query/state machine");

            WavefrontCoroScheduler<Buffer<uint>> wavefront{
                device, coro,
                WavefrontCoroSchedulerConfig{.thread_count = N}};
            clear_and_check(
                [&] { wavefront(output).dispatch(N)(stream); },
                "ray query/wavefront");

            PersistentThreadsCoroScheduler<Buffer<uint>> persistent{
                device, coro,
                PersistentThreadsCoroSchedulerConfig{
                    .thread_count = N, .block_size = N}};
            clear_and_check(
                [&] { persistent(output).dispatch(N)(stream); },
                "ray query/persistent");
        };

    "dead_frontend_suspend_keeps_sparse_callable_token_pairing"_test = [options, expect_filled] {
        constexpr uint N = 64u;
        // FunctionBuilder assigns these tokens before XIR reachability
        // optimization. The first suspend is removed, but its token must not
        // be renumbered onto the surviving continuation.
        constexpr uint dead_token = 1u;
        constexpr uint live_token = 2u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $if (Expr<bool>{false}) {
                $suspend("dead");
            };
            $suspend("live");
            output.write(tid, tid + 73u);
        });

        expect(coro.subroutine_count() == 2u)
            << "only entry and the live sparse-token continuation may be lowered";
        expect(coro.graph().node_count() == coro.subroutine_count());
        expect(coro.graph().node_by_token(dead_token) == nullptr)
            << "a suspend in unreachable control flow must have no graph node/callable";
        expect(coro.graph().node_by_token(live_token) != nullptr);
        expect(coro.trigger_token(0u) == 0u);
        expect(coro.trigger_token(1u) == live_token)
            << "the live callable must retain its sparse front-end token";

        auto clear_and_check = [&](auto &&dispatch, luisa::string_view label) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero});
            dispatch();
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            expect_filled(host, 73u, label);
        };

        StateMachineCoroScheduler<Buffer<uint>> state_machine{device, coro};
        clear_and_check(
            [&] { state_machine(output).dispatch(N)(stream); },
            "dead_suspend_sparse_token_state_machine");

        WavefrontCoroScheduler<Buffer<uint>> wavefront{device, coro};
        clear_and_check(
            [&] { wavefront(output).dispatch(N)(stream); },
            "dead_suspend_sparse_token_wavefront");

        PersistentThreadsCoroScheduler<Buffer<uint>> persistent{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{
                .thread_count = N, .block_size = N}};
        clear_and_check(
            [&] { persistent(output).dispatch(N)(stream); },
            "dead_suspend_sparse_token_persistent");
    };

    "all_dead_suspends_lower_to_entry_only_coroutine"_test = [options, expect_filled] {
        constexpr uint N = 64u;
        constexpr uint dead_token = 37u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $if (Expr<bool>{false}) {
                $suspend(dead_token, "dead-only");
            };
            output.write(tid, tid + 83u);
        });

        expect(coro.subroutine_count() == 1u)
            << "an optimized suspend set T_live = empty must retain exactly the entry callable";
        expect(coro.graph().node_count() == 1u);
        expect(coro.graph().node(0u).token == 0u);
        expect(coro.graph().node_by_token(dead_token) == nullptr);
        expect(coro.trigger_token(0u) == 0u);

        auto clear_and_check = [&](auto &&dispatch, luisa::string_view label) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero});
            dispatch();
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            expect_filled(host, 83u, label);
        };

        StateMachineCoroScheduler<Buffer<uint>> state_machine{device, coro};
        clear_and_check(
            [&] { state_machine(output).dispatch(N)(stream); },
            "all_dead_suspend_state_machine");

        WavefrontCoroScheduler<Buffer<uint>> wavefront{device, coro};
        clear_and_check(
            [&] { wavefront(output).dispatch(N)(stream); },
            "all_dead_suspend_wavefront");

        PersistentThreadsCoroScheduler<Buffer<uint>> persistent{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{
                .thread_count = N, .block_size = N}};
        clear_and_check(
            [&] { persistent(output).dispatch(N)(stream); },
            "all_dead_suspend_persistent");
    };

    "zero_frontend_suspends_lower_to_entry_only_coroutine"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            output.write(tid, tid + 89u);
        });

        expect(coro.subroutine_count() == 1u)
            << "T_front = empty must retain exactly the root entry callable";
        expect(coro.graph().node_count() == 1u);
        expect(coro.graph().node(0u).token == 0u);
        expect(coro.trigger_token(0u) == 0u);

        auto clear_and_check = [&](auto &&dispatch, luisa::string_view label) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero});
            dispatch();
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            expect_filled(host, 89u, label);
        };

        StateMachineCoroScheduler<Buffer<uint>> state_machine{device, coro};
        clear_and_check(
            [&] { state_machine(output).dispatch(N)(stream); },
            "zero_frontend_suspend_state_machine");

        WavefrontCoroScheduler<Buffer<uint>> wavefront{device, coro};
        clear_and_check(
            [&] { wavefront(output).dispatch(N)(stream); },
            "zero_frontend_suspend_wavefront");

        PersistentThreadsCoroScheduler<Buffer<uint>> persistent{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{
                .thread_count = N, .block_size = N}};
        clear_and_check(
            [&] { persistent(output).dispatch(N)(stream); },
            "zero_frontend_suspend_persistent");
    };

    "dead_suspend_before_nested_loop_header_preserves_sparse_cutpoints"_test =
        [options, expect_filled] {
            constexpr uint N = 64u;
            constexpr uint sample_count = 4u;
            constexpr uint dead_token = 1u;
            constexpr uint sample_token = 2u;
            constexpr uint bounce_token = 3u;
            // Per sample: (10 * sample + 1) + sum(1, 2, 3).
            constexpr uint expected_increment = 88u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();
            auto output = device.create_buffer<uint>(N);

            auto coro = Coroutine<void(Buffer<uint>, uint)>(
                [](BufferUInt output, UInt samples) noexcept {
                    auto tid = dispatch_x();
                    UInt total = tid;
                    // This consumes a front-end token, but is removed before
                    // scope distillation and callable materialization.
                    $if (Expr<bool>{false}) {
                        $suspend("dead-before-sample-loop");
                    };
                    $for (sample, samples) {
                        // A loop-header suspension turns every dynamic sample
                        // iteration into a scheduler cycle. Its back-edge must
                        // target this sparse token, not the eliminated token.
                        $suspend("sample-loop-header");
                        total += sample * 10u + 1u;
                        $for (bounce, 3u) {
                            $suspend("bounce-loop-body");
                            total += bounce + 1u;
                        };
                    };
                    output.write(tid, total);
                });

            expect(coro.subroutine_count() == 3u)
                << "only entry and the two reachable loop cutpoints may be lowered";
            expect(coro.graph().node_count() == coro.subroutine_count());
            expect(coro.graph().node_by_token(dead_token) == nullptr)
                << "the optimized-away front-end token must not acquire a callable";
            auto sample_node = coro.graph().node_by_name("sample-loop-header");
            auto bounce_node = coro.graph().node_by_name("bounce-loop-body");
            expect(sample_node != nullptr && sample_node->token == sample_token);
            expect(bounce_node != nullptr && bounce_node->token == bounce_token);
            expect(coro.trigger_token(0u) == 0u);
            expect(coro.trigger_token(1u) == sample_token);
            expect(coro.trigger_token(2u) == bounce_token);

            auto clear_and_check = [&](auto &&dispatch, luisa::string_view label) {
                luisa::vector<uint> zero(N);
                stream << output.copy_from(luisa::span{zero});
                dispatch();
                luisa::vector<uint> host(N);
                stream << output.copy_to(luisa::span{host}) << synchronize();
                expect_filled(host, expected_increment, label);
            };

            StateMachineCoroScheduler<Buffer<uint>, uint> state_machine{
                device, coro};
            clear_and_check(
                [&] { state_machine(output, sample_count).dispatch(N)(stream); },
                "nested_loop_sparse_token_state_machine");

            WavefrontCoroScheduler<Buffer<uint>, uint> wavefront{device, coro};
            clear_and_check(
                [&] { wavefront(output, sample_count).dispatch(N)(stream); },
                "nested_loop_sparse_token_wavefront");

            PersistentThreadsCoroScheduler<Buffer<uint>, uint> persistent{
                device, coro,
                PersistentThreadsCoroSchedulerConfig{
                    .thread_count = N, .block_size = N}};
            clear_and_check(
                [&] { persistent(output, sample_count).dispatch(N)(stream); },
                "nested_loop_sparse_token_persistent");
        };

    "coroutine_lowering_preserves_structured_helper_identity"_test = [] {
        // An ordinary helper is a dependency of the coroutine, not a member of
        // its state machine. Coroutine CFG passes must therefore leave the
        // helper's structured AST identity unchanged. This is important for
        // large shader graphs: a whole-module destructure/restructure makes
        // compilation scale with dependency size even though the helper does
        // not participate in coroutine scheduling.
        constexpr auto helper_name =
            "coro-structured-helper-preservation-oracle";
        Callable<uint(uint)> structured_helper = [](UInt x) noexcept {
            UInt result = x * 3u + 1u;
            $switch (x % 3u) {
                $case (0u) { result += 5u; };
                $case (1u) { result ^= 9u; };
                $default { result += 17u; };
            };
            $for (i, 4u) {
                $if (((x + i) & 1u) != 0u) {
                    result += i + 2u;
                } $else {
                    result ^= i + 3u;
                };
            };
            return result;
        };
        structured_helper.set_name(helper_name);

        // XIR-to-AST translation has its own canonical representation, so the
        // oracle must cross the same translation boundary as a lowered
        // continuation. Comparing against the original DSL AST hash would
        // conflate that canonicalization with coroutine CFG mutation.
        auto oracle_module = xir::ast_to_xir_translate(
            structured_helper.function(), {});
        const xir::FunctionDefinition *oracle_definition = nullptr;
        for (auto *function : oracle_module->function_list()) {
            if (function->isa<xir::CallableFunction>() &&
                function->definition() != nullptr) {
                oracle_definition = function->definition();
                break;
            }
        }
        expect(oracle_definition != nullptr);
        auto oracle_ast = xir::xir_to_ast_translate(
            *oracle_definition, {});
        expect(oracle_ast != nullptr);
        auto canonical_helper_hash = oracle_ast->hash();

#ifdef _WIN32
        _putenv_s("LUISA_CORO_VERIFY_PASS_DOMAIN", "1");
#else
        setenv("LUISA_CORO_VERIFY_PASS_DOMAIN", "1", 1);
#endif
        auto coroutine = Coroutine<void(Buffer<uint>)>(
            [&structured_helper](BufferUInt output) noexcept {
                auto tid = dispatch_x();
                $suspend("before-structured-helper");
                output.write(tid, structured_helper(tid));
            });
        auto lowered = luisa::compute::detail::compile_coroutine_pipeline(
            coroutine.function_builder());
#ifdef _WIN32
        _putenv_s("LUISA_CORO_VERIFY_PASS_DOMAIN", "");
#else
        unsetenv("LUISA_CORO_VERIFY_PASS_DOMAIN");
#endif

        expect(lowered.boundary_verifier_count == 2u)
            << "the complete coroutine transaction must verify only its input and output";
        expect(lowered.nested_pass_boundary_verifier_count == 0u)
            << "composed passes must not repeat full-XIR boundary verification";

#ifdef _WIN32
        _putenv_s("LUISA_XIR_VERIFY_INTERMEDIATE", "1");
#else
        setenv("LUISA_XIR_VERIFY_INTERMEDIATE", "1", 1);
#endif
        auto diagnostic_lowered =
            luisa::compute::detail::compile_coroutine_pipeline(
                coroutine.function_builder());
#ifdef _WIN32
        _putenv_s("LUISA_XIR_VERIFY_INTERMEDIATE", "");
#else
        unsetenv("LUISA_XIR_VERIFY_INTERMEDIATE");
#endif
        expect(diagnostic_lowered.boundary_verifier_count == 2u);
        expect(diagnostic_lowered.nested_pass_boundary_verifier_count > 0u)
            << "the explicit diagnostic environment flag must restore nested pass boundaries";

        size_t dependency_count = 0u;
        size_t canonical_dependency_count = 0u;
        for (auto &&subroutine : lowered.subroutines) {
            for (auto &&dependency : subroutine->custom_callables()) {
                dependency_count++;
                canonical_dependency_count +=
                    dependency->hash() == canonical_helper_hash;
            }
        }
        expect(dependency_count == 1u)
            << "the live continuation must retain its structured helper dependency";
        expect(canonical_dependency_count == dependency_count)
            << "coroutine-only CFG passes must preserve the canonical form of every ordinary helper";
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_all_schedulers(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
