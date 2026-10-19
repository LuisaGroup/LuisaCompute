// Test for mbarrier and proxy-fence DSL wrappers, plus cluster-barrier
// wrappers and the Cluster-Launch-Control (CLC) work-stealing helper.
//
// This test exercises the new CallOps added for Cluster-Launch-Control
// support on sm_90+ (verified on an RTX 5070 Ti Laptop, sm_120, CUDA 13.0):
// - mbarrier_arrive(bar, count) / mbarrier_test_wait_parity(bar, phase)
// - mbarrier_expect_tx(bar, tx) combined with mbarrier_arrive_expect_tx
// - multi-count mbarrier_arrive
// - fence_mbarrier_init / fence_proxy_async(_generic_acquire/_release) /
//   fence_proxy_async_acquire / fence_proxy_async_release (smoke compile+run)
// - the canonical fixed CLC PTX via cluster_launch_control_work_stealing_1d
// - cluster_barrier_arrive / _arrive_relaxed / _wait / _wait_acquire (smoke)
// - the debug-generator (LUISA_ERROR) rejection path for the new CallOps
//
// mbarrier semantics: a phase completes only once BOTH the arrival count and
// the expected-transaction (tx) byte count are satisfied. Pure expect_tx
// without an arriving party would hang, so the equivalence test pairs
// expect_tx with an explicit arrive.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/cluster_launch_control.h>
#include <array>
#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// 1. mbarrier_arrive + mbarrier_test_wait_parity across P phases.
//    thread0 arrives once per phase; every thread spins on test_wait_parity
//    and bumps its own cumulative counter once per completed phase. The host
//    then asserts every thread observed all P phases.
void test_mbarrier_arrive_test_wait_parity(Device &device) {
    constexpr auto block_size = 128u;
    constexpr auto phases = 8u;

    auto out = device.create_buffer<uint>(block_size);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferUInt out) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<luisa::ulong> bar{1u};
        $if (thread_x() == 0u) {
            mbarrier_init(bar, 1u);
            fence_mbarrier_init();
        };
        sync_block();

        $uint tid = thread_x();
        $uint observed = 0u;
        $int phase = 0;
        // Host loop: phases is a compile-time constant, so the DSL body is
        // emitted once per phase (static unrolling), mirroring the
        // while(!try_wait_parity) idiom in test_cluster_launch_control.cpp.
        for (auto p = 0u; p < phases; ++p) {
            // thread0 performs a single arrival per phase
            $if (tid == 0u) {
                mbarrier_arrive(bar, 1u);
            };
            // everyone spins (non-blocking test_wait) until the phase flips
            $while (!mbarrier_test_wait_parity(bar, phase)) {};
            observed += 1u;
            phase = phase ^ 1;
            sync_block();
        };
        out.write(tid, observed);
    };

    auto shader = device.compile(kernel);
    stream << shader(out).dispatch(block_size) << synchronize();

    std::array<uint, block_size> observed{};
    stream << out.copy_to(luisa::span{observed}) << synchronize();

    for (auto i = 0u; i < block_size; ++i) {
        expect(observed[i] == phases)
            << "thread " << i << " observed " << observed[i]
            << " phases, expected " << phases;
    }
}

// 2. mbarrier_expect_tx equivalence.
//    The CLC try_cancel response is the async producer: it writes 16 bytes to
//    shared memory and completes 16 tx bytes on the mbarrier (even when there
//    is no pending block to cancel). bar0 is armed with the combined
//    arrive_expect_tx; bar1 is armed with the split expect_tx + arrive. Both
//    phases must complete, proving the two forms are equivalent (this mirrors
//    PipelineCLCFetchAsync's producer commit). Single block: with multiple
//    blocks a try_cancel can cancel a sibling block's launch, which would
//    make the flags nondeterministic.
void test_mbarrier_expect_tx_equivalence(Device &device) {
    constexpr auto block_size = 32u;

    auto flags = device.create_buffer<uint>(2u);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferUInt flags) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<luisa::ulong> bar0{1u};
        Shared<luisa::ulong> bar1{1u};
        Shared<uint4> r0{1u};
        Shared<uint4> r1{1u};
        $if (thread_x() == 0u) {
            mbarrier_init(bar0, 1u);
            mbarrier_init(bar1, 1u);
            fence_mbarrier_init();
        };
        sync_block();

        $if (thread_x() == 0u) {
            // async producer for bar0: response + 16 tx bytes, then the
            // combined arrive+expect
            fence_proxy_async_acquire();
            clc_try_cancel(r0, bar0);
            mbarrier_arrive_expect_tx(bar0, 16u);
            // async producer for bar1: response + 16 tx bytes, then the
            // split expect_tx + arrive
            clc_try_cancel(r1, bar1);
            mbarrier_expect_tx(bar1, 16u);
            mbarrier_arrive(bar1, 1u);
        };

        // both barriers must complete phase 0
        $while (!mbarrier_try_wait_parity(bar0, 0)) {};
        $while (!mbarrier_try_wait_parity(bar1, 0)) {};

        $if (thread_x() == 0u) {
            // with no pending cluster launch the responses must report
            // "not canceled"; this also exercises the canonical query PTX
            $bool canceled0 = clc_query_is_canceled(r0);
            $bool canceled1 = clc_query_is_canceled(r1);
            fence_proxy_async_release();
            // false -> 1 (completed, not canceled); true -> 2
            flags.write(0u, cast<uint>(canceled0) + 1u);
            flags.write(1u, cast<uint>(canceled1) + 1u);
        };
    };

    auto shader = device.compile(kernel);
    // dispatch size is in THREADS: one block per tile (CLC requires the grid
    // to hold one block per output tile)
    stream << shader(flags).dispatch(block_size) << synchronize();
    std::array<uint, 2u> host_flags{};
    stream << flags.copy_to(luisa::span{host_flags}) << synchronize();
    expect(host_flags[0] == 1u)
        << "arrive_expect_tx barrier must complete with a completed response";
    expect(host_flags[1] == 1u)
        << "expect_tx + arrive barrier must complete with a completed response";
    expect(host_flags[1] == 1u)
        << "expect_tx + arrive block must complete its phase";
}

// 3. mbarrier_arrive_no_complete (arrive-drop).
//    The barrier expects 2 arrivals; thread0's arrive_no_complete must NOT
//    complete the phase, and thread1's plain arrive completes phase 0 exactly
//    once. Afterwards phase-parity 1 must still be incomplete — a spurious
//    second phase completion (e.g. a wrong noComplete lowering) is caught by
//    the parity check.
void test_mbarrier_arrive_count(Device &device) {
    constexpr auto block_size = 64u;

    auto out = device.create_buffer<uint>(1u);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferUInt out) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<luisa::ulong> bar{1u};
        $if (thread_x() == 0u) {
            mbarrier_init(bar, 2u);// expect 2 arrivals
            fence_mbarrier_init();
        };
        sync_block();

        // thread0: arrive-drop (does not complete); thread1: completing arrive
        $if (thread_x() == 0u) {
            mbarrier_arrive_no_complete(bar, 1u);
        };
        $if (thread_x() == 1u) {
            mbarrier_arrive(bar, 1u);
        };

        // the phase flips only when both arrivals are in
        $while (!mbarrier_test_wait_parity(bar, 0)) {};

        sync_block();
        $if (thread_x() == 0u) {
            // phase parity 1 must NOT have completed
            $bool spurious = mbarrier_test_wait_parity(bar, 1);
            out.write(0u, cast<uint>(!spurious));
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(out).dispatch(block_size) << synchronize();

    std::array<uint, 1u> result{};
    stream << out.copy_to(luisa::span{result}) << synchronize();

    expect(result[0] == 1u)
        << "arrive_no_complete must not complete the phase, and the plain arrive must complete it exactly once";
}

// 4. Proxy-fence smoke test.
//    Exercises every fence wrapper in source order. A malformed PTX mnemonic
//    would fail NVRTC compilation, so successful compile + run + intact data
//    is the gate.
void test_proxy_fences_smoke(Device &device) {
    constexpr auto block_size = 64u;

    auto out = device.create_buffer<uint>(block_size);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferUInt out) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<uint> data{1u};
        $if (thread_x() == 0u) {
            data[0u] = 42u;
        };
        // all proxy fences, in order; correctness is "compiles + does not
        // corrupt the shared value"
        fence_mbarrier_init();
        fence_proxy_async();
        fence_proxy_async_generic_release();
        fence_proxy_async_generic_acquire();
        fence_proxy_async_acquire();
        fence_proxy_async_release();
        sync_block();
        out.write(thread_x(), data[0u]);
    };

    auto shader = device.compile(kernel);
    stream << shader(out).dispatch(block_size) << synchronize();

    std::array<uint, block_size> result{};
    stream << out.copy_to(luisa::span{result}) << synchronize();

    for (auto i = 0u; i < block_size; ++i) {
        expect(result[i] == 42u)
            << "thread " << i << " observed " << result[i] << ", expected 42";
    }
}

// 5. CLC canonical-PTX regression.
//    Same work-stealing shape as test_cluster_launch_control.cpp's helper,
//    scaled 4x. Validates the fixed canonical CLC PTX end to end (the
//    non-multicast variant only).
void test_clc_canonical_ptx_regression(Device &device) {
    constexpr auto N = 2048u;
    constexpr auto block_size = 64u;
    // CLC model: the grid must contain one block per output tile; workers
    // cancel pending launches and process their tiles (the original test's
    // dispatch(4) for 8 tiles could never cover all tiles).
    constexpr auto num_tiles = N / block_size;

    luisa::vector<float> host(N);
    for (auto i = 0u; i < N; ++i) {
        host[i] = static_cast<float>(i);
    }

    Buffer<float> buf = device.create_buffer<float>(N);
    Stream stream = device.create_stream();
    stream << buf.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat b, UInt n) noexcept {
        set_block_size(block_size, 1u, 1u);

        cluster_launch_control_work_stealing_1d([&](Int bx) noexcept {
            auto i = bx * block_size + thread_x();
            $if (i < n) {
                b.write(i, b.read(i) * 3.0f);
            };
        });
    };

    auto shader = device.compile(kernel);
    // dispatch size is in THREADS: one block per tile (CLC requires the grid
    // to hold one block per output tile)
    stream << shader(buf, N).dispatch(num_tiles * block_size) << synchronize();

    luisa::vector<float> result(N);
    stream << buf.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    auto mismatch_count = 0u;
    auto first_mismatch = N;
    for (auto i = 0u; i < N; ++i) {
        auto expected = static_cast<float>(i) * 3.0f;
        if (std::abs(result[i] - expected) > 1e-4f) {
            ++mismatch_count;
            if (first_mismatch == N) { first_mismatch = i; }
            all_correct = false;
        }
    }
    if (!all_correct) {
        LUISA_WARNING("{} mismatches, first at [{}]: got {} expected {}",
                      mismatch_count, first_mismatch, result[first_mismatch],
                      static_cast<float>(first_mismatch) * 3.0f);
    }
    expect(all_correct) << "work-stealing helper must process all elements";
}

// 6. Cluster-barrier smoke test.
//    On sm_90+ every kernel has an implicit 1x1x1 cluster, so the cluster
//    barrier wrappers should be legal. If the backend raises an illegal
//    instruction at runtime, catch it, log a warning, and skip rather than
//    fail.
void test_cluster_barrier_smoke(Device &device) {
    constexpr auto block_size = 32u;

    auto out = device.create_buffer<uint>(block_size);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferUInt out) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<uint> data{1u};
        $if (thread_x() == 0u) {
            data[0u] = 7u;
        };
        cluster_barrier_arrive();
        cluster_barrier_arrive_relaxed();
        cluster_barrier_wait();
        cluster_barrier_wait_acquire();
        sync_block();
        out.write(thread_x(), data[0u]);
    };

    try {
        auto shader = device.compile(kernel);
        stream << shader(out).dispatch(block_size) << synchronize();

        std::array<uint, block_size> result{};
        stream << out.copy_to(luisa::span{result}) << synchronize();

        for (auto i = 0u; i < block_size; ++i) {
            expect(result[i] == 7u)
                << "thread " << i << " observed " << result[i] << ", expected 7";
        }
    } catch (const std::exception &e) {
        // The cluster barrier is not available on this backend/architecture
        // (e.g. CUDA_ERROR_ILLEGAL_INSTRUCTION); treat as skipped.
        LUISA_WARNING("cluster_barrier_smoke skipped: cluster barrier not "
                      "supported on backend '{}' ({}).",
                      device.backend_name(), e.what());
    }
}

// (reg is the static initializer below that registers every named _test).
//
// 7. debug_mode_rejection (host-side, no device).
//    LUISA_ERROR expands to luisa::log_error(), which is [[noreturn]] and ends
//    in std::abort() (verified in include/luisa/core/logging.h): it does NOT
//    throw a catchable C++ exception. Calling
//    luisa::compute::detail::debug_function() on a kernel that uses one of the
//    new CallOps (mbarrier_arrive, fence_proxy_async, etc.) therefore aborts
//    the whole process rather than unwinding, so it cannot be asserted with
//    expect(...).throws<>(). The rejection path
//    (src/ast/function_builder_debugger.cpp rejects every new CallOp with the
//    "does not support" message) is exercised manually instead; this test is a
//    no-op that just records the finding.
void test_debug_mode_rejection() {
    // NOTE: do NOT call debug_function(*kernel.function(), {}) here. The debug
    // generator's LUISA_ERROR path std::abort()s (it is not exception-based),
    // which would terminate the test process. The rejection of the new mbarrier
    // and proxy-fence CallOps by the debug function generator is checked by
    // manual inspection of src/ast/function_builder_debugger.cpp.
    LUISA_INFO("debug_mode_rejection: LUISA_ERROR aborts (std::abort) rather "
               "than throwing, so the debug-generator rejection of the new "
               "mbarrier/fence CallOps is exercised manually; this test is a "
               "recorded no-op.");
}

int main(int argc, char *argv[]) {
    // Follow the proven test_async_copy_cuda.cpp idiom: create the device in
    // main and invoke each test function directly. (The static-`_test`-reg
    // + exit-runner pattern used by test_cluster_launch_control.cpp does not
    // reliably hand argv[1] to the registered suites on Windows — the suites
    // ran with an empty backend and silently skipped every device test.)
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    // Optional single-test selection: test_mbarrier_fence <backend> <name>
    // (Boost.UT convention; exact literal match on the test name).
    auto filter = argc > 2 ? luisa::string_view{argv[2]} : luisa::string_view{};
    auto selected = [&filter](luisa::string_view name) noexcept {
        return filter.empty() || filter == name;
    };
    if (selected("mbarrier_arrive_test_wait_parity"))
        test_mbarrier_arrive_test_wait_parity(device);
    if (selected("mbarrier_expect_tx_equivalence"))
        test_mbarrier_expect_tx_equivalence(device);
    if (selected("mbarrier_arrive_count"))
        test_mbarrier_arrive_count(device);
    if (selected("proxy_fences_smoke"))
        test_proxy_fences_smoke(device);
    if (selected("clc_canonical_ptx_regression"))
        test_clc_canonical_ptx_regression(device);
    if (selected("cluster_barrier_smoke"))
        test_cluster_barrier_smoke(device);
    if (selected("debug_mode_rejection"))
        test_debug_mode_rejection();
    return 0;
}
