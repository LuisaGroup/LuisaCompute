// Test for luisa::fiber utilities.
// Covers: scheduler, event, counter, Future<T>, parallel, async, async_parallel.

#include "ut/ut.hpp"

#include <atomic>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// All fiber tests need a scheduler, so we create one for the process lifetime.
static luisa::fiber::scheduler global_scheduler;

// ---- scheduler ----

// ---- event ----

// ---- counter (WaitGroup) ----

// ---- Future<T> ----

// ---- async (void return) ----

// ---- parallel ----

// ---- async_parallel ----

void reg_scheduler_worker_count() {

    "scheduler_worker_count"_test = [] {
        auto count = luisa::fiber::worker_thread_count();
        expect(count > 0u) << "worker_thread_count must be positive";
    };
}

void reg_event_manual() {

    "event_manual_mode"_test = [] {
        luisa::fiber::event evt;
        expect(!evt.test()) << "event should not be signalled initially";
        expect(!evt.is_signalled());

        evt.signal();
        expect(evt.test()) << "event should be signalled after signal()";
        expect(evt.is_signalled());

        // manual mode: test() doesn't clear the signal
        expect(evt.is_signalled()) << "manual event stays signalled after test()";

        evt.clear();
        expect(!evt.is_signalled()) << "event should be cleared after clear()";
    };
}

void reg_event_wait() {

    "event_signal_wait"_test = [] {
        luisa::fiber::event evt;

        // Signal from a scheduled task, wait on main fiber
        luisa::fiber::schedule([evt]() mutable {
            evt.signal();
        });
        evt.wait();
        expect(evt.is_signalled());
    };
}

void reg_event_auto_mode() {

    "event_auto_mode"_test = [] {
        luisa::fiber::event evt{luisa::fiber::event::Mode::Auto, false};
        expect(!evt.test());

        evt.signal();
        // In auto mode, test() returns true and clears the signal
        bool was_signalled = evt.test();
        expect(was_signalled);
        // After auto-clear, should not be signalled
        expect(!evt.is_signalled());
    };
}

void reg_counter_basic() {

    "counter_basic"_test = [] {
        luisa::fiber::counter c{3};

        std::atomic<int> completed{0};
        for (int i = 0; i < 3; ++i) {
            luisa::fiber::schedule([c, &completed]() mutable {
                completed.fetch_add(1, std::memory_order_relaxed);
                c.done();
            });
        }
        c.wait();
        expect(completed.load() == 3_i) << "all 3 tasks should have completed";
    };
}

void reg_future_basic() {

    "future_basic"_test = [] {
        luisa::fiber::Future<int> fut;
        expect(!fut.test()) << "future should not be signalled initially";
        expect(!fut.isSignalled());

        fut.signal(42);
        expect(fut.test());
        expect(fut.isSignalled());

        auto &val = fut.wait();
        expect(val == 42_i);
    };
}

void reg_future_async() {

    "future_from_async"_test = [] {
        auto fut = luisa::fiber::async([]() -> int {
            return 100;
        });
        auto &result = fut.wait();
        expect(result == 100_i);
    };
}

void reg_future_clear() {

    "future_clear"_test = [] {
        luisa::fiber::Future<int> fut;
        fut.signal(10);
        expect(fut.isSignalled());

        fut.clear();
        expect(!fut.isSignalled());
    };
}

void reg_future_signal_overwrite() {

    "future_signal_overwrite"_test = [] {
        luisa::fiber::Future<int> fut;
        fut.signal(1);
        expect(fut.wait() == 1_i);

        // Signal again overwrites the value
        fut.signal(2);
        expect(fut.wait() == 2_i);
    };
}

void reg_async_void() {

    "async_void_return"_test = [] {
        std::atomic<bool> executed{false};
        auto evt = luisa::fiber::async([&executed]() {
            executed.store(true, std::memory_order_release);
        });
        evt.wait();
        expect(executed.load(std::memory_order_acquire)) << "async void lambda should have executed";
    };
}

void reg_parallel_basic() {

    "parallel_basic"_test = [] {
        constexpr uint32_t N = 1000;
        std::atomic<uint32_t> sum{0};
        luisa::fiber::parallel(N, [&sum](uint32_t i) {
            sum.fetch_add(i, std::memory_order_relaxed);
        });
        // sum of 0..999 = 999*1000/2 = 499500
        expect(sum.load() == 499500u);
    };
}

void reg_parallel_single_job() {

    "parallel_single_job"_test = [] {
        std::atomic<int> called{0};
        luisa::fiber::parallel(1u, [&called](uint32_t) {
            called.fetch_add(1, std::memory_order_relaxed);
        });
        expect(called.load() == 1_i);
    };
}

void reg_parallel_zero_jobs() {

    "parallel_zero_jobs"_test = [] {
        std::atomic<int> called{0};
        luisa::fiber::parallel(0u, [&called](uint32_t) {
            called.fetch_add(1, std::memory_order_relaxed);
        });
        expect(called.load() == 0_i) << "zero jobs should call lambda zero times";
    };
}

void reg_parallel_range() {

    "parallel_range_signature"_test = [] {
        // Test the (job_count, lambda(begin, end)) overload
        constexpr uint32_t N = 100;
        std::atomic<uint32_t> total_range{0};
        luisa::fiber::parallel(N, [&total_range](uint32_t begin, uint32_t end) {
            total_range.fetch_add(end - begin, std::memory_order_relaxed);
        });
        expect(total_range.load() == N);
    };
}

void reg_async_parallel() {

    "async_parallel_basic"_test = [] {
        constexpr uint32_t N = 500;
        std::atomic<uint32_t> sum{0};
        auto c = luisa::fiber::async_parallel(N, [&sum](uint32_t i) {
            sum.fetch_add(i, std::memory_order_relaxed);
        });
        c.wait();
        // sum of 0..499 = 499*500/2 = 124750
        expect(sum.load() == 124750u);
    };
}

void reg_async_parallel_with_counter() {

    "async_parallel_with_external_counter"_test = [] {
        constexpr uint32_t N = 200;
        std::atomic<uint32_t> count{0};
        luisa::fiber::counter c{0u};

        luisa::fiber::async_parallel(c, N, [&count](uint32_t) {
            count.fetch_add(1, std::memory_order_relaxed);
        });
        c.wait();
        expect(count.load() == N);
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_scheduler_worker_count();
    reg_event_manual();
    reg_event_wait();
    reg_event_auto_mode();
    reg_counter_basic();
    reg_future_basic();
    reg_future_async();
    reg_future_clear();
    reg_future_signal_overwrite();
    reg_async_void();
    reg_parallel_basic();
    reg_parallel_single_job();
    reg_parallel_zero_jobs();
    reg_parallel_range();
    reg_async_parallel();
    reg_async_parallel_with_counter();
    return 0;
}
