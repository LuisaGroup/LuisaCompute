// Test for luisa::spin_mutex.
// Covers: lock, try_lock, unlock, multi-thread contention.

#include "ut/ut.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

void reg_spin_mutex_basic() {

    "spin_mutex_lock_unlock"_test = [] {
        luisa::spin_mutex m;
        m.lock();
        m.unlock();
        // no crash = pass
        expect(true);
    };
}

void reg_spin_mutex_try_lock() {

    "spin_mutex_try_lock"_test = [] {
        luisa::spin_mutex m;

        // try_lock on unlocked mutex should succeed
        bool locked = m.try_lock();
        expect(locked) << "try_lock should succeed on unlocked mutex";

        // try_lock again should fail (mutex is held)
        bool locked_again = m.try_lock();
        expect(!locked_again) << "try_lock should fail on already-locked mutex";

        m.unlock();

        // After unlock, try_lock should succeed again
        bool locked_after = m.try_lock();
        expect(locked_after) << "try_lock should succeed after unlock";
        m.unlock();
    };
}

void reg_spin_mutex_lock_guard() {

    "spin_mutex_with_lock_guard"_test = [] {
        luisa::spin_mutex m;
        {
            std::lock_guard lock{m};
            // try_lock should fail while held by lock_guard
            bool locked = m.try_lock();
            expect(!locked) << "try_lock should fail while lock_guard holds mutex";
        }
        // After lock_guard scope, should be unlocked
        bool locked = m.try_lock();
        expect(locked) << "try_lock should succeed after lock_guard scope ends";
        m.unlock();
    };
}

void reg_spin_mutex_contention() {

    "spin_mutex_multi_thread_contention"_test = [] {
        luisa::spin_mutex m;
        std::atomic<int> counter{0};
        constexpr int num_threads = 8;
        constexpr int increments_per_thread = 10000;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < increments_per_thread; ++i) {
                    std::lock_guard lock{m};
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto &t : threads) {
            t.join();
        }

        int expected = num_threads * increments_per_thread;
        expect(counter.load() == expected)
            << "counter should be " << expected << " but got " << counter.load();
    };
}

void reg_spin_mutex_protected_data() {

    "spin_mutex_data_integrity"_test = [] {
        luisa::spin_mutex m;
        int shared_data = 0;
        constexpr int num_threads = 4;
        constexpr int ops_per_thread = 5000;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < ops_per_thread; ++i) {
                    m.lock();
                    // read-modify-write without atomics
                    int val = shared_data;
                    shared_data = val + 1;
                    m.unlock();
                }
            });
        }
        for (auto &t : threads) {
            t.join();
        }

        int expected = num_threads * ops_per_thread;
        expect(shared_data == expected)
            << "data integrity violated: expected " << expected << " got " << shared_data;
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_spin_mutex_basic();
    reg_spin_mutex_try_lock();
    reg_spin_mutex_lock_guard();
    reg_spin_mutex_contention();
    reg_spin_mutex_protected_data();
    return 0;
}
