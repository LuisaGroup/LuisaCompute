#include "fallback_command_queue.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>

namespace {
thread_local bool pause_next_wait = false;
std::atomic<pthread_cond_t *> worker_condition{nullptr};
std::atomic<pthread_cond_t *> paused_condition{nullptr};
std::atomic_bool enable_pause{false};
std::atomic_bool release_wait{false};
}// namespace

// Linux-only schedule control in this test executable, never in the backend.
// Pause a real worker after its predicate was false but before wait atomically
// unlocks the mutex. Publishing shutdown cannot cross this gap if it uses the
// same mutex. Interposition changes scheduling, not condition-variable rules.
extern "C" __attribute__((visibility("default"))) int pthread_cond_wait(pthread_cond_t *condition, pthread_mutex_t *mutex) {
    using Wait = int (*)(pthread_cond_t *, pthread_mutex_t *);
    static auto wait = reinterpret_cast<Wait>(dlsym(RTLD_NEXT, "pthread_cond_wait"));
    if (pause_next_wait) {
        worker_condition.store(condition, std::memory_order_release);
        if (enable_pause.load(std::memory_order_acquire)) {
            pause_next_wait = false;
            paused_condition.store(condition, std::memory_order_release);
            while (!release_wait.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            paused_condition.store(nullptr, std::memory_order_release);
        }
    }
    return wait(condition, mutex);
}

extern "C" __attribute__((visibility("default"))) int pthread_cond_broadcast(pthread_cond_t *condition) noexcept {
    using Broadcast = int (*)(pthread_cond_t *);
    static auto broadcast = reinterpret_cast<Broadcast>(dlsym(RTLD_NEXT, "pthread_cond_broadcast"));
    if (paused_condition.load(std::memory_order_acquire) == condition) {
        std::cerr << "Wait predicate was published inside its predicate-to-wait gap\n";
        std::abort();
    }
    return broadcast(condition);
}
#endif

using luisa::compute::fallback::FallbackCommandQueue;

int main(int argc, char **argv) {
#if defined(__linux__)
    if (argc == 2 && std::string_view{argv[1]} == "completion") {
        FallbackCommandQueue queue{8u, 2u};
        std::atomic_uint started{0u};
        std::atomic_bool release_work{false};
        queue.enqueue([] { pause_next_wait = true; });// Arm the dispatcher.
        queue.enqueue_parallel(2u, [&](auto item) noexcept {
            started.fetch_add(1u, std::memory_order_release);
            if (item == 0u) {
                while (started.load(std::memory_order_acquire) != 2u) { std::this_thread::yield(); }
            } else {
                while (!release_work.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            }
        });
        while (started.load(std::memory_order_acquire) != 2u) { std::this_thread::yield(); }
        enable_pause.store(true, std::memory_order_release);
        using Broadcast = int (*)(pthread_cond_t *);
        auto broadcast = reinterpret_cast<Broadcast>(dlsym(RTLD_NEXT, "pthread_cond_broadcast"));
        while (paused_condition.load(std::memory_order_acquire) == nullptr) {
            if (auto condition = worker_condition.load(std::memory_order_acquire)) { broadcast(condition); }
            std::this_thread::yield();
        }
        release_work.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        release_wait.store(true, std::memory_order_release);
        queue.synchronize();
    } else {
        auto queue = std::make_unique<FallbackCommandQueue>(8u, 1u);
        queue->enqueue_parallel(1u, [](auto) noexcept { pause_next_wait = true; });
        queue->synchronize();
        enable_pause.store(true, std::memory_order_release);
        using Broadcast = int (*)(pthread_cond_t *);
        auto broadcast = reinterpret_cast<Broadcast>(dlsym(RTLD_NEXT, "pthread_cond_broadcast"));
        while (paused_condition.load(std::memory_order_acquire) == nullptr) {
            if (auto condition = worker_condition.load(std::memory_order_acquire)) {
                // A permitted spurious wakeup, after parallel_for has returned.
                broadcast(condition);
            }
            std::this_thread::yield();
        }
        std::atomic_bool destroying{false};
        auto destroyer = std::thread{[&] {
            destroying.store(true, std::memory_order_release);
            queue.reset();
        }};
        while (!destroying.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        release_wait.store(true, std::memory_order_release);
        destroyer.join();
    }
#else
    (void)argc;
    (void)argv;
#endif
    // Empty/short dispatches race the submitter's completion wait; immediate
    // destruction races workers returning to their idle wait. No device,
    // generated kernel, external event, or timeout is part of this protocol.
    for (auto workers : {1u, 2u, 4u}) {
        for (auto lifetime = 0u; lifetime < 256u; lifetime++) {
            std::array<std::atomic_uint32_t, 17u> visits{};
            {
                FallbackCommandQueue queue{8u, workers};
                for (auto epoch = 0u; epoch < 32u; epoch++) {
                    queue.enqueue_parallel(0u, [](auto) noexcept { std::abort(); });
                    queue.enqueue_parallel(visits.size(), [&](auto i) noexcept {
                        visits[i].fetch_add(1u, std::memory_order_relaxed);
                    });
                }
                queue.synchronize();
            }
            for (auto &visit : visits) {
                if (visit.load(std::memory_order_relaxed) != 32u) {
                    std::cerr << "Fallback queue lost or repeated a work item\n";
                    return EXIT_FAILURE;
                }
            }
        }
    }
    std::cout << "Fallback queue completion and shutdown regression passed\n";
}
