/// WavefrontCoroScheduler implementation.
/// The template methods are defined in the header (wavefront.h) since
/// they depend on the coroutine's Args... parameter pack.
/// This translation unit serves as a compilation guard and contains
/// explicit template instantiations for common argument combinations.

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>
#include <atomic>
#include <thread>
#ifndef _WIN32
#include <pthread.h>
#endif

namespace luisa::compute::coro {

void detail::coro_scheduler_compile_shaders(
    luisa::span<luisa::move_only_function<void()>> jobs, uint max_threads) {
    auto thread_count = std::min<size_t>(std::max(max_threads, 1u), jobs.size());
    if (!jobs.empty() && std::getenv("LUISA_CORO_PROFILE_COMPILATION") != nullptr) {
        LUISA_INFO("Coroutine shader compile batch: jobs={} threads={}.",
                   jobs.size(), thread_count);
    }
    if (thread_count <= 1u) {
        for (auto &&job : jobs) { job(); }
        return;
    }
    std::atomic_size_t next{0u};
    auto work = [&] {
        for (;;) {
            auto index = next.fetch_add(1u, std::memory_order_relaxed);
            if (index >= jobs.size()) { break; }
            jobs[index]();
        }
    };
#ifndef _WIN32
    // macOS defaults secondary pthreads to 512 KB stacks, which overflows inside
    // LLVM 23 SelectionDAG/DAGTypeLegalizer when compiling large coroutine shards.
    constexpr size_t worker_stack_bytes = 16u * 1024u * 1024u;
    using WorkFn = decltype(work);
    auto pthread_entry = [](void *param) -> void * {
        (*static_cast<WorkFn *>(param))();
        return nullptr;
    };
    pthread_attr_t attr{};
    const bool attr_ok = ::pthread_attr_init(&attr) == 0 &&
                         ::pthread_attr_setstacksize(&attr, worker_stack_bytes) == 0;
    luisa::vector<pthread_t> pthread_workers;
    luisa::vector<std::thread> fallback_workers;
    pthread_workers.reserve(thread_count - 1u);
    for (size_t i = 1u; i < thread_count; ++i) {
        pthread_t tid{};
        if (attr_ok && ::pthread_create(&tid, &attr, pthread_entry, &work) == 0) {
            pthread_workers.emplace_back(tid);
        } else {
            fallback_workers.emplace_back(work);
        }
    }
    if (attr_ok) { ::pthread_attr_destroy(&attr); }
    work();
    for (auto tid : pthread_workers) { ::pthread_join(tid, nullptr); }
    for (auto &t : fallback_workers) { t.join(); }
#else
    luisa::vector<std::jthread> workers;
    workers.reserve(thread_count - 1u);
    for (size_t i = 1u; i < thread_count; ++i) {
        workers.emplace_back(work);
    }
    work();
    // jthread destruction joins before the caller consumes any shader slot.
#endif
}

// Explicit instantiations for common argument combinations.
template class WavefrontCoroScheduler<Buffer<int>>;
template class WavefrontCoroScheduler<Buffer<uint>>;
template class WavefrontCoroScheduler<int>;
template class WavefrontCoroScheduler<>;

}// namespace luisa::compute::coro
