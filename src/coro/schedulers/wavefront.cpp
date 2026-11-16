/// WavefrontCoroScheduler implementation.
/// The template methods are defined in the header (wavefront.h) since
/// they depend on the coroutine's Args... parameter pack.
/// This translation unit serves as a compilation guard and contains
/// explicit template instantiations for common argument combinations.

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>
#include <atomic>
#include <thread>

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
    luisa::vector<std::jthread> workers;
    workers.reserve(thread_count - 1u);
    for (size_t i = 1u; i < thread_count; ++i) {
        workers.emplace_back(work);
    }
    work();
    // jthread destruction joins before the caller consumes any shader slot.
}

// Explicit instantiations for common argument combinations.
template class WavefrontCoroScheduler<Buffer<int>>;
template class WavefrontCoroScheduler<Buffer<uint>>;
template class WavefrontCoroScheduler<int>;
template class WavefrontCoroScheduler<>;

}// namespace luisa::compute::coro
