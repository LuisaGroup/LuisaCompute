#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include <luisa/core/stl/vector.h>

namespace luisa::compute::simd {

// A device-owned persistent worker pool. Each submission is synchronous, and
// concurrent submitters are serialized so one job cannot overwrite another
// job's pool state.
class SIMDThreadPool {

private:
    using Invoke = void (*)(void *, uint64_t, uint64_t) noexcept;

    struct ParallelFor {
        uint64_t count{0u};
        uint64_t grain_size{1u};
        uint32_t active_workers{0u};
        void *context{nullptr};
        Invoke invoke{nullptr};
    };

private:
    luisa::vector<std::thread> _workers;
    std::mutex _submit_mutex;
    std::mutex _work_mutex;
    std::condition_variable _work_available;
    std::condition_variable _work_done;
    ParallelFor _work{};
    alignas(64) std::atomic_uint64_t _next{0u};
    uint64_t _generation{0u};
    uint32_t _working_workers{0u};
    uint32_t _worker_count{1u};
    bool _stopping{false};

private:
    void _worker_loop(uint32_t worker_index) noexcept;
    void _parallel_for(uint64_t count, uint64_t grain_size,
                       void *context, Invoke invoke) noexcept;

public:
    explicit SIMDThreadPool(uint32_t worker_count) noexcept;
    ~SIMDThreadPool() noexcept;
    SIMDThreadPool(const SIMDThreadPool &) = delete;
    SIMDThreadPool(SIMDThreadPool &&) = delete;
    SIMDThreadPool &operator=(const SIMDThreadPool &) = delete;
    SIMDThreadPool &operator=(SIMDThreadPool &&) = delete;

    [[nodiscard]] uint32_t worker_count() const noexcept {
        return _worker_count;
    }

    template<typename F>
    void parallel_for(uint64_t count, uint64_t grain_size,
                      F &&function) noexcept {
        using Task = std::remove_reference_t<F>;
        static_assert(
            std::is_nothrow_invocable_v<Task &, uint64_t, uint64_t>,
            "SIMD parallel tasks must be noexcept.");
        _parallel_for(
            count, grain_size,
            const_cast<void *>(static_cast<const void *>(
                std::addressof(function))),
            [](void *context, uint64_t begin, uint64_t end) noexcept {
                (*static_cast<Task *>(context))(begin, end);
            });
    }
};

}// namespace luisa::compute::simd
