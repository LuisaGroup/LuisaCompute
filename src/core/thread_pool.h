//
// Created by Mike Smith on 2021/12/23.
//

#pragma once

#include <mutex>
#include <future>
#include <concepts>

#include <core/stl/functional.h>
#include <core/stl/memory.h>
#include <core/basic_types.h>
#include <core/shared_function.h>

namespace luisa {

/// Thread pool class
class LC_CORE_API ThreadPool {

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;
    std::atomic_uint _task_count;

private:
    void _dispatch(luisa::SharedFunction<void()> &&task) noexcept;
    void _dispatch_all(luisa::SharedFunction<void()> &&task, size_t max_threads = std::numeric_limits<size_t>::max()) noexcept;

public:
    /// Create a thread pool with num_threads threads
    explicit ThreadPool(size_t num_threads = 0u) noexcept;
    ~ThreadPool() noexcept;
    ThreadPool(ThreadPool &&) noexcept = delete;
    ThreadPool(const ThreadPool &) noexcept = delete;
    ThreadPool &operator=(ThreadPool &&) noexcept = delete;
    ThreadPool &operator=(const ThreadPool &) noexcept = delete;
    /// Return global static ThreadPool instance
    [[nodiscard]] static uint worker_thread_index() noexcept;

public:
    /// Barrier all threads
    void barrier() noexcept;
    /// Synchronize all threads
    void synchronize() noexcept;
    /// Return size of threads
    [[nodiscard]] uint size() const noexcept;
    /// Return count of tasks
    [[nodiscard]] uint task_count() const noexcept { return _task_count.load(); }

    /// Run a function async and return future of return value
    template<typename F>
        requires std::is_invocable_v<F>
    // std::shared_future<R>
    auto async(F &&f) noexcept {
        using R = std::invoke_result_t<F>;
        auto promise = luisa::make_unique<std::promise<R>>(
            std::allocator_arg, luisa::allocator{});
        auto future = promise->get_future().share();
        _task_count.fetch_add(1u);
        _dispatch([promise = std::move(promise), future, f = std::move(f), this]() mutable noexcept {
            if constexpr (std::same_as<R, void>) {
                f();
                promise->set_value();
            } else {
                promise->set_value(f());
            }
            _task_count.fetch_sub(1u);
        });
        return future;
    }

    /// Run a function parallel
    template<typename F>
        requires std::is_invocable_v<F, uint>
    void parallel(uint n, F &&f) noexcept {
        if (n > 0u) {
            _task_count.fetch_add(1u);
            auto counter = luisa::make_unique<std::atomic_uint>(0u);
            _dispatch_all(
                [=, this]() mutable noexcept {
                    auto i = 0u;
                    while ((i = counter->fetch_add(1u)) < n) { f(i); }
                    if (i == n) { _task_count.fetch_sub(1u); }
                },
                n);
        }
    }

    /// Run a function 2D parallel
    template<typename F>
        requires std::is_invocable_v<F, uint, uint>
    void parallel(uint nx, uint ny, F &&f) noexcept {
        parallel(nx * ny, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % nx, i / nx);
        });
    }

    /// Run a function 3D parallel
    template<typename F>
        requires std::is_invocable_v<F, uint, uint, uint>
    void parallel(uint nx, uint ny, uint nz, F &&f) noexcept {
        parallel(nx * ny * nz, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % nx, i / nx % ny, i / nx / ny);
        });
    }
};

}// namespace luisa
