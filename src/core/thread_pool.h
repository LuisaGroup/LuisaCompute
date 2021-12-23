//
// Created by Mike Smith on 2021/12/23.
//

#pragma once

#include <mutex>
#include <future>
#include <thread>
#include <memory>
#include <atomic>
#include <barrier>
#include <concepts>
#include <functional>
#include <condition_variable>

#include <core/allocator.h>
#include <core/basic_types.h>

namespace luisa {

class ThreadPool {

public:
    using barrier_type = std::barrier<decltype([]() noexcept {})>;

private:
    luisa::vector<std::thread> _threads;
    luisa::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    barrier_type _barrier;
    std::condition_variable _cv;
    bool _should_stop;

private:
    void _dispatch(std::function<void()> task) noexcept;
    void _dispatch_all(std::function<void()> task) noexcept;

public:
    explicit ThreadPool(size_t num_threads = 0u) noexcept;
    ~ThreadPool() noexcept;
    ThreadPool(ThreadPool &&) noexcept = delete;
    ThreadPool(const ThreadPool &) noexcept = delete;
    ThreadPool &operator=(ThreadPool &&) noexcept = delete;
    ThreadPool &operator=(const ThreadPool &) noexcept = delete;

public:
    void synchronize() noexcept;
    template<typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto dispatch(F f, Args &&...args) noexcept {
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = luisa::new_with_allocator<std::packaged_task<return_type()>>(
            std::move(f), std::forward<Args>(args)...);
        auto future = task->get_future().share();
        _dispatch([task, future] {
            (*task)();
            luisa::delete_with_allocator(task);
        });
        return future;
    }

    template<typename F>
    void parallel(uint n, F f) noexcept {
        auto counter = luisa::make_shared<std::atomic_uint>(0u);
        _dispatch_all([=]() mutable noexcept {
            for (auto i = counter->fetch_add(1u);
                 i < n;
                 i = counter->fetch_add(1u)) { f(i); }
        });
    }

    template<typename F>
    void parallel(uint2 n, F f) noexcept {
        parallel(n.x * n.y, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % n.x, i / n.x);
        });
    }

    template<typename F>
    void parallel(uint nx, uint ny, F f) noexcept {
        parallel(make_uint2(nx, ny), std::move(f));
    }

    template<typename F>
    void parallel(uint3 n, F f) noexcept {
        parallel(n.x * n.y * n.z, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % n.x, i / n.x % n.y, i / n.x / n.y);
        });
    }

    template<typename F>
    void parallel(uint nx, uint ny, uint nz, F f) noexcept {
        parallel(make_uint3(nx, ny, nz), std::move(f));
    }
};

}// namespace luisa
