//
// Created by Mike Smith on 2021/12/23.
//

#pragma once

#include <mutex>
#include <future>
#include <thread>
#include <memory>
#include <concepts>
#include <functional>
#include <condition_variable>

#include <core/stl.h>
#include <core/basic_types.h>

namespace luisa {

class Barrier;

class ThreadPool {

private:
    luisa::vector<std::thread> _threads;
    luisa::queue<luisa::function<void()>> _tasks;
    std::mutex _mutex;
    luisa::unique_ptr<Barrier> _synchronize_barrier;
    luisa::unique_ptr<Barrier> _dispatch_barrier;
    std::condition_variable _cv;
    bool _should_stop;

private:
    void _dispatch(luisa::function<void()> task) noexcept;
    void _dispatch_all(luisa::function<void()> task, size_t max_threads = std::numeric_limits<size_t>::max()) noexcept;

public:
    explicit ThreadPool(size_t num_threads = 0u) noexcept;
    ~ThreadPool() noexcept;
    ThreadPool(ThreadPool &&) noexcept = delete;
    ThreadPool(const ThreadPool &) noexcept = delete;
    ThreadPool &operator=(ThreadPool &&) noexcept = delete;
    ThreadPool &operator=(const ThreadPool &) noexcept = delete;
    [[nodiscard]] static ThreadPool &global() noexcept;

public:
    void barrier() noexcept;
    void synchronize() noexcept;

    template<typename F>
        requires std::invocable<F>
    auto async(F f) noexcept {
        using R = std::invoke_result_t<F>;
        auto promise = luisa::make_shared<std::promise<R>>(
            std::allocator_arg, luisa::allocator{});
        auto future = promise->get_future().share();
        _dispatch([func = std::move(f), promise = std::move(promise), future]() mutable noexcept {
            if constexpr (std::same_as<R, void>) {
                func();
                promise->set_value();
            } else {
                promise->set_value(func());
            }
        });
        return future;
    }

    template<typename F>
    void parallel(uint n, F f) noexcept {
        if (n > 0u) {
            auto counter = luisa::make_shared<std::atomic_uint>(0u);
            _dispatch_all(
                [=]() mutable noexcept {
                    for (auto i = (*counter)++; i < n; i = (*counter)++) {
                        f(i);
                    }
                },
                n);
        }
    }

    template<typename F>
    void parallel(uint nx, uint ny, F f) noexcept {
        parallel(nx * ny, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % nx, i / nx);
        });
    }

    template<typename F>
    void parallel(uint nx, uint ny, uint nz, F f) noexcept {
        parallel(nx * ny * nz, [=, f = std::move(f)](auto i) mutable noexcept {
            f(i % nx, i / nx % ny, i / nx / ny);
        });
    }
};

}// namespace luisa
