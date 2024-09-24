#pragma once

#include <marl/event.h>
#include <marl/future.h>
#include <marl/waitgroup.h>
#include <marl/finally.h>
#include <luisa/core/shared_function.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/concepts.h>

namespace marl {

#define LUISA_MARL_CONCAT_(a, b) a##b
#define LUISA_MARL_CONCAT(a, b) LUISA_MARL_CONCAT_(a, b)

// defer() is a macro to defer execution of a statement until the surrounding
// scope is closed and is typically used to perform cleanup logic once a
// function returns.
//
// Note: Unlike golang's defer(), the defer statement is executed when the
// surrounding *scope* is closed, not necessarily the function.
//
// Example usage:
//
//  void sayHelloWorld()
//  {
//      defer(printf("world\n"));
//      printf("hello ");
//  }
//
#define luisa_fiber_defer(...) \
    auto LUISA_MARL_CONCAT(defer_, __LINE__) = marl::make_finally([&]() noexcept { __VA_ARGS__; })
}// namespace marl

namespace luisa::fiber {
template<typename Iter>
void _advance(Iter &&ite, size_t n) {
    if constexpr (luisa::concepts::is_iterator<std::remove_cvref_t<Iter>>) {
        std::advance(ite, n);
    } else {
        ite += n;
    }
}
template<typename Iter>
[[nodiscard]] auto _distance(Iter ite, Iter last) {
    if constexpr (luisa::concepts::is_iterator<std::remove_cvref_t<Iter>>) {
        return std::distance(ite, last);
    } else {
        return last > ite ? last - ite : ite - last;
    }
}
class scheduler {

public:
    using internal_t = marl::Scheduler;
    scheduler() noexcept
        : internal(internal_t::Config::allCores()) {
        internal.bind();
    }
    explicit scheduler(uint32_t thread_count) noexcept
        : internal(internal_t::Config().setWorkerThreadCount(static_cast<int>(thread_count))) {
        internal.bind();
    }
    scheduler(scheduler const &) = delete;
    scheduler(scheduler &&) = delete;
    ~scheduler() noexcept {
        internal.unbind();
    }
private:
    internal_t internal;
};
using counter = marl::WaitGroup;
using lock = marl::lock;
using condition_variable = marl::ConditionVariable;
using mutex = marl::mutex;
struct event {
private:
    marl::Event _evt;
public:
    using Mode = marl::Event::Mode;
    event(Mode mode = Mode::Manual, bool init_signalled = false) noexcept
        : _evt{mode, init_signalled} {}
    void signal() const noexcept {
        _evt.signal();
    }
    // signalled = false;
    void clear() const noexcept {
        _evt.clear();
    }
    // while (!signalled) { __wait(); }
    void wait() const noexcept {
        _evt.wait();
    }
    [[nodiscard]] auto test() const noexcept {
        return _evt.test();
    }
    [[nodiscard]] auto is_signalled() const noexcept {
        return _evt.isSignalled();
    }
};
template<typename T>
using future = marl::Future<T>;

inline uint32_t worker_thread_count() {
    return marl::Scheduler::get()->config().workerThread.count;
}
template<class F>
    requires(std::is_invocable_v<F>)
void schedule(F &&f) noexcept {
    marl::schedule(std::forward<F>(f));
}

template<class F>
    requires(std::is_invocable_v<F>)
[[nodiscard]] auto async(F &&lambda) noexcept {
    using RetType = decltype(lambda());
    if constexpr (std::is_same_v<RetType, void>) {
        event evt;
        marl::schedule([evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
            lambda();
            evt.signal();
        });
        return evt;
    } else {
        future<RetType> evt;
        marl::schedule([evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
            evt.signal(lambda());
        });
        return evt;
    }
}
namespace detail {
template<typename T>
struct NonMovableAtomic {
    std::atomic<T> value;
    NonMovableAtomic() noexcept = default;
    NonMovableAtomic(T &&t) noexcept : value{std::move(t)} {}
    NonMovableAtomic(NonMovableAtomic const &) = delete;
    NonMovableAtomic(NonMovableAtomic &&rhs) noexcept
        : value{rhs.value.load()} {
    }
};
}// namespace detail
template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
[[nodiscard]] auto async_parallel(uint32_t job_count, F &&lambda, uint32_t internal_jobs = 1) noexcept {
    auto thread_count = std::clamp<uint32_t>(job_count / internal_jobs, 1u, worker_thread_count());
    counter evt{thread_count};
    luisa::SharedFunction<void()> func{[counter = detail::NonMovableAtomic<uint32_t>(0), job_count, internal_jobs, evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter.value.fetch_add(internal_jobs)) < job_count) {
            auto end = std::min<uint32_t>(i + internal_jobs, job_count);
            for (uint32_t v = i; v < end; ++v) {
                lambda(v);
            }
        }
        evt.done();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        marl::schedule(func);
    }
    return evt;
}

template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
void parallel(uint32_t job_count, F &&lambda, uint32_t internal_jobs = 1) noexcept {
    auto thread_count = std::clamp<uint32_t>(job_count / internal_jobs, 1u, worker_thread_count());
    if (thread_count > 1) {
        counter evt{thread_count};
        luisa::SharedFunction<void()> func{[counter = detail::NonMovableAtomic<uint32_t>(0), job_count, internal_jobs, evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
            uint32_t i = 0u;
            while ((i = counter.value.fetch_add(internal_jobs)) < job_count) {
                auto end = std::min<uint32_t>(i + internal_jobs, job_count);
                for (uint32_t v = i; v < end; ++v) {
                    lambda(v);
                }
            }
            evt.done();
        }};
        for (uint32_t i = 0; i < thread_count; ++i) {
            marl::schedule(func);
        }
        evt.wait();
    } else {
        for (uint32_t i = 0; i < job_count; ++i) {
            lambda(i);
        }
    }
}

template<class F, class Iter>
    requires(std::is_invocable_v<F, Iter, Iter>)
void parallel(Iter begin, Iter end, size_t batch, F f, size_t inplace_batch_threahold = 1ull) {
    auto n = _distance(begin, end);
    size_t batchCount = (n + batch - 1) / batch;
    if (batchCount < inplace_batch_threahold) {
        for (size_t i = 0; i < batchCount; ++i) {
            auto toAdvance = std::min((size_t)n, batch);
            auto l = begin;
            auto r = begin;
            _advance(r, toAdvance);
            n -= toAdvance;
            f(l, r);
            begin = r;
        }
    } else {
        auto thread_count = std::clamp<size_t>(batchCount, 1u, std::thread::hardware_concurrency());
        counter counter(thread_count);
        luisa::SharedFunction<void()> shared_func{[atomic_fence = detail::NonMovableAtomic<size_t>(0), counter, batchCount, batch, n, begin, f = std::forward<F>(f)]() mutable {
            size_t i = 0ull;
            while ((i = atomic_fence.value.fetch_add(1)) < batchCount) {
                auto begin_idx = i * batch;
                auto end_idx = std::min<size_t>((i + 1) * batch, static_cast<size_t>(n));
                f(begin + begin_idx, begin + end_idx);
            }
            counter.done();
        }};
        for (size_t i = 0; i < thread_count; ++i) {
            marl::schedule(shared_func);
        }
        counter.wait();
    }
}
template<class F, class Iter>
    requires(std::is_invocable_v<F, Iter, Iter>)
auto async_parallel(Iter begin, Iter end, size_t batch, F f, size_t inplace_batch_threahold = 1ull) {
    auto n = _distance(begin, end);
    size_t batchCount = (n + batch - 1) / batch;

    auto thread_count = std::clamp<size_t>(batchCount, 1u, std::thread::hardware_concurrency());
    counter counter(thread_count);
    luisa::SharedFunction<void()> shared_func{[atomic_fence = detail::NonMovableAtomic<size_t>(0), counter, batchCount, batch, n, begin, f = std::forward<F>(f)]() mutable {
        size_t i = 0ull;
        while ((i = atomic_fence.value.fetch_add(1)) < batchCount) {
            auto begin_idx = i * batch;
            auto end_idx = std::min<size_t>((i + 1) * batch, static_cast<size_t>(n));
            f(begin + begin_idx, begin + end_idx);
        }
        counter.done();
    }};
    for (size_t i = 0; i < thread_count; ++i) {
        marl::schedule(shared_func);
    }
    return counter;
}

}// namespace luisa::fiber
