#pragma once

#include <marl/event.h>
#include <marl/future.h>
#include <marl/waitgroup.h>
#include <marl/finally.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/shared_function.h>
#include <luisa/core/stl/functional.h>

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

class LC_CORE_API scheduler {

public:
    using internal_t = marl::Scheduler;
    scheduler() noexcept;
    explicit scheduler(uint32_t thread_count) noexcept;
    scheduler(scheduler const &) = delete;
    scheduler(scheduler &&) = delete;
    ~scheduler() noexcept;
private:
    internal_t internal;
};

class LC_CORE_API counter {
public:
    using internal_t = marl::WaitGroup;
    counter(uint32_t init_count = 0) noexcept;
    void wait() const noexcept;
    void add(const uint32_t x) const noexcept;
    [[nodiscard]] bool decrement() const noexcept;
private:
    counter(internal_t &&other) noexcept;
    internal_t internal;
};

class LC_CORE_API event {
public:
    using internal_t = marl::Event;
    event() noexcept;

    void wait() const noexcept;
    void signal() const noexcept;
    [[nodiscard]] bool test() const noexcept;
    void clear() const noexcept;
private:
    event(internal_t &&other) noexcept;
    internal_t internal;
};
namespace detail {
class LC_CORE_API typeless_future {
public:
    using internal_t = marl::Future;
    typeless_future(size_t mem_size) noexcept;

    [[nodiscard]] void *wait() const noexcept;
    void signal(eastl::move_only_function<void(void *)> const &new_ctor, void (*new_dtor)(void *)) const noexcept;
    [[nodiscard]] bool test() const noexcept;
    void clear() const noexcept;
private:
    internal_t internal;
};
}// namespace detail
template<typename T>
class future {
public:
    future() noexcept : _future{sizeof(T)} {}
    [[nodiscard]] T &wait() noexcept {
        return *static_cast<T *>(_future.wait());
    }
    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    void signal(Args &&...args) {
        _future.signal(
            [&](void *ptr) mutable noexcept {
                new (ptr) T{std::forward<Args>(args)...};
            },
            [](void *ptr) noexcept {
                reinterpret_cast<T *>(ptr)->~T();
            });
    }
    [[nodiscard]] bool test() const noexcept { return _future.test(); }
    void clear() const noexcept { _future.clear(); }
private:
    detail::typeless_future _future;
};
LC_CORE_API void schedule(luisa::function<void()> &&func);
LC_CORE_API uint32_t worker_thread_count();

template<class F>
    requires(std::is_invocable_v<F>)
[[nodiscard]] auto async(F &&lambda) noexcept {
    event evt;
    schedule([evt, lambda = std::forward<F>(lambda)] {
        lambda();
        evt.signal();
    });
    return evt;
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
[[nodiscard]] auto async_parallel(uint32_t job_count, F &&lambda) noexcept {
    auto thread_count = std::min<uint32_t>(job_count, worker_thread_count());
    counter evt{thread_count};
    luisa::SharedFunction<void()> func{[counter = detail::NonMovableAtomic<uint32_t>(0), job_count, evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter.value.fetch_add(1u)) < job_count) { lambda(i); }
        evt.decrement();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        schedule(func);
    }
    return evt;
}

template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
void parallel(uint32_t job_count, F &&lambda) noexcept {
    auto thread_count = std::min<uint32_t>(job_count, worker_thread_count());
    counter evt{thread_count};
    luisa::SharedFunction<void()> func{[counter = detail::NonMovableAtomic<uint32_t>(0), job_count, &evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter.value.fetch_add(1u)) < job_count) { lambda(i); }
        evt.decrement();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        schedule(func);
    }
    evt.wait();
}

}// namespace luisa::fiber
