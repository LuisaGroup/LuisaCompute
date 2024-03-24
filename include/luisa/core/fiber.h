#pragma once
#include <marl/event.h>
#include <marl/waitgroup.h>
#include <marl/finally.h>
#include <luisa/core/shared_function.h>
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
    scheduler(scheduler const&) = delete;
    scheduler(scheduler &&) = delete;
    ~scheduler() noexcept {
        internal.unbind();
    }
private:
    internal_t internal;
};
class counter {
public:
    using internal_t = marl::WaitGroup;
    counter(uint32_t init_count = 0) noexcept : internal(init_count) {}
    void wait() const noexcept { internal.wait(); }
    void add(const uint32_t x) const noexcept { internal.add(x); }
    auto decrement() const noexcept { return internal.done(); }
private:
    counter(internal_t &&other) noexcept : internal(std::move(other)) {}
    internal_t internal;
};

class event {
public:
    using internal_t = marl::Event;
    event() noexcept : internal(marl::Event::Mode::Manual) {}

    void wait() const noexcept {
        internal.wait();
    }
    void signal() const noexcept { internal.signal(); }
    [[nodiscard]] auto test() const noexcept { return internal.test(); }
    void clear() const noexcept { internal.clear(); }
private:
    event(internal_t &&other) noexcept : internal(std::move(other)) {}
    internal_t internal;
};

inline void *current_fiber() noexcept { return marl::Scheduler::Fiber::current(); }
template<class F>
    requires(std::is_invocable_v<F>)
[[nodiscard]] auto async(F&& lambda) noexcept {
    event evt;
    marl::schedule([evt, lambda = std::forward<F>(lambda)](){
        lambda();
        evt.signal();
    });
    return evt;
}
template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
[[nodiscard]] auto async_parallel(uint32_t job_count, F &&lambda) noexcept {
    auto thread_count = std::min<uint32_t>(job_count, marl::Scheduler::get()->config().workerThread.count);
    counter evt{thread_count};
    auto counter = luisa::make_unique<std::atomic<uint32_t>>(0);
    luisa::SharedFunction<void()> func{[counter = std::move(counter), job_count, evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter->fetch_add(1u)) < job_count) { lambda(i); }
        evt.decrement();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        marl::schedule(func);
    }
    return evt;
}
template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
void parallel(uint32_t job_count, F &&lambda) noexcept {
    auto thread_count = std::min<uint32_t>(job_count, marl::Scheduler::get()->config().workerThread.count);
    counter evt{thread_count};
    auto counter = luisa::make_unique<std::atomic<uint32_t>>(0);
    luisa::SharedFunction<void()> func{[counter = std::move(counter), job_count, &evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter->fetch_add(1u)) < job_count) { lambda(i); }
        evt.decrement();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        marl::schedule(func);
    }
    evt.wait();
}
}// namespace luisa::fiber