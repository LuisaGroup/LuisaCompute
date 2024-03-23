#pragma once
#include <luisa/core/dll_export.h>
#include <marl/event.h>
#include <marl/waitgroup.h>
#include <marl/finally.h>
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
    }
    explicit scheduler(uint32_t thread_count) noexcept
        : internal(internal_t::Config().setWorkerThreadCount(static_cast<int>(thread_count))) {
    }
    void bind() noexcept { internal.bind(); }
    void unbind() noexcept { internal.unbind(); }

private:
    internal_t internal;
};
class counter {
public:
    using internal_t = marl::WaitGroup;
    counter(uint32_t init_count = 0) noexcept : internal(init_count) {}
    void wait() const noexcept { internal.wait(); }
    void add(const uint32_t x) const noexcept { internal.add(x); }
    void decrement() const noexcept { internal.done(); }
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
    bool test() const noexcept { return internal.test(); }
    void clear() const noexcept { internal.clear(); }
private:
    event(internal_t &&other) noexcept : internal(std::move(other)) {}
    internal_t internal;
};

inline void *current_fiber() noexcept { return marl::Scheduler::Fiber::current(); }

template<class F>
void schedule(F &&lambda, event *event, const char *name = nullptr) noexcept {
    if (event) {
        marl::schedule([event = *event, lambda = std::forward<F>(lambda)]() mutable noexcept {
            luisa_fiber_defer(event.signal());
            lambda();
        });
    } else {
        marl::schedule([lambda = std::forward<F>(lambda)]() mutable noexcept {
            lambda();
        });
    }
}
}// namespace luisa::fiber