#pragma once

#include <mutex>
#include <luisa/core/stl/functional.h>

namespace luisa {

template<typename Mutex = std::mutex>
class thread_safety {

private:
    mutable Mutex _mutex;

public:
    template<typename F>
    decltype(auto) with_lock(F &&f) const noexcept {
        std::lock_guard lock{_mutex};
        return luisa::invoke(std::forward<F>(f));
    }
};

template<>
class thread_safety<void> {
public:
    template<typename F>
    decltype(auto) with_lock(F &&f) const noexcept {
        return luisa::invoke(std::forward<F>(f));
    }
};

template<bool thread_safe, typename Mutex = std::mutex>
using conditional_mutex = std::conditional<thread_safe, Mutex, void>;

template<bool thread_safe, typename Mutex = std::mutex>
using conditional_mutex_t = typename conditional_mutex<thread_safe, Mutex>::type;

}// namespace luisa

