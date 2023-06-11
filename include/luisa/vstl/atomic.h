#pragma once
#include <atomic>
#include <thread>
namespace vstd {
template<typename T>
inline void atomic_max(std::atomic<T> &maximum_value, T const &value) noexcept {
    T prev_value = maximum_value;
    while (
        prev_value < value &&
        !maximum_value.compare_exchange_weak(prev_value, value)) {
        std::this_thread::yield();
    }
}
template<typename T>
inline void atomic_mim(std::atomic<T> &maximum_value, T const &value) noexcept {
    T prev_value = maximum_value;
    while (
        prev_value > value &&
        !maximum_value.compare_exchange_weak(prev_value, value)) {
        std::this_thread::yield();
    }
}
}// namespace vstd
