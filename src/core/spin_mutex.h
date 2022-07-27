//
// Created by Mike Smith on 2021/3/14.
//

#pragma once

#include <atomic>
#include <thread>

#include <core/intrin.h>

namespace luisa {

class spin_mutex {

private:
    std::atomic_flag _flag{false};// ATOMIC_FLAG_INIT not needed as per C++20

public:
    spin_mutex() noexcept = default;

    void lock() noexcept {
        while (_flag.test_and_set(std::memory_order::acquire)) {// acquire lock
            LUISA_INTRIN_PAUSE();
#ifdef __cpp_lib_atomic_flag_test
            while (_flag.test(std::memory_order::relaxed)) {// test lock
                std::this_thread::yield();
            }
#endif
        }
    }
    bool try_lock() noexcept {
        return !_flag.test_and_set(std::memory_order::acquire);
    }
    void unlock() noexcept {
        _flag.clear(std::memory_order::release);// release lock
    }
};

}// namespace luisa
