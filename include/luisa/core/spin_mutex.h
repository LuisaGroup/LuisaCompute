#pragma once

#include <atomic>
#include <thread>

#include <luisa/core/intrin.h>

namespace luisa {

class spin_mutex {

private:
#if __cplusplus < 202002L
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;
#else
    std::atomic_flag _flag;// ATOMIC_FLAG_INIT is not needed as per C++20
#endif

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

