//
// Created by Mike Smith on 2021/3/14.
//

#pragma once

#include <atomic>

namespace luisa {

class spin_mutex {

private:
    std::atomic_flag _flag;

public:
    void lock() noexcept {
        while (_flag.test_and_set(std::memory_order::acquire)) {// acquire lock
            while (_flag.test(std::memory_order::relaxed)) {    // test lock
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
                __asm__ __volatile__("isb\n");
#endif
            }
        }
    }

    void unlock() noexcept {
        _flag.clear(std::memory_order_release);// release lock
    }
};

}// namespace luisa
