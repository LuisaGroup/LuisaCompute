//
// Created by Mike Smith on 2021/3/14.
//

#pragma once

#include <atomic>

#if defined(__x86_64__)
#include <immintrin.h>
#define LUISA_INTRIN_PAUSE() _mm_pause()
#elif defined(_M_X64)
#include <winnt.h>
#define LUISA_INTRIN_PAUSE() YieldProcessor()
#elif defined(__aarch64__)
#define LUISA_INTRIN_PAUSE() asm volatile("isb")
#else
#define LUISA_INTRIN_PAUSE() []{}()
#endif

namespace luisa {

class spin_mutex {

private:
    std::atomic_flag _flag;// ATOMIC_FLAG_INIT not needed as per C++20

public:
    void lock() noexcept {
        while (_flag.test_and_set(std::memory_order::acquire)) {// acquire lock
#ifdef __cpp_lib_atomic_flag_test
            while (_flag.test(std::memory_order::relaxed)) {// test lock
#endif
                LUISA_INTRIN_PAUSE();
#ifdef __cpp_lib_atomic_flag_test
            }
#endif
        }
    }

    void unlock() noexcept {
        _flag.clear(std::memory_order_release);// release lock
    }
};

}// namespace luisa
