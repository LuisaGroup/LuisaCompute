//
// Created by Mike Smith on 2021/3/14.
//

#pragma once

#include <atomic>
#include <emmintrin.h>
#include <thread>

namespace luisa {

class spin_mutex {

private:
    std::atomic_flag _flag;

public:
    void lock() noexcept {
        while (_flag.test_and_set(std::memory_order_acquire)) {// acquire lock
            while (_flag.test(std::memory_order_relaxed)) { // test lock
                _mm_pause();
            }
        }
    }

    void unlock() noexcept {
        _flag.clear(std::memory_order_release);// release lock
    }
};

}// namespace luisa
