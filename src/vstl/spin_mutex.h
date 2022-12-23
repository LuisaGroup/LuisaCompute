#pragma once
#include <vstl/config.h>
#include <atomic>
#include <core/spin_mutex.h>
namespace vstd {
using spin_mutex = luisa::spin_mutex;
class spin_shared_mutex final {
    spin_mutex writeMtx;
    std::atomic_size_t readCount = 0;

public:
    spin_shared_mutex() noexcept {}
    void lock() noexcept {
        writeMtx.lock();
    }
    void unlock() noexcept {
        writeMtx.unlock();
    }
    void lock_shared() noexcept {
        auto readCount = this->readCount.fetch_add(1, std::memory_order_relaxed);
        if (readCount == 0) {
            writeMtx.lock();
        }
    }
    void unlock_shared() noexcept {
        auto readCount = this->readCount.fetch_sub(1, std::memory_order_relaxed);
        if (readCount == 1) {
            writeMtx.unlock();
        }
    }
};

}// namespace vstd