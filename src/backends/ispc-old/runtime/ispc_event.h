#pragma once
#include <mutex>
namespace lc::ispc {
struct Event {
    std::mutex mtx;
    std::condition_variable cv;
    size_t targetFence = 0;
    void Sync();
};
}// namespace lc::ispc