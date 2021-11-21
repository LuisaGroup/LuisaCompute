//
// Created by Mike Smith on 2021/11/20.
//

#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include <condition_variable>

#include <core/basic_types.h>

namespace luisa::compute {

namespace detail {
class ThreadPoolImpl;
}

class ThreadPool {

public:
    using Task1D = std::function<void(uint32_t)>;
    using Task2D = std::function<void(uint2)>;

private:
    std::unique_ptr<detail::ThreadPoolImpl> _thread_pool;

public:
    explicit ThreadPool(size_t num_workers = 0u) noexcept;
    ~ThreadPool() noexcept;
    void dispatch_1d(uint32_t dispatch_size, uint32_t block_size, Task1D task) noexcept;
    void dispatch_2d(uint2 dispatch_size, uint2 block_size, Task2D task) noexcept;
    void dispatch_1d(uint32_t dispatch_size, Task1D task) noexcept { dispatch_1d(dispatch_size, 256u, std::move(task)); }
    void dispatch_2d(uint2 dispatch_size, Task2D task) noexcept { dispatch_2d(dispatch_size, {16u, 16u}, std::move(task)); }
};

}