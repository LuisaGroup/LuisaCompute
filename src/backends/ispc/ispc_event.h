//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <future>

namespace luisa::compute::ispc {

/**
 * @brief ISPC event
 * 
 */
class ISPCEvent {

private:
    std::mutex _mutex;
    std::shared_future<void> _future;

public:
    /**
     * @brief wait
     * 
     */
    void wait() noexcept;
    /**
     * @brief signal
     * 
     * @param future 
     */
    void signal(std::shared_future<void> future) noexcept;
};

}// namespace luisa::compute::ispc
