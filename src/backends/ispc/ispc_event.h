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
    mutable std::mutex _mutex;
    std::shared_future<void> _future;

public:
    /**
     * @brief wait
     * 
     */
    void wait() const noexcept;
    /**
     * @brief signal
     * 
     * @param future 
     */
    void signal(std::shared_future<void> future) noexcept;
    /**
     * @brief future
     */
     [[nodiscard]] std::shared_future<void> future() const noexcept;
};

}// namespace luisa::compute::ispc
