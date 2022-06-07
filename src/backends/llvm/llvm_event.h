//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <future>

namespace luisa::compute::llvm {

class LLVMEvent {

private:
    mutable std::mutex _mutex;
    std::shared_future<void> _future;

public:
    void wait() const noexcept;
    void signal(std::shared_future<void> future) noexcept;
     [[nodiscard]] std::shared_future<void> future() const noexcept;
};

}// namespace luisa::compute::llvm
