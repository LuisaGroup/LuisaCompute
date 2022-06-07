//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/llvm/llvm_event.h>

namespace luisa::compute::llvm {

void LLVMEvent::wait() const noexcept {
    if (auto f = future(); f.valid()) [[likely]] {
        f.wait();
    }
}

void LLVMEvent::signal(std::shared_future<void> future) noexcept {
    std::scoped_lock lock{_mutex};
    _future = std::move(future);
}

std::shared_future<void> LLVMEvent::future() const noexcept {
    std::scoped_lock lock{_mutex};
    return _future;
}

}// namespace luisa::compute::llvm
