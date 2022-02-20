//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/ispc/ispc_event.h>

namespace luisa::compute::ispc {

void ISPCEvent::wait() noexcept {
    auto future = [this] {
        std::scoped_lock lock{_mutex};
        return _future;
    }();
    if (future.valid()) [[likely]] {
        future.wait();
    }
}

void ISPCEvent::signal(std::shared_future<void> future) noexcept {
    std::scoped_lock lock{_mutex};
    _future = std::move(future);
}

}// namespace luisa::compute::ispc
