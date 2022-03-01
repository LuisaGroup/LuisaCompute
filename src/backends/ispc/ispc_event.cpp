//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/ispc/ispc_event.h>

namespace luisa::compute::ispc {

void ISPCEvent::wait() const noexcept {
    if (auto f = future(); f.valid()) [[likely]] {
        f.wait();
    }
}

void ISPCEvent::signal(std::shared_future<void> future) noexcept {
    std::scoped_lock lock{_mutex};
    _future = std::move(future);
}

std::shared_future<void> ISPCEvent::future() const noexcept {
    std::scoped_lock lock{_mutex};
    return _future;
}

}// namespace luisa::compute::ispc
