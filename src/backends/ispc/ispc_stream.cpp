//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/ispc/ispc_event.h>
#include <backends/ispc/ispc_stream.h>

namespace luisa::compute::ispc {

void ISPCStream::dispatch(const CommandList &cmd_list) noexcept {
    // TODO
    _pool.barrier();
}

void ISPCStream::signal(ISPCEvent *event) noexcept {
    event->signal(_pool.async([] {}));
}

void ISPCStream::wait(ISPCEvent *event) noexcept {
    _pool.async([event] { event->wait(); });
    _pool.barrier();
}

}// namespace luisa::compute::ispc
