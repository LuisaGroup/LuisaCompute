//
// Created by Mike Smith on 2023/4/15.
//

#include <backends/metal/metal_event.h>
#include <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

void MetalStream::signal(MetalEvent *event) noexcept {
    event->signal(_queue);
}

void MetalStream::wait(MetalEvent *event) noexcept {
    event->wait(_queue);
}

}// namespace luisa::compute::metal
