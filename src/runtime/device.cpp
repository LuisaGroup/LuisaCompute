//
// Created by Mike Smith on 2021/4/13.
//

#include <runtime/event.h>
#include <runtime/stream.h>
#include <runtime/device.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return create<Stream>();
}

Event Device::create_event() noexcept {
    return create<Event>();
}

}
