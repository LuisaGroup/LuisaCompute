//
// Created by Mike Smith on 2021/4/13.
//

#include <runtime/event.h>
#include <runtime/stream.h>
#include <runtime/texture_heap.h>
#include <runtime/device.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return create<Stream>();
}

Event Device::create_event() noexcept {
    return create<Event>();
}

TextureHeap Device::create_texture_heap(size_t size) noexcept {
    return create<TextureHeap>(size);
}

}
