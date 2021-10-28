//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/device.h>
#include <runtime/heap.h>

namespace luisa::compute {

Heap Device::create_heap(size_t size) noexcept {
    return _create<Heap>(size);
}

Heap::Heap(Device::Interface *device, size_t capacity) noexcept
    : Resource{device, Tag::HEAP, device->create_heap(capacity)},
      _capacity{capacity},
      _texture_slots(slot_count, invalid_handle),
      _buffer_slots(slot_count, invalid_handle) {}

void Heap::destroy_texture(uint32_t index) noexcept {
    if (auto &&h = _texture_slots[index]; h == invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Destroying already destroyed heap texture at slot {} in heap #{}.",
            index, handle());
    } else {
        device()->destroy_texture(h);
        h = invalid_handle;
    }
}

void Heap::destroy_buffer(uint32_t index) noexcept {
    if (auto &&h = _buffer_slots[index]; h == invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Destroying already destroyed heap buffer at slot {} in heap #{}.",
            index, handle());
    } else {
        device()->destroy_buffer(h);
        h = invalid_handle;
    }
}

}// namespace luisa::compute
