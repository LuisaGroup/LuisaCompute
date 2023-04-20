//
// Created by Mike Smith on 2023/4/20.
//

#include <core/logging.h>
#include <backends/metal/metal_buffer.h>

namespace luisa::compute::metal {

metal::MetalBuffer::MetalBuffer(MTL::Device *device, size_t size) noexcept
    : _handle{device->newBuffer(size, MTL::ResourceStorageModePrivate |
                                          MTL::ResourceHazardTrackingModeTracked)} {}

metal::MetalBuffer::~MetalBuffer() {
    _handle->release();
}

MetalBuffer::Binding MetalBuffer::binding(size_t offset) const noexcept {
    auto size = _handle->length();
    LUISA_ASSERT(offset < size, "Offset out of range.");
    return {_handle->gpuAddress() + offset, size - offset};
}

void MetalBuffer::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _handle->setLabel(nullptr);
    } else {
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _handle->setLabel(mtl_name);
        mtl_name->release();
    }
}

}// namespace luisa::compute::metal
