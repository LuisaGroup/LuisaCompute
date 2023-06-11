#include "managed_bindless.h"
#include "py_stream.h"

namespace luisa::compute {

ManagedBindless::ManagedBindless(DeviceInterface *device, size_t slots) noexcept
    : collector(3), array(device, slots) {}

void ManagedBindless::emplace_buffer(size_t index, uint64 handle, size_t offset) noexcept {
    collector.InRef(index, 0, handle);
    array._emplace_buffer_on_update(index, handle, offset);
}

void ManagedBindless::emplace_tex2d(size_t index, uint64 handle, Sampler sampler) noexcept {
    collector.InRef(index, 1, handle);
    array._emplace_tex2d_on_update(index, handle, sampler);
}

void ManagedBindless::emplace_tex3d(size_t index, uint64 handle, Sampler sampler) noexcept {
    collector.InRef(index, 2, handle);
    array._emplace_tex3d_on_update(index, handle, sampler);
}

void ManagedBindless::remove_buffer(size_t index) noexcept {
    collector.InRef(index, 0, invalid_resource_handle);
    array.remove_buffer_on_update(index);
}

void ManagedBindless::remove_tex2d(size_t index) noexcept {
    collector.InRef(index, 1, invalid_resource_handle);
    array.remove_tex2d_on_update(index);
}

void ManagedBindless::remove_tex3d(size_t index) noexcept {
    collector.InRef(index, 2, invalid_resource_handle);
    array.remove_tex3d_on_update(index);
}

void ManagedBindless::Update(PyStream &stream) noexcept {
    stream.add(array.update());
    collector.AfterExecuteStream(stream);
}

}// namespace luisa::compute

