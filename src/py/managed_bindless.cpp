#include <py/managed_bindless.h>
#include <py/py_stream.h>
namespace luisa::compute {
ManagedBindless::ManagedBindless(DeviceInterface *device, uint64 handle) noexcept
    : handle(handle), collector(3), device(device) {
}
ManagedBindless::~ManagedBindless() noexcept {
    device->destroy_bindless_array(handle);
}
void ManagedBindless::emplace_buffer(size_t index, uint64 handle, size_t offset) noexcept {
    collector.InRef(index, 0, handle);
    device->emplace_buffer_in_bindless_array(this->handle, index, handle, offset);
}
void ManagedBindless::emplace_tex2d(size_t index, uint64 handle, Sampler sampler) noexcept {
    collector.InRef(index, 1, handle);
    device->emplace_tex2d_in_bindless_array(this->handle, index, handle, sampler);
}
void ManagedBindless::emplace_tex3d(size_t index, uint64 handle, Sampler sampler) noexcept {
    collector.InRef(index, 2, handle);
    device->emplace_tex3d_in_bindless_array(this->handle, index, handle, sampler);
}
void ManagedBindless::remove_buffer(size_t index) noexcept {
    collector.InRef(index, 0, 0);
    device->remove_buffer_from_bindless_array(handle, index);
}
void ManagedBindless::remove_tex2d(size_t index) noexcept {
    collector.InRef(index, 1, 0);
    device->remove_tex2d_from_bindless_array(handle, index);
}
void ManagedBindless::remove_tex3d(size_t index) noexcept {
    collector.InRef(index, 2, 0);
    device->remove_tex3d_from_bindless_array(handle, index);
}
void ManagedBindless::Update(PyStream &stream) noexcept {
    stream.add(BindlessArrayUpdateCommand::create(handle));
    collector.AfterExecuteStream(stream);
}
}// namespace luisa::compute