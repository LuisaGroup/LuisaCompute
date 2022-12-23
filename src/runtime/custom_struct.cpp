#include <dsl/dispatch_indirect.h>
#include <runtime/device.h>
#include <runtime/buffer.h>

namespace luisa::compute {
template<size_t i, typename T>
Buffer<T> Device::create_dispatch_buffer(size_t capacity) noexcept {
    Buffer<T> v;
    // Resource
    v._device = _impl;
    auto ptr = _impl.get();
    auto buffer = ptr->create_dispatch_buffer(i, capacity);
    v._handle = buffer.handle;
    v._tag = Resource::Tag::BUFFER;
    // Buffer
    v._size = buffer.size / custom_struct_size;
    return v;
};

Buffer<DispatchArgs1D> Device::create_1d_dispatch_buffer(size_t capacity) noexcept {
    return create_dispatch_buffer<1, DispatchArgs1D>(capacity);
}
Buffer<DispatchArgs2D> Device::create_2d_dispatch_buffer(size_t capacity) noexcept {
    return create_dispatch_buffer<2, DispatchArgs2D>(capacity);
}
Buffer<DispatchArgs3D> Device::create_3d_dispatch_buffer(size_t capacity) noexcept {
    return create_dispatch_buffer<3, DispatchArgs3D>(capacity);
}
Buffer<AABB> Device::create_aabb_buffer(size_t capacity) noexcept {
    Buffer<AABB> v;
    v._device = _impl;
    auto ptr = _impl.get();
    auto buffer = ptr->create_aabb_buffer(capacity);
    v._handle = buffer.handle;
    v._tag = Resource::Tag::BUFFER;
    // Buffer
    v._size = buffer.size / custom_struct_size;
    return v;
}
}// namespace luisa::compute