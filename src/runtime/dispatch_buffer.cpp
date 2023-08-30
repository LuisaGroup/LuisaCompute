#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute {

IndirectDispatchBuffer Device::create_indirect_dispatch_buffer(size_t capacity) noexcept {
    return _create<IndirectDispatchBuffer>(capacity);
}

IndirectDispatchBuffer::~IndirectDispatchBuffer() noexcept {
    if (*this) { device()->destroy_buffer(handle()); }
}

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const IndirectDispatchBuffer &buffer) noexcept {
    buffer._check_is_valid();
    _encoder.encode_buffer(buffer.handle(), 0u, buffer.capacity());
    return *this;
}

}// namespace detail

}// namespace luisa::compute

