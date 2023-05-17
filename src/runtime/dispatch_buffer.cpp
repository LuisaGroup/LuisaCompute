#include <runtime/dispatch_buffer.h>
#include <runtime/device.h>
#include <runtime/shader.h>

namespace luisa::compute {

IndirectDispatchBuffer Device::create_indirect_dispatch_buffer(size_t capacity) noexcept {
    return _create<IndirectDispatchBuffer>(capacity);
}

IndirectDispatchBuffer::~IndirectDispatchBuffer() noexcept {
    if (*this) { device()->destroy_buffer(handle()); }
}
namespace detail {
ShaderInvokeBase &ShaderInvokeBase::operator<<(const IndirectDispatchBuffer &buffer) noexcept {
    _encoder.encode_buffer(buffer.handle(), 0, buffer.size_bytes());
    return *this;
}
}// namespace detail
}// namespace luisa::compute
