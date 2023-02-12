#include <runtime/dispatch_buffer.h>
#include <runtime/device.h>
namespace luisa::compute {
DispatchArgsBuffer Device::create_dispatch_buffer(size_t capacity) noexcept {
    return {_impl.get(), capacity};
}
}// namespace luisa::compute