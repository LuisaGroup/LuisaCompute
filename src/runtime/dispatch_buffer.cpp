#include <runtime/dispatch_buffer.h>
#include <runtime/device.h>

namespace luisa::compute {

IndirectDispatchBuffer Device::create_indirect_dispatch_buffer(size_t capacity) noexcept {
    return _create<IndirectDispatchBuffer>(capacity);
}

}// namespace luisa::compute
