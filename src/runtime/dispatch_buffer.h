#pragma once

#include <runtime/resource.h>
#include <runtime/device_interface.h>
#include <ast/type_registry.h>

namespace luisa::compute {
struct IndirectKernelDispatch {};
class IndirectDispatchBuffer;
}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::IndirectKernelDispatch,
                            "LC_IndirectKernelDispatch")

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::IndirectDispatchBuffer,
                            "buffer<LC_IndirectKernelDispatch>")

namespace luisa::compute {

namespace detail {
class IndirectDispatchBufferExprProxy;
}

class IndirectDispatchBuffer final : public Resource {

private:
    size_t _capacity{};
    size_t _byte_size{};

private:
    friend class Device;
    IndirectDispatchBuffer(DeviceInterface *device, size_t capacity, BufferCreationInfo info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _capacity{capacity},
          _byte_size{info.total_size_bytes} {}

public:
    IndirectDispatchBuffer(DeviceInterface *device, size_t capacity) noexcept
        : IndirectDispatchBuffer{device, capacity,
                                 device->create_buffer(Type::of<IndirectKernelDispatch>(), capacity)} {
    }
    IndirectDispatchBuffer() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto size_bytes() const noexcept { return _byte_size; }

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::IndirectDispatchBufferExprProxy *>(this);
    }
};

}// namespace luisa::compute
