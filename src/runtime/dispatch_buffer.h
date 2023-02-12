#pragma once
#include <core/concepts.h>
#include <core/mathematics.h>
#include <runtime/command.h>
#include <runtime/resource.h>
#include <runtime/device_interface.h>
#include <runtime/custom_struct.h>
namespace luisa::compute {
template<>
class Buffer<DispatchArgs> final : public Resource {
private:
    size_t _capacity{};
    size_t _byte_size{};

private:
    friend class Device;
    Buffer(DeviceInterface *device, size_t capacity, BufferCreationInfo info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _capacity{capacity},
          _byte_size{info.total_size_bytes} {}

public:
    Buffer(DeviceInterface *device, size_t capacity) noexcept
        : Buffer{device, capacity, device->create_buffer(Type::of<DispatchArgs>(), capacity)} {
    }
    Buffer() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto byte_size() const noexcept { return _byte_size; }
};
}// namespace luisa::compute