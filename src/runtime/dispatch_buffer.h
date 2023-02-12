#pragma once
#include <runtime/resource.h>
#include <runtime/device_interface.h>
#include <runtime/custom_struct.h>
namespace luisa::compute {
class DispatchArgsBuffer final : public Resource {
private:
    size_t _capacity{};
    size_t _byte_size{};

private:
    friend class Device;
    DispatchArgsBuffer(DeviceInterface *device, size_t capacity, BufferCreationInfo info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _capacity{capacity},
          _byte_size{info.total_size_bytes} {}

public:
    DispatchArgsBuffer(DeviceInterface *device, size_t capacity) noexcept
        : DispatchArgsBuffer{device, capacity, device->create_buffer(Type::of<DispatchArgs>(), capacity)} {
    }
    DispatchArgsBuffer() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto byte_size() const noexcept { return _byte_size; }
};
namespace detail{
template<>
struct TypeDesc<DispatchArgsBuffer> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "buffer<DispatchArgs>"sv;
    }
};
}
}// namespace luisa::compute