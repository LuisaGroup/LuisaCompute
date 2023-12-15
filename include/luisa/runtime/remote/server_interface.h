#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/logging.h>
namespace luisa::compute {
class LC_RUNTIME_API ServerInterface {
public:
    explicit ServerInterface(luisa::shared_ptr<DeviceInterface> device_impl) noexcept;
    void execute(luisa::span<const std::byte> data) noexcept;
};
}// namespace luisa::compute