#pragma once
#include <runtime/device.h>
namespace luisa::compute {
class DxDevice : public Device::Interface {
public:
    explicit DxDevice(Context ctx) noexcept : Device::Interface{std::move(ctx)} {}
};
}// namespace luisa::compute