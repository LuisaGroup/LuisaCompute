#pragma once

#include <runtime/depth_format.h>
#include <runtime/image.h>

namespace luisa::compute {

class LC_RUNTIME_API DepthBuffer : public Resource {
    friend class ResourceGenerator;
private:
    uint2 _size{};
    DepthFormat _format{};
    DepthBuffer(const ResourceCreationInfo &create_info, DeviceInterface *device, DepthFormat format, uint2 size) noexcept;

public:
    DepthBuffer(DeviceInterface *device, DepthFormat format, uint2 size) noexcept;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] luisa::unique_ptr<Command> clear(float value) const noexcept;
    [[nodiscard]] ImageView<float> to_img() noexcept;
};

}// namespace luisa::compute
