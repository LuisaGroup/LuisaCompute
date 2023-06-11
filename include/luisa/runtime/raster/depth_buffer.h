#pragma once

#include <luisa/runtime/depth_format.h>
#include <luisa/runtime/image.h>

namespace luisa::compute {
class RasterExt;
class LC_RUNTIME_API DepthBuffer : public Resource {
    friend class ResourceGenerator;

private:
    uint2 _size{};
    RasterExt *_raster_ext{};
    DepthFormat _format{};
    DepthBuffer(const ResourceCreationInfo &create_info,
                RasterExt *raster_ext,
                DeviceInterface *device,
                DepthFormat format, uint2 size) noexcept;

public:
    DepthBuffer(DeviceInterface *device, RasterExt *raster_ext, DepthFormat format, uint2 size) noexcept;
    ~DepthBuffer() noexcept;
    DepthBuffer(DepthBuffer &&) noexcept = default;
    DepthBuffer(DepthBuffer const &) noexcept = delete;
    DepthBuffer &operator=(DepthBuffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    DepthBuffer &operator=(DepthBuffer const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] luisa::unique_ptr<Command> clear(float value) const noexcept;
    // to regular image type, try to write to this image in compute-kernel is illegal.
    [[nodiscard]] ImageView<float> to_img() noexcept;
};

}// namespace luisa::compute

