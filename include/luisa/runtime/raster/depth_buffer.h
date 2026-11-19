#pragma once

#include <luisa/runtime/depth_format.h>
#include <luisa/runtime/image.h>

namespace luisa::compute {
class RasterExt;
class LUISA_RUNTIME_API DepthBuffer : public Resource {
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
    DepthBuffer() noexcept = default;
    DepthBuffer(DeviceInterface *device, RasterExt *raster_ext, DepthFormat format, uint2 size) noexcept;
    ~DepthBuffer() noexcept override { _release(); }
    DepthBuffer(DepthBuffer &&) noexcept = default;
    DepthBuffer(DepthBuffer const &) noexcept = default;
    DepthBuffer &operator=(DepthBuffer &&rhs) noexcept = default;
    DepthBuffer &operator=(DepthBuffer const &rhs) noexcept = default;
    using Resource::operator bool;
    using Resource::release;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] luisa::unique_ptr<Command> clear(float value) const noexcept;
    // to regular image type, try to write to this image in compute-kernel is illegal.
    [[nodiscard]] ImageView<float> to_img() noexcept;

protected:
    void destroy() noexcept override;
};

}// namespace luisa::compute
