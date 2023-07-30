#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/backends/ext/raster_cmd.h>

namespace luisa::compute {
DepthBuffer::DepthBuffer(const ResourceCreationInfo &create_info, RasterExt *raster_ext, DeviceInterface *device, DepthFormat format, uint2 size) noexcept
    : Resource(
          device,
          Tag::DEPTH_BUFFER,
          create_info),
      _size(size), _raster_ext{raster_ext}, _format(format) {
}
DepthBuffer Device::create_depth_buffer(DepthFormat depth_format, uint2 size) noexcept {
    return _create<DepthBuffer>(extension<RasterExt>(), depth_format, size);
}

DepthBuffer::DepthBuffer(DeviceInterface *device, RasterExt *raster_ext, DepthFormat format, uint2 size) noexcept
    : Resource(
          device,
          Tag::DEPTH_BUFFER,
          raster_ext->create_depth_buffer(format, size.x, size.y)),
      _size{size}, _raster_ext{raster_ext}, _format{format} {
#ifndef NDEBUG
    if (format == DepthFormat::None) {
        LUISA_ERROR_WITH_LOCATION("Depth format cannot be none!");
    }
#endif
}

luisa::unique_ptr<Command> DepthBuffer::clear(float value) const noexcept {
    return luisa::make_unique<ClearDepthCommand>(handle(), value);
}

DepthBuffer::~DepthBuffer() noexcept {
    if (*this) _raster_ext->destroy_depth_buffer(handle());
}

namespace detail {

PixelStorage depth_to_storage(DepthFormat fmt) noexcept {
    switch (fmt) {
        case DepthFormat::D16: return PixelStorage::SHORT1;
        case DepthFormat::D24S8: [[fallthrough]];
        case DepthFormat::D32: return PixelStorage::FLOAT1;
        case DepthFormat::D32S8A24: return PixelStorage::FLOAT2;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unknown depth format 0x{:02x}.",
                              luisa::to_underlying(fmt));
}

}// namespace detail

ImageView<float> DepthBuffer::to_img() noexcept {
    return {handle(), detail::depth_to_storage(_format), 0u, _size};
}

}// namespace luisa::compute
