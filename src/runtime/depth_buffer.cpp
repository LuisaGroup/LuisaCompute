#include <raster/depth_buffer.h>
#include <runtime/device.h>
#include <core/logging.h>
namespace luisa::compute {
DepthBuffer Device::create_depth_buffer(DepthFormat depth_format, uint2 size) noexcept {
    return _create<DepthBuffer>(depth_format, size);
}
DepthBuffer::DepthBuffer(DeviceInterface *device, DepthFormat format, uint2 size) noexcept
    : Resource(
          device,
          Tag::DEPTH_BUFFER,
          device->create_depth_buffer(format, size.x, size.y)),
      _size(size), _format(format) {
#ifndef NDEBUG
    if (format == DepthFormat::None) {
        LUISA_ERROR("Depth format cannot be none!");
    }
#endif
}
luisa::unique_ptr<Command> DepthBuffer::clear(float value) const noexcept {
    return luisa::make_unique<ClearDepthCommand>(handle(), value);
}
namespace detail {
PixelStorage depth_to_storage(DepthFormat fmt) noexcept {
    PixelStorage stg;
    switch (fmt) {
        case DepthFormat::D16:
            stg = PixelStorage::SHORT1;
            break;
        case DepthFormat::D24S8:
        case DepthFormat::D32:
            stg = PixelStorage::FLOAT1;
            break;
        case DepthFormat::D32S8A24:
            stg = PixelStorage::FLOAT2;
            break;
        default:
            stg = PixelStorage::FLOAT1;
            assert(false);
            break;
    }
    return stg;
}
}// namespace detail
ImageView<float> DepthBuffer::to_img() noexcept {
    return {handle(), detail::depth_to_storage(_format), 0u, _size};
}
}// namespace luisa::compute