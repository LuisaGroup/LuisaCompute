#pragma once

#include <luisa/runtime/depth_format.h>

#include "metal_texture.h"

namespace luisa::compute::metal {

class MetalDepthBuffer final : public MetalTextureBase {

private:
    MTL::Texture *_handle;
    DepthFormat _format;
    uint2 _size;

public:
    MetalDepthBuffer(MTL::Device *device, DepthFormat format, uint width, uint height) noexcept;
    ~MetalDepthBuffer() noexcept override;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::DEPTH; }
    [[nodiscard]] MTL::Texture *handle(uint level = 0u) const noexcept override;
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto pixel_format() const noexcept { return _handle->pixelFormat(); }
    [[nodiscard]] bool has_stencil() const noexcept {
        return _format == DepthFormat::D24S8 ||
               _format == DepthFormat::D32S8A24;
    }
    void set_name(luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::metal
