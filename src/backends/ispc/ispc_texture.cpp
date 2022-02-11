//
// Created by Mike Smith on 2022/2/11.
//

#include <backends/ispc/ispc_texture.h>

namespace luisa::compute::ispc {

ISPCTexture::ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept {
    // TODO
}

ISPCTexture::Handle ISPCTexture::handle() const noexcept {
    // TODO
    return ISPCTexture::Handle();
}

}// namespace luisa::compute::ispc
