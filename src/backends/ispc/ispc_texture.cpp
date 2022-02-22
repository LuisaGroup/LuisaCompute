//
// Created by Mike Smith on 2022/2/11.
//

#include <backends/ispc/ispc_texture.h>

namespace luisa::compute::ispc {

ISPCTexture::ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept:
    width(size.x),
    height(size.y),
    lodLevel(mip_levels)
{
    if (format != PixelFormat::RGBA32F) LUISA_ERROR_WITH_LOCATION("unsupported format");
    if (dim != 2) LUISA_ERROR_WITH_LOCATION("unsupported dimension");
    if (lodLevel > MAXLOD) LUISA_ERROR_WITH_LOCATION("maximal LoD exceeded");
    // mipmap allocate
    int offset[MAXLOD+1];
    offset[0] = 0;
    for (int i=1, w=width, h=height; i<=lodLevel; ++i)
    {
        offset[i] = offset[i-1] + w*h*4;
        w = std::max(w/2, 1);
        h = std::max(h/2, 1);
    }
    lods[0] = new float[offset[lodLevel]*4];
    for (int i=1; i<lodLevel; ++i)
        lods[i] = lods[0] + offset[i];
}

ISPCTexture::Handle ISPCTexture::handle() const noexcept {
    return {(void*)this};
}

}// namespace luisa::compute::ispc
