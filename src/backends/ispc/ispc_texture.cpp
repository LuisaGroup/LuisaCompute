//
// Created by Mike Smith on 2022/2/11.
//

#include <backends/ispc/ispc_texture.h>


namespace luisa::compute::ispc {

ISPCTexture::ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept:
    format(format),
    dim(dim),
    lodLevel(mip_levels)
{
    this->size[0] = size.x;
    this->size[1] = size.y;
    this->size[2] = size.z;
    if (dim != 2) LUISA_ERROR_WITH_LOCATION("unsupported dimension");
    if (lodLevel > MAXLOD) LUISA_ERROR_WITH_LOCATION("maximal LoD exceeded");
    // mipmap allocate
    uint pxsize = pixel_format_size(format);
    int offset[MAXLOD+1];
    offset[0] = 0;
    for (int i=0; i<lodLevel; ++i)
    {
        uint w = std::max(size.x >> i, 1u);
        uint h = std::max(size.y >> i, 1u);
        offset[i+1] = offset[i] + w*h * pxsize;
    }
    lods[0] = (void*) new uint8_t[offset[lodLevel]];
    LUISA_WARNING("new float array{}", offset[lodLevel]);
    for (int i=1; i<lodLevel; ++i)
        lods[i] = (void*)( (uint8_t*)lods[0] + offset[i]);
}

ISPCTexture::Handle ISPCTexture::handle() const noexcept {
    return {(void*)this};
}

ISPCTexture::~ISPCTexture(){
    delete[] (uint8_t*)lods[0];
}

}// namespace luisa::compute::ispc
