//
// Created by Mike Smith on 2022/2/11.
//

#include <core/stl.h>
#include <backends/ispc/ispc_texture.h>

namespace luisa::compute::ispc {

ISPCTexture::ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept : storage(pixel_format_to_storage(format)),
                                                                                               dim(dim),
                                                                                               lodLevel(mip_levels) {
    if (dim == 2) size.z = 1;
    this->size[0] = size.x;
    this->size[1] = size.y;
    this->size[2] = size.z;
    if (dim != 2 && dim != 3) LUISA_ERROR_WITH_LOCATION("Only 2D / 3D texture is supported");
    if (lodLevel > MAXLOD) LUISA_ERROR_WITH_LOCATION("maximal LoD exceeded");
    if (lodLevel == 0) LUISA_ERROR_WITH_LOCATION("LoD must be at least 1");
    // mipmap allocate
    uint pxsize = pixel_storage_size(storage);
    int offset[MAXLOD + 1];
    offset[0] = 0;
    for (int i = 0; i < lodLevel; ++i) {
        uint w = std::max(size.x >> i, 1u);
        uint h = std::max(size.y >> i, 1u);
        uint d = std::max(size.z >> i, 1u);
        offset[i + 1] = offset[i] + w * h * d * pxsize;
    }
    lods[0] = allocate<uint8_t>(offset[lodLevel]);
    LUISA_VERBOSE_WITH_LOCATION(
        "New float array for ISPC with {} bytes.",
        offset[lodLevel]);
    for (int i = 1; i < lodLevel; ++i) {
        lods[i] = (void *)((uint8_t *)lods[0] + offset[i]);
    }
}

ISPCTexture::Handle ISPCTexture::handle() const noexcept {
    return {(void *)this};
}

ISPCTexture::~ISPCTexture() noexcept {
    deallocate(static_cast<uint8_t *>(lods[0]));
}

}// namespace luisa::compute::ispc
