//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <runtime/pixel.h>

namespace luisa::compute::ispc {


// Make sure layout matches ISPC
class ISPCTexture {

public:
    struct Handle {
        // Note: remember t modify the
        // `generate_ispc_library.py`
        // script as well
        const void *ptr;// TODO
    };
    struct TextureView {
        const void* ptr;
        uint32_t level, dummy;
    };

public:
    ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept;
    [[nodiscard]] Handle handle() const noexcept;

public:
    static const unsigned MAXLOD = 20;
    uint width;
    uint height;
    uint lodLevel;
    float* lods[MAXLOD];

};

}// namespace luisa::compute::ispc
