//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <runtime/pixel.h>

namespace luisa::compute::ispc {

/**
 * @brief Texture of ISPC
 * 
 */
class ISPCTexture {

public:
    /**
     * @brief Handle for deivce usage
     * 
     */
    struct Handle {
        // Note: remember to modify the
        // `generate_ispc_library.py`
        // script as well
        const void *ptr;// TODO
    };
    struct TextureView {
        const void* ptr;
        uint32_t level, dummy;
    };

public:
    /**
     * @brief Construct a new ISPCTexture object
     * 
     * @param format pixel format
     * @param dim dimension
     * @param size size
     * @param mip_levels mipmap levels 
     */
    ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept;
    ~ISPCTexture() noexcept;
    /**
     * @brief Return handle for device usage
     * 
     * @return Handle 
     */
    [[nodiscard]] Handle handle() const noexcept;

public:

    PixelStorage storage;
    uint dim;
    uint size[3];

    static const unsigned MAXLOD = 20;
    uint lodLevel;
    void* lods[MAXLOD];

};

}// namespace luisa::compute::ispc
