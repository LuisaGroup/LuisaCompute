//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <runtime/pixel.h>

namespace luisa::compute::ispc {

// TODO
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
        // Note: remember t modify the
        // `generate_ispc_library.py`
        // script as well
        const void *ptr;// TODO
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
    /**
     * @brief Return handle for device usage
     * 
     * @return Handle 
     */
    [[nodiscard]] Handle handle() const noexcept;
};

}// namespace luisa::compute::ispc
