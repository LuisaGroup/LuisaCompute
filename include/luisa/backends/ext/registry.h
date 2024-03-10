#pragma once

#include <cstdint>
#include <luisa/core/stl/string.h>

namespace luisa::compute {

enum struct CustomCommandUUID : uint32_t {

    CUSTOM_DISPATCH_EXT_BEGIN = 0x0000u,
    CUSTOM_DISPATCH = CUSTOM_DISPATCH_EXT_BEGIN,

    RASTER_EXT_BEGIN = 0x0100u,
    RASTER_DRAW_SCENE = RASTER_EXT_BEGIN,
    RASTER_CLEAR_DEPTH,

    DSTORAGE_EXT_BEGIN = 0x0200u,
    DSTORAGE_READ = DSTORAGE_EXT_BEGIN,

    DENOISER_EXT_BEGIN = 0x0300u,
    DENOISER_DENOISE = DENOISER_EXT_BEGIN,

    CUDA_CUSTOM_COMMAND_BEGIN = 0x0400u,
    CUDA_LCUB_COMMAND = CUDA_CUSTOM_COMMAND_BEGIN,

    REGISTERED_END = 0xffffu,
};

}// namespace luisa::compute

namespace luisa {

[[nodiscard]] inline luisa::string to_string(compute::CustomCommandUUID uuid) noexcept {
    switch (uuid) {
        case compute::CustomCommandUUID::CUSTOM_DISPATCH: return "CUSTOM_DISPATCH";
        case compute::CustomCommandUUID::RASTER_DRAW_SCENE: return "RASTER_DRAW_SCENE";
        case compute::CustomCommandUUID::RASTER_CLEAR_DEPTH: return "RASTER_CLEAR_DEPTH";
        case compute::CustomCommandUUID::DSTORAGE_READ: return "DSTORAGE_READ";
        case compute::CustomCommandUUID::DENOISER_DENOISE: return "DENOISER_DENOISE";
        case compute::CustomCommandUUID::CUDA_LCUB_COMMAND: return "CUDA_LCUB_COMMAND";
        default: break;
    }
    return "UNKNOWN";
}

}// namespace luisa
