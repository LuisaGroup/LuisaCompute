//
// Created by Mike Smith on 2021/9/30.
//

#pragma once

#include <network/render_config.h>

namespace luisa::compute {

RenderConfig::RenderConfig(
    uint32_t render_id, std::string_view scene, uint2 resolution, size_t spp, uint2 tile_size, size_t tile_spp, size_t tiles_in_flight) noexcept
    : _resolution{resolution},
      _render_id{render_id},
      _spp{static_cast<uint>(spp)},
      _tile_size{tile_size},
      _tile_spp{static_cast<uint>(tile_spp)},
      _max_tiles_in_flight{static_cast<uint>(tiles_in_flight)},
      _name_length{static_cast<uint>(scene.size())} {
    if (scene.size() > max_name_length) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Scene name too long: {} (length = {} > max = {}).",
            scene, scene.size(), max_name_length);
    }
    std::memcpy(_scene.data(), scene.data(), _name_length);
}

}// namespace luisa::compute
