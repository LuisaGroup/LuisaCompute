//
// Created by Mike Smith on 2021/9/30.
//

#pragma once

#include <array>
#include <string_view>

#include <core/basic_types.h>
#include <core/logging.h>

namespace luisa::compute {

class RenderConfig {

    static constexpr auto max_name_length = 32u;

private:
    uint2 _resolution{};
    uint _render_id{};
    uint _spp{};
    uint2 _tile_size{};
    uint _tile_spp{};
    uint _max_tiles_in_flight{};
    uint _name_length{};
    std::array<char, max_name_length> _scene{};
    // TODO: camera & other stuff

public:
    RenderConfig(uint32_t render_id, std::string_view scene, uint2 resolution, size_t spp, uint2 tile_size, size_t tile_spp, size_t tiles_in_flight) noexcept;
    [[nodiscard]] auto scene() const noexcept { return std::string_view{_scene.data(), _name_length}; }
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] auto spp() const noexcept { return _spp; }
    [[nodiscard]] auto tile_size() const noexcept { return _tile_size; }
    [[nodiscard]] auto tile_spp() const noexcept { return _tile_spp; }
    [[nodiscard]] auto tiles_in_flight() const noexcept { return _max_tiles_in_flight; }
    [[nodiscard]] auto render_id() const noexcept { return _render_id; }

    [[nodiscard]] auto operator==(const RenderConfig &rhs) const noexcept {
        return _render_id == rhs._render_id;
    }
};

}// namespace luisa::compute
