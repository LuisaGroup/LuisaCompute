#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Texture : public RWResource {
    uint _dim;
    luisa::uint3 _tile_size;
    size_t _tile_size_bytes;
    PixelFormat _format;
    // Base-level size and mip count, when known.  Extensions that address a
    // subresource (direct storage) validate their requests against them.
    luisa::uint3 _size{};
    uint _mip_levels{0u};

public:
    Texture(uint64_t handle, uint dim, bool simul,
            luisa::uint3 tile_size, PixelFormat format,
            size_t tile_size_bytes = 0u,
            luisa::uint3 size = luisa::uint3{},
            uint mip_levels = 0u)
        : RWResource(handle, Tag::TEXTURE, !simul),
          _dim{dim},
          _tile_size{tile_size},
          _tile_size_bytes{tile_size_bytes},
          _format{format},
          _size{size},
          _mip_levels{mip_levels} {}
    auto dim() const { return _dim; }
    auto format() const { return _format; }
    auto tile_size() const { return _tile_size; }
    auto tile_size_bytes() const { return _tile_size_bytes; }
    /// Base-level size, or `(0,0,0)` when it is not tracked by the caller.
    [[nodiscard]] auto size() const noexcept { return _size; }
    /// Number of mip levels, or 0 when it is not tracked by the caller.
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    static constexpr luisa::string_view validation_res_name{"Texture"};
};
}// namespace lc::validation
