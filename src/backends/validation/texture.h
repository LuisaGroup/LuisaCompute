#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Texture : public RWResource {
    uint _dim;
    luisa::uint3 _tile_size;
    size_t _tile_size_bytes;
    PixelFormat _format;

public:
    Texture(uint64_t handle, uint dim, bool simul,
            luisa::uint3 tile_size, PixelFormat format,
            size_t tile_size_bytes = 0u)
        : RWResource(handle, Tag::TEXTURE, !simul),
          _dim{dim},
          _tile_size{tile_size},
          _tile_size_bytes{tile_size_bytes},
          _format{format} {}
    auto dim() const { return _dim; }
    auto format() const { return _format; }
    auto tile_size() const { return _tile_size; }
    auto tile_size_bytes() const { return _tile_size_bytes; }
    static constexpr luisa::string_view validation_res_name{"Texture"};
};
}// namespace lc::validation
