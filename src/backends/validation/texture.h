#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Texture : public RWResource {
    uint _dim;
    luisa::uint3 _tile_size;
    PixelFormat _format;

public:
    Texture(uint64_t handle, uint dim, bool simul, luisa::uint3 tile_size, PixelFormat format) : RWResource(handle, Tag::TEXTURE, !simul), _dim{dim}, _tile_size{tile_size}, _format{format} {}
    auto dim() const { return _dim; }
    auto format() const { return _format; }
    auto tile_size() const { return _tile_size; }
};
}// namespace lc::validation
