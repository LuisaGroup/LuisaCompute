#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Texture : public RWResource {
    uint _dim;

public:
    Texture(uint64_t handle, uint dim) : RWResource(handle, Tag::TEXTURE, true), _dim{dim} {}
    auto dim() const { return _dim; }
};
}// namespace lc::validation
