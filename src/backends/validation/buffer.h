#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer : public RWResource {
    uint64_t _tile_size;

public:
    Buffer(uint64_t handle, uint64_t tile_size) : RWResource(handle, Tag::BUFFER, false), _tile_size{tile_size} {}
    auto tile_size() const { return _tile_size; }
};
}// namespace lc::validation
