#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer;
class Mesh : public RWResource {

public:
    Buffer *vert{};
    Buffer *index{};
    Mesh(uint64_t handle)
        : RWResource(handle, Tag::MESH, false) {}
    void set(Stream *stream, Usage usage) override;
};
}// namespace lc::validation