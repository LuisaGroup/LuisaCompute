#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer;
class Mesh : public RWResource {

public:
    Buffer *vert{};
    Buffer *index{};
    Range vert_range;
    Range index_range;
    bool imported{false};
    Mesh(uint64_t handle)
        : RWResource(handle, Tag::MESH, false) {}
    void set(Stream *stream, Usage usage, Range range) override;
    static constexpr luisa::string_view validation_res_name{"Mesh"};
};
}// namespace lc::validation
