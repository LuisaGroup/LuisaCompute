#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Mesh;
class Accel : public RWResource {
    vstd::vector<Mesh *> _meshes;
    vstd::unordered_map<Mesh *, uint64_t> _ref_count;

public:
    Accel(uint64_t handle) : RWResource(handle, Tag::ACCEL, false) {}
    void set(Stream *stream, Usage usage) override;
};
}// namespace lc::validation