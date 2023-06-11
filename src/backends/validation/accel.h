#pragma once
#include "rw_resource.h"
#include <luisa/runtime/rhi/command.h>
namespace lc::validation {
class Mesh;
class Accel : public RWResource {
    vstd::vector<uint64_t> _meshes;
    vstd::unordered_map<uint64_t, uint64_t> _ref_count;

public:
    bool init_build{false};
    Accel(uint64_t handle) : RWResource(handle, Tag::ACCEL, false) {}
    void set(Stream *stream, Usage usage, Range range) override;
    void modify(size_t size, Stream *stream, luisa::span<AccelBuildCommand::Modification const> modifies);
};
}// namespace lc::validation
