#pragma once
#include "resource.h"
#include <ast/usage.h>
#include <vstl/common.h>
namespace lc::validation {
class Stream;
using namespace luisa::compute;
struct RWInfo {
    Usage usage{Usage::NONE};
    uint64_t last_frame{0};
};
class RWResource : public Resource {
    vstd::unordered_map<Stream const *, RWInfo> _info;
    bool _non_simultaneous;
    void _set(Stream const *stream, Usage usage);

public:
    auto non_simultaneous() const { return _non_simultaneous; }
    void read(Stream const *stream);
    void write(Stream const *stream);
    RWResource(uint64_t handle, bool non_simultaneous);
    virtual ~RWResource() = default;
};
}// namespace lc::validation