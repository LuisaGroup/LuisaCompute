#pragma once
#include "resource.h"
#include <ast/usage.h>
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

public:
    static void set_usage(Stream *stream, RWResource *res, Usage usage);
    virtual void set(Stream *stream, Usage usage) {
        set_usage(stream, this, usage);
    }
    auto non_simultaneous() const { return _non_simultaneous; }
    auto const &info() const { return _info; }
    RWResource(uint64_t handle, Tag tag, bool non_simultaneous);
    virtual ~RWResource();
};
}// namespace lc::validation