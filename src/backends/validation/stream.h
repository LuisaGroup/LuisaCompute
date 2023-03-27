#pragma once
#include "resource.h"
#include <ast/usage.h>
#include <vstl/common.h>
namespace lc::validation {
using namespace luisa::compute;
class Event;
class RWResource;
class Stream;
struct CompeteResource {
    Stream const *dst_stream;
    RWResource const *res;
    Usage from;
    Usage to;
};
class Stream : public Resource {
    uint64_t _executed_layer{1};
    uint64_t _synced_layer{0};
    vstd::unordered_map<Stream const *, uint64_t> waited_stream;
    uint64_t stream_synced_frame(Stream const *stream) const;

public:
    vstd::unordered_map<RWResource const*, Usage> res_usages;
    auto executed_layer() const { return _executed_layer; }
    auto synced_layer() const { return _synced_layer; }
    Stream(uint64_t handle);
    void signal(Event *evt);
    void wait(Event *evt);
    void check_compete();
};
}// namespace lc::validation