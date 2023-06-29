#pragma once
#include "rw_resource.h"
#include <luisa/vstl/common.h>
namespace lc::validation {
class Stream;
class Event : public RWResource {
public:
    struct Signaled {
        uint64_t event_fence;
        uint64_t stream_fence;
    };
    vstd::unordered_map<Stream *, Signaled> signaled;
    Event(uint64_t handle) : RWResource{handle, Tag::EVENT, false} {}
    void sync(uint64_t fence);
};
}// namespace lc::validation
