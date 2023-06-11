#pragma once
#include "rw_resource.h"
#include <luisa/vstl/common.h>
namespace lc::validation {
class Stream;
class Event : public RWResource {
public:
    vstd::unordered_map<Stream *, uint64_t> signaled;
    Event(uint64_t handle) : RWResource{handle, Tag::EVENT, false} {}
    void sync();
};
}// namespace lc::validation
