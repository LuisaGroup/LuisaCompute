#pragma once
#include "resource.h"
#include <vstl/common.h>
namespace lc::validation {
class Stream;
class Event : public Resource {
public:
    vstd::unordered_map<Stream *, uint64_t> signaled;
    Event(uint64_t handle) : Resource{handle, Tag::EVENT} {}
    void sync();
};
}// namespace lc::validation