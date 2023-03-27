#pragma once
#include "resource.h"
#include <vstl/common.h>
namespace lc::validation {
class Stream;
class Event : public Resource {
public:
    vstd::unordered_map<Stream const *, uint64_t> signaled;
    Event(uint64_t handle) : Resource{handle} {}
};
}// namespace lc::validation