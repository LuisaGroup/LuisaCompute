#pragma once
#include "rw_resource.h"
namespace lc::validation {
class Buffer : public RWResource {
public:
    Buffer(uint64_t handle) : RWResource(handle, Tag::BUFFER, false) {}
};
}// namespace lc::validation
