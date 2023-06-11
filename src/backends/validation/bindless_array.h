#pragma once
#include "rw_resource.h"
namespace lc::validation {
class BindlessArray : public RWResource {
public:
    BindlessArray(uint64_t handle) : RWResource(handle, Tag::BINDLESS_ARRAY, false) {}
};
}// namespace lc::validation
