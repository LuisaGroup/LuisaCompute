#pragma once
#include <stdint.h>
namespace lc::validation {
class Resource {
    uint64_t _handle;

public:
    Resource(uint64_t handle) : _handle{handle} {}
    auto handle() const { return _handle; }
    virtual ~Resource() = default;
};
}// namespace lc::validation