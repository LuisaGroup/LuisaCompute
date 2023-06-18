#pragma once
#include "rw_resource.h"
namespace lc::validation {
class SwapChain : public RWResource {
public:
    SwapChain(uint64_t handle) : RWResource(handle, Tag::SWAP_CHAIN, true) {}
};
}// namespace lc::validation
