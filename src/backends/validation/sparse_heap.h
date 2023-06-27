#pragma once
#include "rw_resource.h"
namespace lc::validation {
class SparseHeap final : public RWResource {
    uint64_t _size;

public:
    auto size() const { return _size; }
    SparseHeap(uint64_t handle, uint64_t size) : RWResource(handle, Tag::SPARSE_BUFFER_HEAP, false), _size{size} {}
    ~SparseHeap();
};
}// namespace lc::validation