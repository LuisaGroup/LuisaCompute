#pragma once
#include "rw_resource.h"
namespace lc::validation {
class SparseHeap final : public RWResource {
    uint64_t _size;

public:
    auto size() const { return _size; }
    SparseHeap(uint64_t handle, uint64_t size, Tag tag)
        : RWResource(handle, tag, false), _size{size} {
        LUISA_ASSERT(
            tag == Tag::SPARSE_BUFFER_HEAP ||
                tag == Tag::SPARSE_TEXTURE_HEAP,
            "Validation sparse heap requires a sparse-heap resource tag.");
    }
    ~SparseHeap();
    static constexpr luisa::string_view validation_res_name{"SparseHeap"};
};
}// namespace lc::validation
