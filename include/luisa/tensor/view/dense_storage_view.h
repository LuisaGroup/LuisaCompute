#pragma once

#include <cstdint>

namespace luisa::compute::tensor {
class DenseStorageView {
public:
    uint64_t buffer_handle;
    void* buffer_native_handle;
    uint64_t buffer_stride;
    uint64_t buffer_offset;
    uint64_t buffer_total_size;
};
}// namespace luisa::compute::tensor