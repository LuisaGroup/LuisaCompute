#pragma once

#include <cstdint>

namespace luisa::compute::tensor {
class DenseStorageView {
public:
    uint64_t buffer_handle = 0;
    void* buffer_native_handle = nullptr;
    uint64_t buffer_stride = 0;
    uint64_t buffer_offset = 0;
    uint64_t buffer_total_size = 0;
};
}// namespace luisa::compute::tensor