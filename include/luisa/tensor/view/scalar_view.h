#pragma once

#include <cstdint>

namespace luisa::compute::tensor {
class ScalarView {
public:
    uint64_t buffer_handle;
    uint64_t buffer_offset;
};
}// namespace luisa::compute::tensor