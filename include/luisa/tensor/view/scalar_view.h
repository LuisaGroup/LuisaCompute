#pragma once

#include <cstdint>
#include "dense_storage_view.h"

namespace luisa::compute::tensor {
class ScalarDesc {
public:
    int offset;
    bool is_host;// if is_host, the scalar is allocated on host(on heap or on stack), otherwise on device
};

class ScalarView {
public:
    DenseStorageView storage;
    ScalarDesc desc;
};
}// namespace luisa::compute::tensor