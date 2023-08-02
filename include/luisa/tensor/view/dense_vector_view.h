#pragma once
#include <cstdint>
#include "dense_storage_view.h"

namespace luisa::compute::tensor {
class DenseVectorDesc {
public:
    int offset = 0;// start
    int n = 0;     // size
    int inc = 1;   // stride
};

class DenseVectorView {
public:
    DenseStorageView storage;
    DenseVectorDesc desc;
};
}// namespace luisa::compute::tensor