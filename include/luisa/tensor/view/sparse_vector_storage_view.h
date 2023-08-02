#pragma once

#include <luisa/tensor/view/dense_storage_view.h>

namespace luisa::compute::tensor {
class SparseVectorStorageView {
public:
    DenseStorageView values;
    DenseStorageView indices;
};
}// namespace luisa::compute::tensor