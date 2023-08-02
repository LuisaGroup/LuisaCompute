#pragma once

#include <luisa/tensor/view/dense_storage_view.h>

namespace luisa::compute::tensor {
class BasicSparseMatrixStorageView {
public:
    DenseStorageView values;
    DenseStorageView i_data;
    DenseStorageView j_data;
};
}// namespace luisa::compute::tensor