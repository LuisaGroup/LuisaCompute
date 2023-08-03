#pragma once

#include <cstdint>
#include "sparse_vector_storage_view.h"

namespace luisa::compute::tensor {
class SparseVectorDesc {
public:
    int nnz;
};

class SparseVectorView {
public:
    int n; // logical size
    SparseVectorStorageView storage;
    SparseVectorDesc desc;
};
}// namespace luisa::compute::tensor