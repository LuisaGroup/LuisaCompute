#pragma once

#include <cstdint>
#include "sparse_vector_storage_view.h"

namespace luisa::compute::tensor {
class SparseVectorDesc {
public:
    int n;
    int nnz;
};

class SparseVectorView {
public:
    SparseVectorStorageView storage;
    SparseVectorDesc desc;
};
}// namespace luisa::compute::tensor