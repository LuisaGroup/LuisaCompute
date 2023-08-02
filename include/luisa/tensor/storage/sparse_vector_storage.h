#pragma once

#include <luisa/tensor/storage/dense_storage.h>
#include <luisa/tensor/view/sparse_vector_storage_view.h>

namespace luisa::compute::tensor {
template<typename T>
class SparseVectorStorage {
public:
    DenseStorage<T> values;
    DenseStorage<int> indices;
    SparseVectorStorageView view() const noexcept {
        return SparseVectorStorageView{
            .values = values.view(),
            .indices = indices.view()};
	}
};
}// namespace luisa::compute::tensor