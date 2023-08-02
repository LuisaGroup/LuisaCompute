#pragma once

#include <luisa/tensor/storage/dense_storage.h>
#include <luisa/tensor/view/sparse_matrix_storage_view.h>

namespace luisa::compute::tensor {
template<typename T>
class BasicSparseMatrixStorage {
public:
    DenseStorage<T> values;
    DenseStorage<int> i_data;
    DenseStorage<int> j_data;
    auto view() const noexcept {
        return BasicSparseMatrixStorageView{
            .values = values.view(),
            .i_data = i_data.view(),
            .j_data = j_data.view()};
    }
};

template<typename T>
class COOMatrixStorage : public BasicSparseMatrixStorage<T> {
public:
    auto &value_data() noexcept { return this->values; }
    const auto &value_data() const noexcept { return this->values; }

    auto &row_ind() noexcept { return this->i_data; }
    const auto &row_ind() const noexcept { return this->i_data; }

    auto &col_ind() noexcept { return this->j_data; }
    const auto &col_ind() const noexcept { return this->j_data; }
};

template<typename T>
class CSRMatrixStorage : public BasicSparseMatrixStorage<T> {
public:
    auto &value_data() noexcept { return this->values; }
    const auto &value_data() const noexcept { return this->values; }

    auto &row_ptr() noexcept { return this->i_data; }
    const auto &row_ptr() const noexcept { return this->i_data; }

    auto &col_ind() noexcept { return this->j_data; }
    const auto &col_ind() const noexcept { return this->j_data; }
};

template<typename T>
class CSCMatrixStorage : public BasicSparseMatrixStorage<T> {
public:
    auto &value_data() noexcept { return this->values; }
    const auto &value_data() const noexcept { return this->values; }

    auto &col_ptr() noexcept { return this->i_data; }
    const auto &col_ptr() const noexcept { return this->i_data; }

    auto &row_ind() noexcept { return this->j_data; }
    const auto &row_ind() const noexcept { return this->j_data; }
};
}// namespace luisa::compute::tensor