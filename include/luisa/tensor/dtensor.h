#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>
#include "view.h"

namespace luisa::compute::tensor {
enum class TensorType {
    NONE = -1,
    SCALAR = 0,
    VECTOR = 1,
    MATRIX = 2
};

enum class TensorBasicDataType {
    NONE = 0,
    INT32 = 1,
    INT64 = 2,
    FLOAT32 = 3,
    FLOAT64 = 4
};

class LC_TENSOR_API DTensor {
public:
    DTensor(TensorBasicDataType basic_data_type) noexcept : _basic_data_type{basic_data_type} {}
    virtual ~DTensor() noexcept {}
    // copy ctor
    DTensor(const DTensor &other) noexcept;
    // move ctor
    DTensor(DTensor &&other) noexcept;

    TensorType type() const noexcept {
        if (_shape.size() == 0) return TensorType::SCALAR;
        if (_shape.size() == 1) return TensorType::VECTOR;
        if (_shape.size() == 2) return TensorType::MATRIX;
    }

    bool is_batched() const noexcept { return _batch_desc != nullptr; }
    bool is_stride_batched() const noexcept { return is_batched() && _batch_desc->_batch_stride > 0; }
    bool is_sparse() const noexcept { return _sparse_vector_desc || _sparse_matrix_desc; }
    bool is_dense() const noexcept { return _scalar_desc || _dense_vector_desc || _dense_matrix_desc; }
    bool is_scalar() const noexcept { return _shape.size() == 0; }
    bool is_vector() const noexcept { return _shape.size() == 1; }
    bool is_matrix() const noexcept { return _shape.size() == 2; }

    TensorBasicDataType basic_data_type() const noexcept { return _basic_data_type; }

    ScalarView scalar_view() const noexcept;
    DenseVectorView dense_vector_view() const noexcept;
    DenseMatrixView dense_matrix_view() const noexcept;
    SparseVectorView sparse_vector_view() const noexcept;
    SparseMatrixView sparse_matrix_view() const noexcept;

    BatchView batch_view() const noexcept;

    virtual BackendTensorRes *backend_tensor_res() const noexcept = 0;

protected:
    virtual vector<DenseStorageView> dense_storage_view() const noexcept = 0;
    virtual SparseVectorStorageView sparse_vector_storage_view() const noexcept = 0;
    virtual BasicSparseMatrixStorageView basic_sparse_matrix_storage_view() const noexcept = 0;
    virtual BatchStorageView batch_storage_view() const noexcept = 0;

    template<typename T>
    using U = luisa::unique_ptr<T>;
    template<typename T>
    using S = luisa::shared_ptr<T>;

    // basic info:
    TensorBasicDataType _basic_data_type = TensorBasicDataType::NONE;// float, double or int ?
    luisa::vector<int> _shape;                                       // 0, [N], [M,N], [L,M,N] ...?

    U<ScalarDesc> _scalar_desc = nullptr;
    U<DenseVectorDesc> _dense_vector_desc = nullptr;
    U<DenseMatrixDesc> _dense_matrix_desc = nullptr;
    U<SparseVectorDesc> _sparse_vector_desc = nullptr;
    U<SparseMatrixDesc> _sparse_matrix_desc = nullptr;

    U<BatchDesc> _batch_desc = nullptr;

    MatrixOperation _matrix_operation = MatrixOperation::NONE;

    void do_transpose() {
        _matrix_operation = _matrix_operation == MatrixOperation::TRANS ? MatrixOperation::NONE : MatrixOperation::TRANS;
    }
};
}// namespace luisa::compute::tensor