#pragma once
#include <luisa/tensor/backend_tensor_res.h>
#include <luisa/tensor/dtensor.h>
#include <luisa/core/logging.h>
#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "weak_type_ex.h"

namespace luisa::compute::cpu::tensor {
class BatchRes {
    vector<void *> _array_of_ptr;
public:
    BatchRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        luisa::vector<DenseStorageView> storage{};
        if (tensor.is_vector())// sparse vector
            storage = tensor.dense_vector_view().storage;
        else if (tensor.is_matrix())// sparse matrix
            storage = tensor.dense_matrix_view().storage;
        // _array_of_ptr.resize(storage.size());
        std::transform(storage.begin(), storage.end(), (void **)tensor.batch_view().storage.buffer_native_handle,
                       [&](const DenseStorageView &d) {
                           return reinterpret_cast<void *>(raw_ptr(d));
                       });
    }
};

class CblasDenseRes : public luisa::compute::tensor::BackendTensorRes {
    BatchRes _batch_res;
public:
    CblasDenseRes(const luisa::compute::tensor::DTensor &tensor) noexcept
        : _batch_res{tensor} {}

    virtual ~CblasDenseRes() noexcept {}
};

class CblasSparseMatrixRes : public luisa::compute::tensor::BackendTensorRes {
    sparse_matrix_t _sparse_matrix = nullptr;
public:
    CblasSparseMatrixRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        auto view = tensor.sparse_matrix_view();
        auto type = tensor.basic_data_type();
        sparse_status_t status;
        switch (view.desc.format) {
            case luisa::compute::tensor::SparseMatrixFormat::COO: {
                status = create_coo_ex(type,
                                       &_sparse_matrix,
                                       view.row, view.col, view.desc.nnz,
                                       raw<MKL_INT>(view.storage.i_data),
                                       raw<MKL_INT>(view.storage.j_data),
                                       raw<void>(view.storage.values));
            } break;
            case luisa::compute::tensor::SparseMatrixFormat::CSR: {
                status = create_csr_ex(type,
                                       &_sparse_matrix,
                                       view.row, view.col,
                                       raw<MKL_INT>(view.storage.i_data),
                                       raw<MKL_INT>(view.storage.j_data),
                                       raw<void>(view.storage.values));

            } break;
            case luisa::compute::tensor::SparseMatrixFormat::CSC: {
                status = create_csc_ex(type,
                                       &_sparse_matrix,
                                       view.row, view.col,
                                       raw<MKL_INT>(view.storage.i_data),
                                       raw<MKL_INT>(view.storage.j_data),
                                       raw<void>(view.storage.values));
            } break;
            default:
                LUISA_ERROR_WITH_LOCATION("unsupported sparse matrix format");
                break;
        }
        LUISA_ASSERT(status == SPARSE_STATUS_SUCCESS, "failed to create sparse matrix");
    }

    virtual ~CblasSparseMatrixRes() noexcept {
        if (_sparse_matrix) mkl_sparse_destroy(_sparse_matrix);
    }

    sparse_matrix_t sparse_matrix() const noexcept { return _sparse_matrix; }
};
}// namespace luisa::compute::cpu::tensor