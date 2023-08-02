#pragma once
#include <luisa/tensor/backend_tensor_res.h>
#include <cusparse.h>
#include <luisa/tensor/dtensor.h>
#include <luisa/core/logging.h>
#include "../utils/cusparse_check.h"
#include "enum_map.h"
#include "raw_ptr_cast.h"

namespace luisa::compute::cuda::tensor {
class CusparseDnVecDescRes : public luisa::compute::tensor::BackendTensorRes {
    cusparseDnVecDescr_t _desc_handle = nullptr;
public:
    CusparseDnVecDescRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        LUISA_ASSERT(tensor.basic_data_type() == luisa::compute::tensor::TensorBasicDataType::FLOAT32, "now only float32 is supported");
        auto view = tensor.dense_vector_view();
        cusparseCreateDnVec(&_desc_handle, view.desc.n, raw<float>(view.storage), cuda_enum_map(tensor.basic_data_type()));
    }

    virtual ~CusparseDnVecDescRes() noexcept {
        cusparseDestroyDnVec(_desc_handle);
    }

    auto desc_handle() const noexcept { return _desc_handle; }
};

class CusparseDnMatDescRes : public luisa::compute::tensor::BackendTensorRes {
    cusparseDnMatDescr_t _desc_handle = nullptr;
public:
    CusparseDnMatDescRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        LUISA_ASSERT(tensor.basic_data_type() == luisa::compute::tensor::TensorBasicDataType::FLOAT32, "now only float32 is supported");
        auto view = tensor.dense_matrix_view();
        if (view.desc.shape == luisa::compute::tensor::DenseMatrixShape::GENERAL) {
            cusparseCreateDnMat(&_desc_handle,
                                view.desc.row, view.desc.col, view.desc.lda, raw<float>(view.storage),
                                cuda_enum_map(tensor.basic_data_type()), CUSPARSE_ORDER_COL);
        }
    }

    bool valid() const noexcept { return !_desc_handle; }

    virtual ~CusparseDnMatDescRes() noexcept {
        if (_desc_handle) cusparseDestroyDnMat(_desc_handle);
    }

    auto desc_handle() const noexcept { return _desc_handle; }
};

class CusparseSpMatDescRes : public luisa::compute::tensor::BackendTensorRes {
    cusparseSpMatDescr_t _desc_handle = nullptr;
public:
    CusparseSpMatDescRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        LUISA_ASSERT(tensor.basic_data_type() == luisa::compute::tensor::TensorBasicDataType::FLOAT32, "now only float32 is supported");
        auto view = tensor.sparse_matrix_view();
        switch (view.desc.format) {
            case luisa::compute::tensor::SparseMatrixFormat::COO: {
                LUISA_CHECK_CUSPARSE(cusparseCreateCoo(&_desc_handle,
                                                       view.desc.row, view.desc.col, view.desc.nnz,
                                                       raw<int>(view.storage.i_data), raw<int>(view.storage.j_data), raw<float>(view.storage.values),
                                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                       cuda_enum_map(tensor.basic_data_type()))// now only float32
                );
            } break;
            case luisa::compute::tensor::SparseMatrixFormat::CSR: {
                LUISA_CHECK_CUSPARSE(cusparseCreateCsr(&_desc_handle,
                                                       view.desc.row, view.desc.col, view.desc.nnz,
                                                       raw<int>(view.storage.i_data), raw<int>(view.storage.j_data), raw<float>(view.storage.values),
                                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                       cuda_enum_map(tensor.basic_data_type()))// now only float32
                );
            } break;
            case luisa::compute::tensor::SparseMatrixFormat::CSC: {
                LUISA_CHECK_CUSPARSE(cusparseCreateCsc(&_desc_handle,
                                                       view.desc.row, view.desc.col, view.desc.nnz,
                                                       raw<int>(view.storage.i_data), raw<int>(view.storage.j_data), raw<float>(view.storage.values),
                                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                       cuda_enum_map(tensor.basic_data_type()))// now only float32
                );
            } break;
            default:
                LUISA_ERROR_WITH_LOCATION("unsupported sparse matrix format");
                break;
        }
    }

    virtual ~CusparseSpMatDescRes() noexcept {
        cusparseDestroySpMat(_desc_handle);
    }

    auto desc_handle() const noexcept { return _desc_handle; }
};

class CusparseSpVecDescRes : public luisa::compute::tensor::BackendTensorRes {
    cusparseSpVecDescr_t _desc_handle = nullptr;
public:
    CusparseSpVecDescRes(const luisa::compute::tensor::DTensor &tensor) {
        LUISA_ASSERT(tensor.basic_data_type() == luisa::compute::tensor::TensorBasicDataType::FLOAT32, "now only float32 is supported");
        auto view = tensor.sparse_vector_view();
        cusparseCreateSpVec(&_desc_handle,
                            view.desc.n, view.desc.nnz,
                            raw<int>(view.storage.indices), raw<float>(view.storage.values),
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                            cuda_enum_map(tensor.basic_data_type())// now only float32
        );
    }
    virtual ~CusparseSpVecDescRes() noexcept {
        cusparseDestroySpVec(_desc_handle);
    }
    auto desc_handle() const noexcept { return _desc_handle; }
};
}// namespace luisa::compute::cuda::tensor