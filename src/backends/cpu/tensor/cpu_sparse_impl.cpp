#include "cpu_las.h"
#include <luisa/tensor/dtensor.h>
#include <mkl.h>

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cpu_tensor_res.h"
#include "weak_type_ex.h"

namespace luisa::compute::cpu::tensor {
using namespace luisa::compute::tensor;
static sparse_matrix_t sp_mat_desc(const DTensor &A) noexcept { return dynamic_cast<CblasSparseMatrixRes *>(A.backend_tensor_res())->sparse_matrix(); }
static sparse_operation_t sparse_enum_map(MatrixOperation op) noexcept;
static matrix_descr sparse_enum_map(const SparseMatrixView &view) noexcept;

void CpuLAS::sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept {
    LUISA_ERROR_WITH_LOCATION("unspported with mkl");
}
void CpuLAS::gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept {
    LUISA_ERROR_WITH_LOCATION("unspported with mkl");
}
void CpuLAS::scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    LUISA_ERROR_WITH_LOCATION("unspported with mkl");
}
size_t CpuLAS::spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    LUISA_ERROR_WITH_LOCATION("unspported with mkl");
    return size_t();
}
void CpuLAS::spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept {
    LUISA_ERROR_WITH_LOCATION("unspported with mkl");
}
size_t CpuLAS::spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept {
    return 0ul;
}
void CpuLAS::spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto sp_mat_A_ = sp_mat_desc(sp_mat_A);
    auto A_view = sp_mat_A.sparse_matrix_view();
    auto beta_ = beta.scalar_view();
    auto x_ = dn_vec_x.dense_vector_view();
    auto y_ = dn_vec_y.dense_vector_view();
    auto type = sp_mat_A.basic_data_type();

    invoke([=] {
        auto status = sparse_spmv_ex(type, sparse_enum_map(A_view.operation),
                raw<void>(alpha_),
                sp_mat_A_,
                sparse_enum_map(A_view),
                raw<void>(x_),
                raw<void>(beta_),
                raw<void>(y_));
        LUISA_ASSERT(status == SPARSE_STATUS_SUCCESS, "error in spmv");
    });
}
sparse_operation_t sparse_enum_map(MatrixOperation op) noexcept {
    switch (op) {
        case luisa::compute::tensor::MatrixOperation::NONE:
            return sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE;
        case luisa::compute::tensor::MatrixOperation::TRANS:
            return sparse_operation_t::SPARSE_OPERATION_TRANSPOSE;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported matrix operation");
            return sparse_operation_t{};
    }
}
matrix_descr sparse_enum_map(const SparseMatrixView &view) noexcept {
    matrix_descr descr{};
    descr.type = sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
    descr.diag = sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
    return descr;
}

//sparse_fill_mode_t cublas_enum_map(DenseMatrixFillMode op) noexcept {
//    sparse_fill_mode_t ret;
//    switch (op) {
//        case luisa::compute::tensor::DenseMatrixFillMode::UPPER:
//            ret = sparse_fill_mode_t::SPARSE_FILL_MODE_UPPER;
//            break;
//        case luisa::compute::tensor::DenseMatrixFillMode::LOWER:
//            ret = sparse_fill_mode_t::SPARSE_FILL_MODE_LOWER;
//            break;
//        default:
//            LUISA_ERROR_WITH_LOCATION("error matrix fill mode mapping.");
//            break;
//    }
//    return ret;
//}
//
//sparse_diag_type_t cublas_enum_map(DenseMatrixDiagType op) noexcept {
//    sparse_diag_type_t ret;
//    switch (op) {
//        case luisa::compute::tensor::DenseMatrixDiagType::NON_UNIT:
//            ret = sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
//        case luisa::compute::tensor::DenseMatrixDiagType::UNIT:
//            ret = sparse_diag_type_t::SPARSE_DIAG_UNIT;
//        default:
//            LUISA_ERROR_WITH_LOCATION("error matrix diag type mapping.");
//            break;
//    }
//    return ret;
//}
//sparse_matrix_type_t cublas_enum_map(luisa::compute::tensor::DenseMatrixShape op) noexcept {
//    sparse_matrix_type_t ret;
//    switch (op) {
//        case luisa::compute::tensor::DenseMatrixShape::GENERAL:
//            ret = sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
//            break;
//        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR:
//            ret = sparse_matrix_type_t::SPARSE_MATRIX_TYPE_TRIANGULAR;
//            break;
//        default:
//            break;
//    }
//    return ret;
//}
}// namespace luisa::compute::cpu::tensor