#include "cpu_las.h"
#include <mkl.h>

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cpu_tensor_res.h"
#include "week_type_ex.h"

namespace luisa::compute::cpu::tensor {
void CpuLAS::sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}
void CpuLAS::gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}
void CpuLAS::scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}
size_t CpuLAS::spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
    return size_t();
}
void CpuLAS::spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}
size_t CpuLAS::spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
    return size_t();
}
void CpuLAS::spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}
}// namespace luisa::compute::cpu::tensor