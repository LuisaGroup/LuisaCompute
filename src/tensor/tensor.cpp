#include <luisa/tensor/tensor.h>
#include <luisa/core/logging.h>

namespace luisa::compute::tensor {
class JitSession::Impl {
};
thread_local JitSession *_current = nullptr;
JitSession &JitSession::get() noexcept {
    if (_current == nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "No evaluation scope found. "
            "Please make sure you are calling this function inside a kernel.");
    }
    return *_current;
}
}// namespace luisa::compute::tensor

namespace luisa::compute::tensor {
ScalarView DTensor::scalar_view() const noexcept {
    uint64_t buffer_handle, buffer_offset, buffer_total_size;
    buffer_info(buffer_handle, buffer_offset, buffer_total_size);
    return ScalarView{buffer_handle, buffer_offset};
}

DenseVectorView DTensor::dense_vector_view() const noexcept {
    uint64_t buffer_handle, buffer_offset, buffer_total_size;
    buffer_info(buffer_handle, buffer_offset, buffer_total_size);
    DenseVectorView ret;
    ret.buffer_handle = buffer_handle;
    ret.buffer_offset = buffer_offset;

    ret.inc = _dense_vector_view_data.incx;
    ret.n = _shape[0];
    return ret;
}

DenseMatrixView DTensor::dense_matrix_view() const noexcept {
    uint64_t buffer_handle, buffer_offset, buffer_total_size;
    buffer_info(buffer_handle, buffer_offset, buffer_total_size);

    DenseMatrixView ret;
    ret.buffer_handle = buffer_handle;
    ret.buffer_offset = buffer_offset;

    ret.kl = _dense_matrix_view_data._kl;
    ret.ku = _dense_matrix_view_data._ku;

    ret.row = _shape[0];
    ret.column = _shape[1];
    ret.lda = _dense_matrix_view_data._lda;
    ret.shape = _dense_matrix_view_data._shape;

    ret.property = _dense_matrix_view_data._property;
    ret.operation = _dense_matrix_view_data._operation;
    ret.fill_mode = _dense_matrix_view_data._fill_mode;
    ret.diag_type = _dense_matrix_view_data._diag_type;

    return ret;
}
}// namespace luisa::compute::tensor