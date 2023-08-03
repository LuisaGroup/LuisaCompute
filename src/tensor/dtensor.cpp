#include <luisa/core/logging.h>
#include <luisa/tensor/dtensor.h>
#include <luisa/tensor/las_interface.h>

namespace luisa::compute::tensor {
DTensor::DTensor(const DTensor &other) noexcept {
    _basic_data_type = other._basic_data_type;
    _shape = other._shape;

    if (other._scalar_desc) _scalar_desc = make_unique<ScalarDesc>(*other._scalar_desc);
    if (other._dense_vector_desc) _dense_vector_desc = make_unique<DenseVectorDesc>(*other._dense_vector_desc);
    if (other._dense_matrix_desc) _dense_matrix_desc = make_unique<DenseMatrixDesc>(*other._dense_matrix_desc);
    _matrix_operation = other._matrix_operation;
}

DTensor::DTensor(DTensor &&other) noexcept {
    _basic_data_type = other._basic_data_type;
    _shape = std::move(other._shape);

    _scalar_desc = std::move(other._scalar_desc);
    _dense_vector_desc = std::move(other._dense_vector_desc);
    _dense_matrix_desc = std::move(other._dense_matrix_desc);
    _matrix_operation = other._matrix_operation;
}

ScalarView DTensor::scalar_view() const noexcept {
    LUISA_ASSERT(_scalar_desc, "mismatching tensor type");
    return ScalarView{
        .storage = dense_storage_view(),
        .desc = *_scalar_desc};
}

DenseVectorView DTensor::dense_vector_view() const noexcept {
    LUISA_ASSERT(_dense_vector_desc, "mismatching tensor type");
    return DenseVectorView{
        .n = _shape[0],
        .storage = dense_storage_view(),
        .desc = *_dense_vector_desc};
}

DenseMatrixView DTensor::dense_matrix_view() const noexcept {
    LUISA_ASSERT(_dense_matrix_desc, "mismatching tensor type");
    return DenseMatrixView{
        .row = _shape[0],
        .col = _shape[1],
        .storage = dense_storage_view(),
        .desc = *_dense_matrix_desc,
        .operation = _matrix_operation};
}

SparseVectorView DTensor::sparse_vector_view() const noexcept {
    LUISA_ASSERT(_sparse_vector_desc, "mismatching tensor type");
    return SparseVectorView{
        .n = _shape[0],
        .storage = sparse_vector_storage_view(),
        .desc = *_sparse_vector_desc};
}

SparseMatrixView DTensor::sparse_matrix_view() const noexcept {
    LUISA_ASSERT(_sparse_matrix_desc, "mismatching tensor type");
    return SparseMatrixView{
        .row = _shape[0],
        .col = _shape[1],
        .storage = basic_sparse_matrix_storage_view(),
        .desc = *_sparse_matrix_desc,
        .operation = _matrix_operation};
}
}// namespace luisa::compute::tensor