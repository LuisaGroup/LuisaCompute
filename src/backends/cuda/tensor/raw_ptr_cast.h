#pragma once
#include <luisa/tensor/view.h>
#include "../cuda_buffer.h"

namespace luisa::compute::cuda::tensor {
using DenseStorageView = luisa::compute::tensor::DenseStorageView;
using ScalarView = luisa::compute::tensor::ScalarView;
using DenseVectorView = luisa::compute::tensor::DenseVectorView;
using DenseMatrixView = luisa::compute::tensor::DenseMatrixView;

std::byte *
raw_ptr(const DenseStorageView &s) {
    auto cuda_buffer = reinterpret_cast<CUDABuffer *>(s.buffer_handle);
    auto ptr = reinterpret_cast<std::byte *>(cuda_buffer->handle());
    return ptr + s.buffer_offset * s.buffer_stride;
}

std::byte *raw_ptr(const ScalarView &s) { return raw_ptr(s.storage) + s.desc.offset * s.storage.buffer_stride; }
std::byte *raw_ptr(const DenseVectorView &s) { return raw_ptr(s.storage) + s.desc.offset * s.storage.buffer_stride; }
std::byte *raw_ptr(const DenseMatrixView &s) { return raw_ptr(s.storage) + s.desc.offset * s.storage.buffer_stride; }

template<typename T>
auto raw(const DenseStorageView &s) { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const ScalarView &s) { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const DenseVectorView &s) { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const DenseMatrixView &s) { return reinterpret_cast<T *>(raw_ptr(s)); }
  
}// namespace luisa::compute::cuda::tensor