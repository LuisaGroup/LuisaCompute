#pragma once
#include <luisa/tensor/view.h>

namespace luisa::compute::cpu::tensor {
using DenseStorageView = luisa::compute::tensor::DenseStorageView;
using ScalarView = luisa::compute::tensor::ScalarView;
using DenseVectorView = luisa::compute::tensor::DenseVectorView;
using DenseMatrixView = luisa::compute::tensor::DenseMatrixView;
using BatchView = luisa::compute::tensor::BatchView;

std::byte *raw_ptr(const DenseStorageView &s) noexcept {
    auto ptr = reinterpret_cast<std::byte *>(s.buffer_native_handle);
    return ptr + s.buffer_offset * s.buffer_stride;
}

std::byte *raw_ptr(const ScalarView &s) noexcept {
    return raw_ptr(s.storage) + s.desc.offset * s.storage.buffer_stride;
}

std::byte *raw_ptr(const DenseVectorView &s) noexcept {
    auto &ss = s.storage[0];
    return raw_ptr(ss) + s.desc.offset * ss.buffer_stride;
}
std::byte *raw_ptr(const DenseMatrixView &s) noexcept {
    auto &ss = s.storage[0];
    return raw_ptr(ss) + s.desc.offset * ss.buffer_stride;
}

template<typename T>
auto raw(const DenseStorageView &s) noexcept { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const ScalarView &s) noexcept { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const DenseVectorView &s) noexcept { return reinterpret_cast<T *>(raw_ptr(s)); }
template<typename T>
auto raw(const DenseMatrixView &s) noexcept { return reinterpret_cast<T *>(raw_ptr(s)); }

template<typename T>
auto raw(const BatchView &b) noexcept {
    LUISA_ASSERT(b.desc.batch_stride <= 0, "never used for strided batch tensor");
    return reinterpret_cast<T **>(raw_ptr(b.storage));
}

}// namespace luisa::compute::cuda::tensor