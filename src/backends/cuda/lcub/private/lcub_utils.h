#pragma once

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <cuda_runtime_api.h>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/cuda/lcub/dcub/dcub_utils.h>

namespace luisa::compute::cuda::lcub {

namespace details {

template<typename T>
inline T *raw(luisa::compute::BufferView<T> buffer_view) noexcept {
    if (!buffer_view) return nullptr;
    return reinterpret_cast<T *>(buffer_view.native_handle()) + buffer_view.offset();
}

template<typename T>
inline const T &raw(const T &value) noexcept { return value; }

template<typename T>
inline T &raw(T &value) noexcept { return value; }

inline size_t cuda_to_lc_buffer_size(size_t size_bytes) noexcept {
    constexpr auto unit = sizeof(int);
    return (size_bytes + unit - 1) / unit;
}

inline size_t lc_to_cuda_buffer_size(size_t size_int) noexcept {
    return size_int * sizeof(int);
}

template<typename F>
inline cudaError_t inner(size_t &temp_storage_size, F &&func) noexcept {
    size_t temp_storage_bytes = -1;
    auto error = func(temp_storage_bytes);
    temp_storage_size = cuda_to_lc_buffer_size(temp_storage_bytes);
    return error;
}

template<typename F>
inline cudaError_t inner(luisa::compute::BufferView<int> d_temp_storage, F &&func) noexcept {
    size_t temp_storage_bytes = lc_to_cuda_buffer_size(d_temp_storage.size());
    return func(temp_storage_bytes);
}

}

}// namespace luisa::compute::cuda::cub::details
