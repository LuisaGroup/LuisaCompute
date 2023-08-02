#pragma once

#include <luisa/runtime/buffer.h>
#include <luisa/tensor/view/dense_storage_view.h>

namespace luisa::compute::tensor {
template<typename T>
class DenseStorage {
public:
    Buffer<T> buffer;
    auto view() const noexcept {
        return DenseStorageView{
            .buffer_handle = buffer.handle(),
            .buffer_stride = buffer.stride(),
            .buffer_offset = 0u,
            .buffer_total_size = buffer.size(),
        };
    }
    operator DenseStorageView() const noexcept {
        return view();
    }
    auto copy_from(const T *data) noexcept {
        return buffer.copy_from(data);
    }
    auto copy_to(T *data) noexcept {
        return buffer.copy_to(data);
    }
    auto copy_from(const DenseStorage &other) noexcept {
        return buffer.copy_from(other.buffer);
    }
};
}// namespace luisa::compute::tensor