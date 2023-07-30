#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>
#include "view.h"
#include "las_interface.h"
namespace luisa::compute::tensor {
class LC_TENSOR_API JitSession {
    class Impl;
public:
    // TODO: move to private
    JitSession() noexcept;
    JitSession &get() noexcept;
    Stream &stream() noexcept;
};

//class DTensor {
//     Device &device;
//     bool _requires_grad = false;
//     bool _reserve_memory = false;
//     bool _dirty = false;
//     std::array<size_t, 3> _shape;
//     std::array<size_t, 3> _stride;
//     luisa::optional<Buffer<T>> _buffer;
//     luisa::optional<Var<T>> _var;
// public:
//     using shape_type = std::array<size_t, 3>;
//     using value_type = T;

// private:
//     static size_t compute_size(shape_type s) {
//         size_t size = 1;
//         for (auto i : s) {
//             size *= i;
//         }
//         return size;
//     }
// public:

//     explicit DTensor(Device &device) : device(device) {}

//     DTensor(Device &device, Buffer<T> buffer, shape_type shape) : device(device), _buffer(buffer), _shape(shape) {
//     }
//     static DTensor zeros(Device &device, shape_type shape) noexcept {
//         auto size = compute_size(shape);
//         auto tensor = Tensor{device, device.create_buffer<T>(size), shape};
//     }
//     void fill(const T &value) noexcept {
//     }
//     [[nodiscard]] luisa::optional<Buffer<T> &> buffer() noexcept;
//     [[nodiscard]] Tensor &requires_grad(bool requires_grad) noexcept {
//         _requires_grad = requires_grad;
//         return *this;
//     }
//     [[nodiscard]] Tensor &reserve_memory(bool reserve_memory) noexcept {
//         _reserve_memory = reserve_memory;
//         return *this;
//     }
//     // inplace operations
//     void scatter_(const Tensor<uint, Dim> &index, const Tensor<T, Dim> &src) noexcept {
//     }
//     [[nodiscard]] DTensor<T> gather(const DTensor<uint> &index) const noexcept {
//         // TODO: implement
//         return DTensor<T>{device};
//     }

//     // template<size_t... Is>
//     // [[nodiscard]] Tensor<T, Dim + sizeof...(Is)> repeat(Is...) {
//     //     // TODO: implement
//     //     return Tensor<T, Dim>{device};
//     // }
// };

// template<class R, size_t Dim, class... Ts>
// Tensor<R, Dim> map(const Tensor<Ts, Dim> &... ts) noexcept {
//     // TODO: implement
// }

enum class TensorType {
    SCALAR = 0,
    VECTOR = 1,
    MATRIX = 2
};

class DTensor {// Simple, just for test
    Device &_device;
    TensorType type;
public:
    DTensor(Device &device) noexcept : _device{device} {}

    void as_scalar() {
        type = TensorType::SCALAR;
        buffer = _device.create_buffer<float>(1);
    }

    void as_dense_vector(int size) {
        type = TensorType::VECTOR;
        buffer = _device.create_buffer<float>(size);
    }

    int lda = -1;
    void as_dense_matrix(int row, int col) {
        type = TensorType::MATRIX;
        lda = row;
        buffer = _device.create_buffer<float>(lda * col);
    }
    luisa::compute::Buffer<float> buffer;
    uint64_t scalar_view() const noexcept{
        return buffer.handle();
    }
    DenseMatrixView dense_matrix_view() const noexcept;
    DenseVectorView dense_vector_view() const noexcept {
        DenseVectorView ret;
        ret.buffer_handle = buffer.handle();
        ret.inc = 1;
        ret.offset = 0;
        ret.size = buffer.size();
        return ret;
    }
};
}// namespace luisa::compute::tensor