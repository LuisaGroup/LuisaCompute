#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>
#include "view.h"

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
    NONE = -1,
    SCALAR = 0,
    VECTOR = 1,
    MATRIX = 2
};
class TensorMaker;
template<typename T>
class Tensor;

enum class TensorBasicDataType {
    NONE = 0,
    INT32 = 1,
    INT64 = 2,
    FLOAT32 = 3,
    FLOAT64 = 4
};

class LC_TENSOR_API DTensor {// Simple, just for test
public:
    DTensor() noexcept {}

    TensorType type() const noexcept {
        if (_shape.size() == 0) return TensorType::SCALAR;
        if (_shape.size() == 1) return TensorType::VECTOR;
        if (_shape.size() == 2) return TensorType::MATRIX;
    }

    ScalarView scalar_view() const noexcept;
    DenseVectorView dense_vector_view() const noexcept;
    DenseMatrixView dense_matrix_view() const noexcept;


protected:
    virtual void buffer_info(uint64_t &buffer_handle, uint64_t &buffer_offset, uint64_t &buffer_total_size) const noexcept = 0;

private:
    friend class TensorMaker;
    template<typename T>
    friend class Tensor;

    // basic info:
    TensorBasicDataType _basic_data_type = TensorBasicDataType::NONE;// float, double or int ?
    luisa::vector<int> _shape;                                       // 0, [N], [M,N], [L,M,N] ...?

    struct {// dense vector
        int incx = 1;
    } _dense_vector_view_data = {};

    struct {// dense matrix
        int _lda = -1;
        int _kl = -1, _ku = -1;// for band matrix
        MatrixOperation _operation = MatrixOperation::NONE;
        DenseMatrixShape _shape = DenseMatrixShape::GENERAL;
        DenseMatrixProperty _property = DenseMatrixProperty::NONE;
        DenseMatrixFillMode _fill_mode = DenseMatrixFillMode::NONE;
        DenseMatrixDiagType _diag_type = DenseMatrixDiagType::NON_UNIT;
    } _dense_matrix_view_data = {};

    static void trans(MatrixOperation &op) {
        op = op == MatrixOperation::TRANS ? MatrixOperation::NONE : MatrixOperation::TRANS;
    }
};

template<typename Ty>
constexpr TensorBasicDataType enum_data_type() {
    using T = std::remove_all_extents_t<Ty>;
    if constexpr (std::is_same_v<T, int>)
        return TensorBasicDataType::INT32;
    else if constexpr (std::is_same_v<T, int64_t>)
        return TensorBasicDataType::INT64;
    else if constexpr (std::is_same_v<T, float>)
        return TensorBasicDataType::FLOAT32;
    else if constexpr (std::is_same_v<T, double>)
        return TensorBasicDataType::FLOAT64;
    else {
        LUISA_ERROR_WITH_LOCATION(
            "Unsupported data type: {}",
            typeid(T).name());
        return TensorBasicDataType::NONE;
    }
}

template<typename Ty>
class Tensor : public DTensor {

public:
    explicit Tensor(Device &device) noexcept : DTensor{}, _device{device} {
        _basic_data_type = enum_data_type<Ty>();
    }

    explicit Tensor(const DTensor &tensor, Device &device) noexcept : DTensor{tensor}, _device{device} {
        _basic_data_type = enum_data_type<Ty>();
        _has_storage = false;
        _buffer = {};
    }

    BufferView<Ty> buffer_view() const noexcept {
        if (_has_storage)
            return _buffer.view();
        else
            return _buffer_view;
    }

    auto copy_from(const Ty *data) noexcept { return buffer_view().copy_from(data); }
    auto copy_to(Ty *data) noexcept { return buffer_view().copy_to(data); }

    void alloc_scalar() {
        _has_storage = true;
        _shape.clear();
        _buffer = _device.create_buffer<Ty>(1);
    }

    void alloc_dense_vector(int n) {
        _has_storage = true;
        _shape = {n};
        _dense_vector_view_data.incx = 1;
        _buffer = _device.create_buffer<Ty>(n);
    }

    void alloc_dense_matrix(int row, int col) noexcept {
        _has_storage = true;
        _shape = {row, col};
        // pow2 maybe better
        auto lda = luisa::next_pow2(static_cast<uint32_t>(col));
        _dense_matrix_view_data._lda = lda;
        _dense_matrix_view_data._shape = DenseMatrixShape::GENERAL;
        _dense_matrix_view_data._operation = MatrixOperation::NONE;
        _dense_matrix_view_data._property = DenseMatrixProperty::NONE;
        _dense_matrix_view_data._fill_mode = DenseMatrixFillMode::NONE;
        _dense_matrix_view_data._diag_type = DenseMatrixDiagType::NON_UNIT;

        _buffer = _device.create_buffer<Ty>(lda * row);
    }

    Tensor T() const noexcept {
        Tensor ret{*this, _device};
        ret._buffer_view = buffer_view();
        DTensor::trans(ret._dense_matrix_view_data._operation);
        return ret;
    }
protected:
    void buffer_info(uint64_t &buffer_handle, uint64_t &buffer_offset, uint64_t &buffer_total_size) const noexcept override {
        if (_has_storage) {
            buffer_handle = _buffer.handle();
            buffer_offset = 0;
            buffer_total_size = _buffer.size();
        } else {
            buffer_handle = _buffer_view.handle();
            buffer_offset = _buffer_view.offset();
            buffer_total_size = _buffer_view.size();
        }
    }
private:
    bool _has_storage = false;
    Device &_device;
    Buffer<Ty> _buffer;
    BufferView<Ty> _buffer_view;
};

class TensorMaker {// Simple, just for test
    Device &_device;
    luisa::vector<Buffer<float>> float_buffers;//just for test
    luisa::vector<Buffer<int>> int_buffers;    //just for test
public:
    TensorMaker(Device &device) noexcept : _device{device} {}

    template<typename T = float>
    Tensor<T> scalar() noexcept {
        Tensor<T> tensor{_device};
        tensor.alloc_scalar();
        return tensor;
    }

    Tensor<float> dense_vector(int size) noexcept {
        Tensor<float> tensor{_device};
        tensor.alloc_dense_vector(size);
        return tensor;
    }

    // make general matrix
    Tensor<float> dense_matrix(int row, int col) noexcept {
        Tensor<float> tensor{_device};
        tensor.alloc_dense_matrix(row, col);
        return tensor;
    }
};
}// namespace luisa::compute::tensor