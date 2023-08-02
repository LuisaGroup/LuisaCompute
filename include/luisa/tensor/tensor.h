#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include "dtensor.h"
#include "storage.h"

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
}// namespace luisa::compute::tensor

namespace luisa::compute::tensor {
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
    Tensor(Device &device, LASInterface *las) noexcept : DTensor{enum_data_type<Ty>()},
                                                         _device{device},
                                                         _las{las} {}

    // only copy descriptor not the underlying storage
    Tensor(const Tensor &other) noexcept : DTensor{other},
                                           _device{other._device},
                                           _las{other._las},
                                           _backend_tensor_res{other._backend_tensor_res},
                                           _dense_storage{other._dense_storage},
                                           _sparse_vector_storage{other._sparse_vector_storage},
                                           _basic_sparse_matrix_storage{other._basic_sparse_matrix_storage},
                                           _has_storage{false} {}

    virtual ~Tensor() noexcept {}

    bool has_storage() const noexcept { return _has_storage; }
    DenseStorage<Ty> &dense_storage() noexcept { return *_dense_storage; }
    const DenseStorage<Ty> &dense_storage() const noexcept { return *_dense_storage; }
    SparseVectorStorage<Ty> &sparse_vector_storage() noexcept { return *_sparse_vector_storage; }
    const SparseVectorStorage<Ty> &sparse_vector_storage() const noexcept { return *_sparse_vector_storage; }
    BasicSparseMatrixStorage<Ty> &sparse_matrix_storage() noexcept { return *_basic_sparse_matrix_storage; }
    const BasicSparseMatrixStorage<Ty> &sparse_matrix_storage() const noexcept { return *_basic_sparse_matrix_storage; }

    void alloc_scalar() noexcept {
        _has_storage = true;
        _shape.clear();

        _dense_storage = make_shared<DenseStorage<Ty>>();
        _dense_storage->buffer = _device.create_buffer<Ty>(1);

        _scalar_desc = make_unique<ScalarDesc>();
        _scalar_desc->is_host = false;// hard code

        alloc_backend_tensor_res();
    }

    void alloc_dense_vector(int n) noexcept {
        _has_storage = true;
        _shape = {n};

        _dense_storage = make_shared<DenseStorage<Ty>>();
        _dense_storage->buffer = _device.create_buffer<Ty>(n);

        _dense_vector_desc = make_unique<DenseVectorDesc>();
        _dense_vector_desc->inc = 1;// hard code
        _dense_vector_desc->n = n;
        _dense_vector_desc->offset = 0;// hard code

        alloc_backend_tensor_res();
    }

    void alloc_dense_matrix(int row, int col) noexcept {
        _has_storage = true;
        _shape = {row, col};

        auto lda = col > 4 ? (col + 3) / 4 * 4 : col;

        // storage
        _dense_storage = make_shared<DenseStorage<Ty>>();
        _dense_storage->buffer = _device.create_buffer<Ty>(lda * row);

        // desc
        _dense_matrix_desc = make_unique<DenseMatrixDesc>();
        _dense_matrix_desc->offset = 0;
        _dense_matrix_desc->row = row;
        _dense_matrix_desc->col = col;
        _dense_matrix_desc->lda = lda;
        _dense_matrix_desc->shape = DenseMatrixShape::GENERAL;
        _dense_matrix_desc->property = DenseMatrixProperty::NONE;
        _dense_matrix_desc->fill_mode = DenseMatrixFillMode::NONE;
        _dense_matrix_desc->diag_type = DenseMatrixDiagType::NON_UNIT;

        // common desc
        _matrix_operation = MatrixOperation::NONE;
        alloc_backend_tensor_res();
    }

    void alloc_coo_matrix(int row, int col, int nnz) noexcept {
        _has_storage = true;
        _shape = {row, col};

        auto coo = make_shared<COOMatrixStorage<Ty>>();
        coo->value_data().buffer = _device.create_buffer<Ty>(nnz);
        coo->row_ind().buffer = _device.create_buffer<int>(nnz);
        coo->col_ind().buffer = _device.create_buffer<int>(nnz);
        _basic_sparse_matrix_storage = coo;

        _sparse_matrix_desc = make_unique<SparseMatrixDesc>();
        _sparse_matrix_desc->format = SparseMatrixFormat::COO;
        _sparse_matrix_desc->row = row;
        _sparse_matrix_desc->col = col;
        _sparse_matrix_desc->nnz = nnz;

        _matrix_operation = MatrixOperation::NONE;
        alloc_backend_tensor_res();
    }

    void alloc_sparse_vector(int n, int nnz) noexcept {
        _has_storage = true;
        _shape = {n};

        _sparse_vector_storage = make_shared<SparseVectorStorage<Ty>>();
        _sparse_vector_storage->values.buffer = _device.create_buffer<Ty>(nnz);
        _sparse_vector_storage->indices.buffer = _device.create_buffer<int>(nnz);

        _sparse_vector_desc = make_unique<SparseVectorDesc>();
        _sparse_vector_desc->n = n;
        _sparse_vector_desc->nnz = nnz;

        alloc_backend_tensor_res();
    }

    Tensor T() const noexcept {
        auto ret = Tensor{*this};
        ret.do_transpose();
        return ret;
    }

    virtual BackendTensorRes *backend_tensor_res() const noexcept override {
        return _backend_tensor_res.get();
    }

protected:
    virtual DenseStorageView dense_storage_view() const noexcept override;
    virtual SparseVectorStorageView sparse_vector_storage_view() const noexcept override;
    virtual BasicSparseMatrixStorageView basic_sparse_matrix_storage_view() const noexcept override;
private:
    Device &_device;
    LASInterface *_las = nullptr;

    bool _has_storage = false;

    template<typename T>
    using U = luisa::unique_ptr<T>;
    template<typename T>
    using S = luisa::shared_ptr<T>;

    S<DenseStorage<Ty>> _dense_storage;
    S<SparseVectorStorage<Ty>> _sparse_vector_storage;
    S<BasicSparseMatrixStorage<Ty>> _basic_sparse_matrix_storage;// COO CSR CSC ...

    S<BackendTensorRes> _backend_tensor_res;
    void alloc_backend_tensor_res() noexcept {
        _backend_tensor_res = _las->alloc_backend_tensor_res(*this);
    }
};

class TensorMaker {// Simple, just for test
    Device &_device;
    LASInterface *_las = nullptr;
public:
    TensorMaker(Device &device, LASInterface *las) noexcept : _device{device}, _las{las} {}

    template<typename T = float>
    Tensor<T> scalar() noexcept {
        Tensor<T> tensor{_device, _las};
        tensor.alloc_scalar();
        return tensor;
    }

    Tensor<float> dense_vector(int size) noexcept {
        Tensor<float> tensor{_device, _las};
        tensor.alloc_dense_vector(size);
        return tensor;
    }

    // make general matrix
    Tensor<float> dense_matrix(int row, int col) noexcept {
        Tensor<float> tensor{_device, _las};
        tensor.alloc_dense_matrix(row, col);
        return tensor;
    }

    Tensor<float> coo_matrix(int row, int col, int nnz) noexcept {
        Tensor<float> tensor{_device, _las};
        tensor.alloc_coo_matrix(row, col, nnz);
        return tensor;
    }

    Tensor<float> sparse_vector(int n, int nnz) {
        Tensor<float> tensor{_device, _las};
        tensor.alloc_sparse_vector(n, nnz);
        return tensor;
    }

    DenseStorage<int> dense_storage(int size_in_byte) noexcept {
        DenseStorage<int> dense_storage;
        dense_storage.buffer = _device.create_buffer<int>((size_in_byte + sizeof(int) - 1) / sizeof(int));
        return dense_storage;
    }
};

void test() {
    Device *d;
    LASInterface *las;
    Tensor<float> t{*d, las};
    auto tt = t.T();
}

}// namespace luisa::compute::tensor

// impl
namespace luisa::compute::tensor {
template<typename Ty>
DenseStorageView Tensor<Ty>::dense_storage_view() const noexcept {
    LUISA_ASSERT(_dense_storage, "mismatching tensor type");
    return _dense_storage->view();
}

template<typename Ty>
SparseVectorStorageView Tensor<Ty>::sparse_vector_storage_view() const noexcept {
    LUISA_ASSERT(_sparse_vector_storage, "mismatching tensor type");
    return _sparse_vector_storage->view();
}

template<typename Ty>
BasicSparseMatrixStorageView Tensor<Ty>::basic_sparse_matrix_storage_view() const noexcept {
    LUISA_ASSERT(_basic_sparse_matrix_storage, "mismatching tensor type");
    return _basic_sparse_matrix_storage->view();
}
}// namespace luisa::compute::tensor