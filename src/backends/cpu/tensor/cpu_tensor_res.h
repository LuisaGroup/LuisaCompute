#pragma once
#include <luisa/tensor/backend_tensor_res.h>
#include <luisa/tensor/dtensor.h>
#include <luisa/core/logging.h>
#include "enum_map.h"
#include "raw_ptr_cast.h"

namespace luisa::compute::cpu::tensor {
class BatchRes {
    vector<void *> _array_of_ptr;
public:
    BatchRes(const luisa::compute::tensor::DTensor &tensor) noexcept {
        luisa::vector<DenseStorageView> storage{};
        if (tensor.is_vector())// sparse vector
            storage = tensor.dense_vector_view().storage;
        else if (tensor.is_matrix())// sparse matrix
            storage = tensor.dense_matrix_view().storage;
        // _array_of_ptr.resize(storage.size());
        std::transform(storage.begin(), storage.end(), (void **)tensor.batch_view().storage.buffer_native_handle,
                       [&](const DenseStorageView &d) {
                           return reinterpret_cast<void *>(raw_ptr(d));
                       });
        //std::transform(storage.begin(), storage.end(), _array_of_ptr.begin(),
        //               [&](const DenseStorageView &d) {
        //                   return reinterpret_cast<void *>(raw_ptr(d));
        //               });
        //std::memcpy(tensor.batch_view().storage.buffer_native_handle, 
        //    _array_of_ptr.data(), _array_of_ptr.size() * sizeof(void *));
    }
};

class CblasRes : public luisa::compute::tensor::BackendTensorRes {
    BatchRes _batch_res;
public:
    CblasRes(const luisa::compute::tensor::DTensor &tensor) noexcept
        : _batch_res{tensor} {}

    virtual ~CblasRes() noexcept {}
};
}// namespace luisa::compute::cpu::tensor