#pragma once
#include <luisa/tensor/las_interface.h>

namespace luisa::compute::tensor {
template<typename T>
class Tensor;

class BackendTensorRes {
public:
    BackendTensorRes() noexcept {}
    virtual ~BackendTensorRes() noexcept {};

    BackendTensorRes(const BackendTensorRes &) = delete;
    BackendTensorRes &operator=(const BackendTensorRes &) = delete;
    BackendTensorRes(BackendTensorRes &&) noexcept = default;
    BackendTensorRes &operator=(BackendTensorRes &&) noexcept = default;
};
}// namespace luisa::compute::tensor