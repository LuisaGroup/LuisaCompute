#pragma once
#include <luisa/tensor/las_interface.h>
#include "../../common/rust_device_common.h"

namespace luisa::compute::cpu::tensor {
class CpuLAS : public luisa::compute::tensor::LASInterface {
    template<typename T>
    using S = luisa::shared_ptr<T>;

    using BackendTensorRes = luisa::compute::tensor::BackendTensorRes;
    using DTensor = luisa::compute::tensor::DTensor;
    using MatrixMulOptions = luisa::compute::tensor::MatrixMulOptions;
    using DenseStorageView = luisa::compute::tensor::DenseStorageView;
    
    uint64_t _stream_handle = 0ul;
    DeviceInterface &_device;
public:
    CpuLAS(DeviceInterface &device, uint64_t stream_handle) noexcept;
    virtual ~CpuLAS() noexcept;
    virtual S<BackendTensorRes> alloc_backend_tensor_res(const DTensor &) noexcept override;

    // BLAS
    // level-1

    // Note the result of Iamax and Iamin is an index starting from 1 (not 0, because of the fortran style)
    virtual void iamax(DTensor &result, const DTensor &vec_x) noexcept override;
    virtual void iamin(DTensor &result, const DTensor &vec_x) noexcept override;

    virtual void dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept override;
    virtual void nrm2(DTensor &result, const DTensor &vec_x) noexcept override;

    // level-2
    virtual void mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept override;
    virtual void sv(DTensor &x, const DTensor &A) noexcept override;
    virtual void mv_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept override;
    virtual void mv_strided_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept override;


    // level-3
    virtual void mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept override;
    virtual void sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept override;
    virtual void mm_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept override;
    virtual void mm_stride_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept override;
    virtual void sm_batched(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept override;

    // SPARSE
    virtual void sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept override;
    virtual void gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept override;
    virtual void scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept override;

    virtual size_t spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept override;
    virtual void spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept override;

    // level-2
    virtual size_t spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept override;
    virtual void spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept override;

private:
    template<typename F>
    void invoke(F&& func) noexcept {
        CommandList cmdlist;
        cmdlist.add_callback(std::forward<F>(func));
        _device.dispatch(_stream_handle, std::move(cmdlist));
    }
};
}// namespace luisa::compute::cuda::tensor