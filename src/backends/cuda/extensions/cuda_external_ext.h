#pragma once

#include <luisa/backends/ext/cuda/cuda_external_ext.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAExternalExtImpl final : public CUDAExternalExt {

protected:
    CUDADevice *device;

public:
    explicit CUDAExternalExtImpl(CUDADevice *device) : device{device} {}
    ~CUDAExternalExtImpl() noexcept override = default;

    void cuda_stream_signal(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) override;

    void cuda_stream_wait(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) override;

    // buffer
    void buffer_copy_async(
        /*CUdeviceptr*/ void *dst_buffer,
        /*CUdeviceptr*/ void *src_buffer,
        size_t size,
        /*CUstream*/ void *stream) override;

    void sync_stream(void *cu_stream_ptr) override;
};

}// namespace luisa::compute::cuda
