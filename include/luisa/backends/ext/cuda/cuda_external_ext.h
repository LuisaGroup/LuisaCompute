#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

class CUDAExternalExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "CUDAExternalExt";

    virtual ~CUDAExternalExt() noexcept = default;

    virtual void cuda_stream_signal(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) = 0;

    virtual void cuda_stream_wait(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) = 0;

    virtual void buffer_copy_async(
        /*CUdeviceptr*/ void *dst_buffer,
        /*CUdeviceptr*/ void *src_buffer,
        size_t size,
        /*CUstream*/ void *stream) = 0;

    virtual void sync_stream(void *cu_stream_ptr) = 0;
};

}// namespace luisa::compute
