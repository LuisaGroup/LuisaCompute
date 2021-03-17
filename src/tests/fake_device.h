//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <runtime/device.h>

namespace luisa::compute {

class FakeDevice : public Device {
    uint64_t _create_buffer(size_t size_bytes) noexcept override { return 0; }
    void _dispose_buffer(uint64_t handle) noexcept override {}
    uint64_t _create_stream() noexcept override { return 0; }
    void _dispose_stream(uint64_t handle) noexcept override {}
    void _dispatch(uint64_t stream, BufferCopyCommand command) noexcept override {}
    void _dispatch(uint64_t stream, BufferUploadCommand command) noexcept override {}
    void _dispatch(uint64_t stream, BufferDownloadCommand command) noexcept override {}
    void _dispatch(uint64_t stream, KernelLaunchCommand command) noexcept override {}
    void _dispatch(uint64_t stream_handle, SynchronizeCommand command) noexcept override {}
};

}
