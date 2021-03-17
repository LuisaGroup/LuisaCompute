//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <runtime/device.h>

namespace luisa::compute {

class FakeDevice : public Device {
    uint64_t _create_buffer(size_t) noexcept override { return 0; }
    void _dispose_buffer(uint64_t) noexcept override {}
    uint64_t _create_stream() noexcept override { return 0; }
    void _dispose_stream(uint64_t) noexcept override {}
    void _dispatch(uint64_t stream, BufferCopyCommand) noexcept override {}
    void _dispatch(uint64_t stream, BufferUploadCommand) noexcept override {}
    void _dispatch(uint64_t stream, BufferDownloadCommand) noexcept override {}
    void _dispatch(uint64_t stream, KernelLaunchCommand) noexcept override {}
    void _dispatch(uint64_t stream_handle, SynchronizeCommand) noexcept override {}
    void _dispatch(uint64_t stream_handle, std::function<void()> f) noexcept override { f(); }
};

}
