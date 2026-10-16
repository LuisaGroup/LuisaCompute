#pragma once

#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute::simd {

class SIMDThreadPool;

class SIMDStream {

private:
    SIMDThreadPool *_thread_pool;
    DeviceInterface::StreamLogCallback _log_callback;

public:
    explicit SIMDStream(SIMDThreadPool *thread_pool) noexcept
        : _thread_pool{thread_pool} {}
    ~SIMDStream() noexcept = default;

    void dispatch(CommandList &&list) noexcept;
    void synchronize() noexcept {}
    void set_log_callback(
        DeviceInterface::StreamLogCallback callback) noexcept {
        _log_callback = std::move(callback);
    }
    [[nodiscard]] auto native_handle() noexcept { return this; }
};

}// namespace luisa::compute::simd
