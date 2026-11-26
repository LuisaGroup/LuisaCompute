//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "../common/host_transfer_range.h"
#include "fallback_command_queue.h"

namespace luisa::compute::fallback {

class FallbackStream final {

private:
    struct PendingHostDownload {
        HostByteRange range;
        luisa::shared_ptr<std::atomic_bool> done;
    };

    std::mutex _download_mutex;
    std::condition_variable _download_cv;
    luisa::vector<PendingHostDownload> _pending_downloads;
    FallbackCommandQueue _queue;

    void _wait_if_overlapping_host_download(HostByteRange range) noexcept;
    [[nodiscard]] luisa::shared_ptr<std::atomic_bool> _record_host_download(HostByteRange range) noexcept;
    void _complete_host_download(const luisa::shared_ptr<std::atomic_bool> &done) noexcept;

#define LUISA_FALLBACK_STREAM_ENQUEUE_COMMAND_DECL(COMMAND_TYPE) \
    void _enqueue(luisa::unique_ptr<COMMAND_TYPE> cmd) noexcept;
    LUISA_MAP(LUISA_FALLBACK_STREAM_ENQUEUE_COMMAND_DECL, LUISA_COMPUTE_RUNTIME_COMMANDS)
#undef LUISA_FALLBACK_STREAM_ENQUEUE_COMMAND_DECL

public:
    explicit FallbackStream(size_t queue_size = 8u) noexcept;
    ~FallbackStream() noexcept = default;
    [[nodiscard]] auto queue() noexcept { return &_queue; }
    [[nodiscard]] auto native_handle() noexcept { return queue(); }
    void synchronize() noexcept {
        _queue.synchronize();
        std::lock_guard lock{_download_mutex};
        _pending_downloads.clear();
    }
    void dispatch(CommandList &&cmd_list) noexcept;
    void dispatch(luisa::move_only_function<void()> &&f) noexcept;
};

}// namespace luisa::compute::fallback
