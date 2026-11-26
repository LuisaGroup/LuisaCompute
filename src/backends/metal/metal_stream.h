#pragma once

#include <atomic>
#include <mutex>
#include <utility>

#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/string.h>

#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/command_list.h>
#include "../common/host_transfer_range.h"
#include "metal_api.h"
#include "metal_stage_buffer_pool.h"

LUISA_EXTERN_C void luisa_compute_metal_stream_print_function_logs(MTL::LogContainer *logs);

namespace luisa::compute::metal {

class MetalDevice;
class MetalEvent;
class MetalTexture;
class MetalSwapchain;
class MetalCommandEncoder;

class MetalStream {

public:
    using CallbackContainer = luisa::vector<MetalCallbackContext *>;

private:
    MetalDevice *_owner_device{nullptr};
    MTL::CommandQueue *_queue;
    spin_mutex _upload_pool_creation_mutex;
    spin_mutex _download_pool_creation_mutex;
    spin_mutex _callback_mutex;
    spin_mutex _dispatch_mutex;
    spin_mutex _host_download_mutex;
    std::mutex _callback_execution_mutex;
    std::atomic_uint64_t _submitted_callback_lists{0u};
    std::atomic_uint64_t _completed_callback_lists{0u};
    struct PendingHostDownload {
        HostByteRange range;
        uint64_t callback_ticket{0u};
    };
    luisa::vector<PendingHostDownload> _pending_host_downloads;
    luisa::unique_ptr<MetalStageBufferPool> _upload_pool{nullptr};
    luisa::unique_ptr<MetalStageBufferPool> _download_pool{nullptr};
    luisa::queue<CallbackContainer> _callback_lists{};
    MTL::Fence *_fence{nullptr};
    bool _fence_pending{false};

protected:
    void _wait_if_overlapping_host_download(luisa::span<const luisa::unique_ptr<Command>> cmds) noexcept;
    void _record_pending_host_downloads(luisa::span<const luisa::unique_ptr<Command>> cmds, uint64_t callback_ticket) noexcept;
    void _do_dispatch(MetalCommandEncoder &encoder, CommandList &&list) noexcept;
    virtual void _encode(MetalCommandEncoder &encoder, Command *command) noexcept;

public:
    MetalStream(MTL::Device *device, size_t max_commands) noexcept;
    MetalStream(MetalDevice *owner_device, size_t max_commands) noexcept;
    virtual ~MetalStream() noexcept;
    virtual void signal(MetalEvent *event, uint64_t value) noexcept;
    virtual void wait(MetalEvent *event, uint64_t value) noexcept;
    virtual void synchronize() noexcept;
    virtual void dispatch(CommandList &&list) noexcept;
    void present(MetalSwapchain *swapchain, MetalTexture *image) noexcept;
    virtual void set_name(luisa::string_view name) noexcept;
    [[nodiscard]] auto owner_device() const noexcept { return _owner_device; }
    [[nodiscard]] auto device() const noexcept { return _queue->device(); }
    [[nodiscard]] auto queue() const noexcept { return _queue; }
    [[nodiscard]] auto fence() const noexcept { return _fence; }
    [[nodiscard]] bool take_fence_pending() noexcept { return std::exchange(_fence_pending, false); }
    void mark_fence_pending() noexcept { _fence_pending = true; }
    [[nodiscard]] MetalStageBufferPool *upload_pool() noexcept;
    [[nodiscard]] MetalStageBufferPool *download_pool() noexcept;
    virtual void submit(MTL::CommandBuffer *command_buffer, CallbackContainer &&callbacks) noexcept;
};

}// namespace luisa::compute::metal
