//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <core/stl/queue.h>
#include <core/stl/string.h>

#include <runtime/rhi/stream_tag.h>
#include <runtime/command_list.h>
#include <backends/metal/metal_api.h>
#include <backends/metal/metal_stage_buffer_pool.h>

namespace luisa::compute::metal {

class MetalEvent;
class MetalTexture;
class MetalSwapchain;

class MetalStream {

public:
    using CallbackContainer = luisa::vector<MetalCallbackContext *>;

private:
    MTL::CommandQueue *_queue;
    MetalStageBufferPool _upload_pool;
    MetalStageBufferPool _download_pool;
    luisa::queue<CallbackContainer> _callback_lists;
    spin_mutex _callback_mutex;

public:
    MetalStream(MTL::Device *device, StreamTag tag, size_t max_commands) noexcept;
    ~MetalStream() noexcept;
    void signal(MetalEvent *event) noexcept;
    void wait(MetalEvent *event) noexcept;
    void synchronize() noexcept;
    void dispatch(CommandList &&list) noexcept;
    void present(MetalSwapchain *swapchain, MetalTexture *image) noexcept;
    void set_name(luisa::string_view name) noexcept;
    [[nodiscard]] auto device() const noexcept { return _queue->device(); }
    [[nodiscard]] auto queue() const noexcept { return _queue; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    void submit(MTL::CommandBuffer *command_buffer, CallbackContainer &&callbacks) noexcept;
};

}// namespace luisa::compute::metal
