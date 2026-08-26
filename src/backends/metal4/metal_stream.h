#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/string.h>

#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include "metal_api.h"
#include "metal_stage_buffer_pool.h"

namespace luisa::compute::metal {

class MetalEvent;
class MetalTexture;
class MetalSwapchain;
class MetalCommandEncoder;

class MetalStream {

public:
    using CallbackContainer = luisa::vector<MetalCallbackContext *>;
    using LogCallback = DeviceInterface::StreamLogCallback;

    struct Submission {
        std::atomic_bool completed{false};
    };
    using SubmissionHandle = luisa::shared_ptr<Submission>;

private:
    MTL4::CommandQueue *_queue{nullptr};
    MTL::CommandQueue *_acceleration_structure_compatibility_queue{nullptr};
    MTL::LogState *_log_state{nullptr};
    NS::String *_name{nullptr};
    size_t _max_commands{0u};
    size_t _inflight_commands{0u};
    std::mutex _inflight_mutex;
    std::condition_variable _inflight_cv;
    spin_mutex _upload_pool_creation_mutex;
    spin_mutex _download_pool_creation_mutex;
    spin_mutex _callback_mutex;
    spin_mutex _dispatch_mutex;
    std::mutex _callback_execution_mutex;
    mutable spin_mutex _log_callback_mutex;
    std::atomic_uint64_t _submitted_callback_lists{0u};
    std::atomic_uint64_t _completed_callback_lists{0u};
    luisa::unique_ptr<MetalStageBufferPool> _upload_pool{nullptr};
    luisa::unique_ptr<MetalStageBufferPool> _download_pool{nullptr};
    luisa::queue<CallbackContainer> _callback_lists{};
    LogCallback _log_callback;

protected:
    void _do_dispatch(MetalCommandEncoder &encoder, CommandList &&list) noexcept;
    virtual void _encode(MetalCommandEncoder &encoder, Command *command) noexcept;
    void _emit_shader_log(NS::String *subsystem,
                          NS::String *category,
                          NS::String *message) const noexcept;

public:
    MetalStream(MTL::Device *device, size_t max_commands) noexcept;
    virtual ~MetalStream() noexcept;
    virtual void signal(MetalEvent *event, uint64_t value) noexcept;
    virtual void wait(MetalEvent *event, uint64_t value) noexcept;
    virtual void synchronize() noexcept;
    virtual void dispatch(CommandList &&list) noexcept;
    void present(MetalSwapchain *swapchain, MetalTexture *image) noexcept;
    virtual void set_name(luisa::string_view name) noexcept;
    [[nodiscard]] auto device() const noexcept { return _queue->device(); }
    [[nodiscard]] auto queue() const noexcept { return _queue; }
    [[nodiscard]] bool supports_address_driven_acceleration_structures() const noexcept {
        return _acceleration_structure_compatibility_queue == nullptr;
    }
    [[nodiscard]] auto acceleration_structure_compatibility_queue() const noexcept {
        return _acceleration_structure_compatibility_queue;
    }
    [[nodiscard]] auto log_state() const noexcept { return _log_state; }
    [[nodiscard]] auto name() const noexcept { return _name; }
    void set_log_callback(LogCallback callback) noexcept;
    [[nodiscard]] MetalStageBufferPool *upload_pool() noexcept;
    [[nodiscard]] MetalStageBufferPool *download_pool() noexcept;
    [[nodiscard]] virtual SubmissionHandle submit(
        MTL4::CommandBuffer *command_buffer,
        MTL4::CommandAllocator *command_allocator,
        CallbackContainer &&callbacks) noexcept;
};

}// namespace luisa::compute::metal
