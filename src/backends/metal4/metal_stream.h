#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include "../common/host_transfer_range.h"
#include "metal_api.h"
#include "metal_stage_buffer_pool.h"
#include "metal_timing.h"

namespace luisa::compute::metal {

class MetalDevice;
class MetalEvent;
class MetalTexture;
class MetalSwapchain;
class MetalCommandEncoder;

class MetalStream {

public:
    using CallbackContainer = luisa::vector<MetalCallbackContext *>;
    using LogCallback = DeviceInterface::StreamLogCallback;

    static constexpr size_t pooled_argument_table_buffer_slots = 16u;
    static constexpr size_t pooled_argument_table_texture_slots = 4u;

    struct CommandAllocatorBundle {
        MTL4::CommandAllocator *allocator{nullptr};
        MTL4::ArgumentTable *argument_table{nullptr};
    };

    struct Submission {
        uint64_t ticket{0u};
        std::atomic_bool completed{false};
    };
    using SubmissionHandle = luisa::shared_ptr<Submission>;

    struct PendingSubmission {
        uint64_t ticket{0u};
        MTL4::CommandBuffer *command_buffer{nullptr};
        CommandAllocatorBundle allocator_bundle{};
        CallbackContainer callbacks{};
        SubmissionHandle submission{};
    };

private:
    MetalDevice *_owner_device{nullptr};
    MTL4::CommandQueue *_queue{nullptr};
    MTL::SharedEvent *_completion_event{nullptr};
    MTL::ResidencySet *_stream_residency_set{nullptr};
    MTL::CommandQueue *_acceleration_structure_compatibility_queue{nullptr};
    MTL::LogState *_log_state{nullptr};
    MTL4::CommandBufferOptions *_command_buffer_options{nullptr};
    NS::String *_name{nullptr};
    size_t _max_commands{0u};
    size_t _inflight_commands{0u};
    std::atomic_bool _has_unsubmitted_queue_ops{false};
    std::mutex _retire_mutex;
    std::mutex _submission_mutex;
    std::atomic<std::thread::id> _retiring_thread{};
    std::atomic_uint64_t _submitted_ticket{0u};
    std::atomic_uint64_t _retired_ticket{0u};
    std::atomic_uint64_t _active_feedback_handlers{0u};
    luisa::queue<PendingSubmission> _pending_submissions{};
    std::mutex _inflight_mutex;
    std::condition_variable _inflight_cv;
    spin_mutex _upload_pool_creation_mutex;
    spin_mutex _download_pool_creation_mutex;
    spin_mutex _stream_residency_mutex;
    spin_mutex _callback_mutex;
    spin_mutex _dispatch_mutex;
    spin_mutex _command_allocator_pool_mutex;
    std::mutex _callback_execution_mutex;
    mutable spin_mutex _log_callback_mutex;
    std::atomic_uint64_t _submitted_callback_lists{0u};
    std::atomic_uint64_t _completed_callback_lists{0u};
    luisa::unique_ptr<MetalStageBufferPool> _upload_pool{nullptr};
    luisa::unique_ptr<MetalStageBufferPool> _download_pool{nullptr};
    luisa::queue<CallbackContainer> _callback_lists{};
    luisa::vector<CommandAllocatorBundle> _command_allocator_pool;
    LogCallback _log_callback;
    // Only the opt-in diagnostic path takes this mutex. The disabled path
    // does not allocate counters or alter command encoding/submission.
    std::atomic_bool _timing_enabled{false};
    spin_mutex _timing_mutex;
    luisa::shared_ptr<MetalTimingSession> _timing_session;
    struct PendingHostDownload {
        HostByteRange range;
        SubmissionHandle submission;
    };
    spin_mutex _host_download_mutex;
    luisa::vector<PendingHostDownload> _pending_host_downloads;

protected:
    void _wait_if_overlapping_host_download(luisa::span<const luisa::unique_ptr<Command>> cmds) noexcept;
    void _record_pending_host_downloads(luisa::span<const luisa::unique_ptr<Command>> cmds, const SubmissionHandle &submission) noexcept;
    void _do_dispatch(MetalCommandEncoder &encoder, CommandList &&list) noexcept;
    virtual void _encode(MetalCommandEncoder &encoder, Command *command) noexcept;
    void _emit_shader_log(NS::String *subsystem,
                          NS::String *category,
                          NS::String *message) const noexcept;

private:
    friend class MetalCommandEncoder;
    [[nodiscard]] size_t _command_allocator_pool_capacity() const noexcept;
    [[nodiscard]] CommandAllocatorBundle _acquire_command_allocator() noexcept;
    void _recycle_command_allocator(CommandAllocatorBundle bundle) noexcept;
    void _register_stream_allocation(const MTL::Allocation *allocation) noexcept;
    void _wait_for_completion_ticket(uint64_t target) noexcept;
    void _retire_submissions_up_to(uint64_t target, bool blocking) noexcept;

public:
    MetalStream(MTL::Device *device, size_t max_commands,
                MetalDevice *owner_device = nullptr) noexcept;
    virtual ~MetalStream() noexcept;
    [[nodiscard]] auto has_device_residency() const noexcept { return _owner_device != nullptr; }
    [[nodiscard]] auto has_stream_residency() const noexcept { return _stream_residency_set != nullptr; }
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
    [[nodiscard]] auto command_buffer_options() const noexcept { return _command_buffer_options; }
    [[nodiscard]] auto name() const noexcept { return _name; }
    void set_log_callback(LogCallback callback) noexcept;
    [[nodiscard]] bool timing_enabled() const noexcept { return _timing_enabled.load(std::memory_order_acquire); }
    [[nodiscard]] luisa::shared_ptr<MetalTimingSession> timing_session() noexcept;
    [[nodiscard]] bool begin_timing(luisa::shared_ptr<MetalTimingSession> session) noexcept;
    [[nodiscard]] Metal4TimingSample end_timing() noexcept;
    [[nodiscard]] MetalStageBufferPool *upload_pool() noexcept;
    [[nodiscard]] MetalStageBufferPool *download_pool() noexcept;
    void wait_submission(const SubmissionHandle &submission) noexcept;
    [[nodiscard]] virtual SubmissionHandle submit(
        MTL4::CommandBuffer *command_buffer,
        CommandAllocatorBundle allocator_bundle,
        CallbackContainer &&callbacks,
        MetalTimingSubmission timing = {},
        bool async_host_completion = false) noexcept;
};

}// namespace luisa::compute::metal
