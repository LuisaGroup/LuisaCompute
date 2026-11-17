#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include "metal_api.h"
#include "metal_accel.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

class MetalBindlessArray;

class MetalCommandEncoder : public MutableCommandVisitor {

private:
    MetalStream *_stream;
    MTL4::CommandBuffer *_command_buffer{nullptr};
    MetalStream::CommandAllocatorBundle _allocator_bundle{};
    MTL4::ComputeCommandEncoder *_open_compute_encoder{nullptr};
    MTL::Stages _last_compute_stages{0};
    luisa::fixed_vector<const MTL::Allocation *, 64u> _epoch_read_allocations;
    luisa::fixed_vector<const MTL::Allocation *, 64u> _epoch_written_allocations;
    luisa::fixed_vector<MetalBindlessArray *, 8u> _epoch_read_bindless_arrays;
    luisa::vector<MetalCallbackContext *> _callbacks;
    luisa::unordered_set<const MTL::Allocation *> _allocations;
    luisa::fixed_vector<const MTL::ResidencySet *, 8u> _residency_sets;
    luisa::fixed_vector<luisa::shared_ptr<MetalAccel::ResidencySnapshot>, 4u> _accel_snapshots;
    MetalTimingSubmission _timing;
    bool _async_host_completion{false};
    bool _in_tail_downloads{false};
    bool _last_blit_was_download{false};
    bool _epoch_force_barrier{false};

protected:
    void _prepare_command_buffer() noexcept;

public:
    explicit MetalCommandEncoder(MetalStream *stream) noexcept;
    ~MetalCommandEncoder() noexcept override;
    [[nodiscard]] auto stream() const noexcept { return _stream; }
    [[nodiscard]] auto device() const noexcept { return _stream->device(); }
    [[nodiscard]] MTL4::CommandBuffer *command_buffer() noexcept;
    [[nodiscard]] MTL4::ComputeCommandEncoder *compute_encoder() noexcept;
    [[nodiscard]] MTL4::ComputeCommandEncoder *coalesced_compute_encoder(MTL::Stages next_stages) noexcept;
    [[nodiscard]] MTL4::ComputeCommandEncoder *coalesced_dispatch_encoder(
        luisa::span<const MTL::Allocation *const> read_allocations,
        luisa::span<const MTL::Allocation *const> write_allocations,
        luisa::span<MetalBindlessArray *const> read_bindless_arrays,
        bool force_barrier) noexcept;
    [[nodiscard]] MTL4::ComputeCommandEncoder *coalesced_download_blit_encoder() noexcept;
    void set_in_tail_downloads(bool value) noexcept { _in_tail_downloads = value; }
    void close_compute_encoder() noexcept;
    [[nodiscard]] MTL4::RenderCommandEncoder *render_encoder(
        MTL4::RenderPassDescriptor *descriptor) noexcept;
    [[nodiscard]] MTL4::ArgumentTable *argument_table(
        size_t buffer_count,
        size_t texture_count = 0u,
        size_t sampler_count = 0u) noexcept;
    [[nodiscard]] MTL::GPUAddress upload(const void *data, size_t size) noexcept;
    void use_resource(const MTL::Allocation *allocation) noexcept;
    void use_device_resource(const MTL::Allocation *allocation) noexcept;
    void use_stage_allocation(const MetalStageBufferPool::Allocation *allocation) noexcept;
    void use_residency_set(MTL::ResidencySet *residency_set) noexcept;
    void use_accel_residency(luisa::shared_ptr<MetalAccel::ResidencySnapshot> snapshot) noexcept;
    void note_timing_non_dispatch_work() noexcept;
    void note_timing_indirect_dispatch() noexcept;
    [[nodiscard]] uint32_t begin_dispatch_timing(MTL4::ComputeCommandEncoder *encoder,
                                                uint64_t shader_checksum, uint3 dispatch_size, uint3 block_size) noexcept;
    void end_dispatch_timing(MTL4::ComputeCommandEncoder *encoder, uint32_t ordinal) noexcept;
    void visit(BufferUploadCommand *command) noexcept override;
    void visit(BufferDownloadCommand *command) noexcept override;
    void visit(BufferCopyCommand *command) noexcept override;
    void visit(BufferToTextureCopyCommand *command) noexcept override;
    void visit(ShaderDispatchCommand *command) noexcept override;
    void visit(TextureUploadCommand *command) noexcept override;
    void visit(TextureDownloadCommand *command) noexcept override;
    void visit(TextureCopyCommand *command) noexcept override;
    void visit(TextureToBufferCopyCommand *command) noexcept override;
    void visit(AccelBuildCommand *command) noexcept override;
    void visit(CurveBuildCommand *command) noexcept override;
    void visit(MeshBuildCommand *command) noexcept override;
    void visit(ProceduralPrimitiveBuildCommand *command) noexcept override;
    void visit(MotionInstanceBuildCommand *command) noexcept override;
    void visit(BindlessArrayUpdateCommand *command) noexcept override;
    void visit(CustomCommand *command) noexcept override;
    void add_callback(MetalCallbackContext *cb) noexcept;
    void require_async_host_completion() noexcept { _async_host_completion = true; }
    [[nodiscard]] virtual MetalStream::SubmissionHandle submit(
        CommandList::CallbackContainer &&user_callbacks) noexcept;
    void submit_and_wait(CommandList::CallbackContainer &&user_callbacks = {}) noexcept;

    template<typename F>
    void with_upload_buffer(size_t size, F &&f) noexcept {
        _prepare_command_buffer();
        auto upload_buffer = _stream->upload_pool()->allocate(size);
        f(upload_buffer);
        add_callback(upload_buffer);
    }

    template<typename F>
    void with_download_buffer(size_t size, F &&f) noexcept {
        _prepare_command_buffer();
        auto download_buffer = _stream->download_pool()->allocate(size);
        f(download_buffer);
        add_callback(download_buffer);
    }
};

}// namespace luisa::compute::metal
