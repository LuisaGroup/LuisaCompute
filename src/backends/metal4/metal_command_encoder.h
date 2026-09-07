#pragma once

#include <unordered_set>

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include "metal_api.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

class MetalCommandEncoder : public MutableCommandVisitor {

private:
    MetalStream *_stream;
    MTL4::CommandBuffer *_command_buffer{nullptr};
    MTL4::CommandAllocator *_command_allocator{nullptr};
    luisa::vector<MetalCallbackContext *> _callbacks;
    std::unordered_set<const MTL::Allocation *> _allocations;

protected:
    void _prepare_command_buffer() noexcept;

public:
    explicit MetalCommandEncoder(MetalStream *stream) noexcept;
    ~MetalCommandEncoder() noexcept override;
    [[nodiscard]] auto stream() const noexcept { return _stream; }
    [[nodiscard]] auto device() const noexcept { return _stream->device(); }
    [[nodiscard]] MTL4::CommandBuffer *command_buffer() noexcept;
    [[nodiscard]] MTL4::ComputeCommandEncoder *compute_encoder() noexcept;
    [[nodiscard]] MTL4::RenderCommandEncoder *render_encoder(
        MTL4::RenderPassDescriptor *descriptor) noexcept;
    [[nodiscard]] MTL4::ArgumentTable *argument_table(
        size_t buffer_count,
        size_t texture_count = 0u,
        size_t sampler_count = 0u) noexcept;
    [[nodiscard]] MTL::GPUAddress upload(const void *data, size_t size) noexcept;
    void use_resource(const MTL::Allocation *allocation) noexcept;
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
