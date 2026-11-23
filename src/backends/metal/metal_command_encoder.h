#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include "metal_api.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

class MetalBindlessArray;

class MetalCommandEncoder : public MutableCommandVisitor {

private:
    MetalStream *_stream;
    MTL::CommandBuffer *_command_buffer{nullptr};
    MTL::ComputeCommandEncoder *_open_compute_encoder{nullptr};
    MTL::BlitCommandEncoder *_open_blit_encoder{nullptr};
    luisa::vector<MetalCallbackContext *> _callbacks;
    luisa::fixed_vector<const MTL::Resource *, 128u> _written_buffers;
    luisa::fixed_vector<const MTL::Resource *, 32u> _written_textures;
    bool _written_buffers_overflow{false};
    bool _written_textures_overflow{false};
    bool _wrote_bindless{false};

protected:
    void _prepare_command_buffer() noexcept;

public:
    explicit MetalCommandEncoder(MetalStream *stream) noexcept;
    ~MetalCommandEncoder() noexcept override;
    [[nodiscard]] auto stream() const noexcept { return _stream; }
    [[nodiscard]] auto owner_device() const noexcept { return _stream->owner_device(); }
    [[nodiscard]] auto device() const noexcept { return _stream->device(); }
    [[nodiscard]] MTL::CommandBuffer *command_buffer() noexcept;
    [[nodiscard]] MTL::ComputeCommandEncoder *compute_encoder() noexcept;
    [[nodiscard]] MTL::BlitCommandEncoder *blit_encoder() noexcept;
    void close_compute_encoder() noexcept;
    void close_blit_encoder() noexcept;
    void close_encoders() noexcept;
    [[nodiscard]] MTL::AccelerationStructureCommandEncoder *begin_acceleration_structure_encoder() noexcept;
    void end_acceleration_structure_encoder(MTL::AccelerationStructureCommandEncoder *encoder) noexcept;
    void record_written_buffer(const MTL::Resource *resource) noexcept;
    void record_written_texture(const MTL::Resource *resource) noexcept;
    void record_written_bindless() noexcept;
    void prepare_compute_dispatch_barriers(MTL::ComputeCommandEncoder *compute_encoder,
                                           luisa::span<MetalBindlessArray *const> read_bindless_arrays,
                                           bool reads_any_buffer,
                                           bool reads_any_texture) noexcept;
    void set_command_buffer_label(NS::String *label) noexcept;
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
    virtual MTL::CommandBuffer *submit(CommandList::CallbackContainer &&user_callbacks) noexcept;

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

