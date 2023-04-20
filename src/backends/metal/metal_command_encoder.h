//
// Created by Mike Smith on 2023/4/16.
//

#pragma once

#include <runtime/rhi/command.h>
#include <runtime/command_list.h>
#include <backends/metal/metal_api.h>
#include <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

class MetalCommandEncoder : public MutableCommandVisitor {

private:
    MetalStream *_stream;
    MTL::CommandBuffer *_command_buffer{nullptr};
    luisa::vector<MetalCallbackContext *> _callbacks;

private:
    void _prepare_command_buffer() noexcept;

public:
    explicit MetalCommandEncoder(MetalStream *stream) noexcept;
    ~MetalCommandEncoder() noexcept override = default;
    [[nodiscard]] MTL::CommandBuffer *command_buffer() noexcept;
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
    void visit(MeshBuildCommand *command) noexcept override;
    void visit(ProceduralPrimitiveBuildCommand *command) noexcept override;
    void visit(BindlessArrayUpdateCommand *command) noexcept override;
    void visit(CustomCommand *command) noexcept override;
    void add_callback(MetalCallbackContext *cb) noexcept;
    MTL::CommandBuffer *submit(CommandList::CallbackContainer &&user_callbacks) noexcept;

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
