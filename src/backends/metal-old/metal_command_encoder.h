//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#import <runtime/rhi/command.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

class MetalCommandEncoder final : public CommandVisitor {

private:
    MetalDevice *_device;
    MetalStream *_stream;
    id<MTLCommandBuffer> _command_buffer;
    
private:
    [[nodiscard]] MetalBufferView _upload(const void *host_ptr, size_t size) noexcept;
    [[nodiscard]] MetalBufferView _download(void *host_ptr, size_t size) noexcept;

public:
    MetalCommandEncoder(MetalDevice *device, MetalStream *stream) noexcept;
    [[nodiscard]] auto command_buffer() const noexcept { return _command_buffer; }
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;
    void visit(const ShaderDispatchCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
    void visit(const BindlessArrayUpdateCommand *command) noexcept override;
};

}// namespace luisa::compute::metal
