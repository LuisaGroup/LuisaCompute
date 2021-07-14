//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#import <runtime/command.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

class MetalCommandEncoder : public CommandVisitor {

private:
    MetalDevice *_device;
    id<MTLCommandBuffer> _command_buffer;
    MetalRingBuffer &_upload_ring_buffer;
    MetalRingBuffer &_download_ring_buffer;
    
private:
    [[nodiscard]] MetalBufferView _upload(const void *host_ptr, size_t size) noexcept;
    [[nodiscard]] MetalBufferView _download(void *host_ptr, size_t size) noexcept;

public:
    MetalCommandEncoder(MetalDevice *device,
                        id<MTLCommandBuffer> cb,
                        MetalRingBuffer &upload_ring_buffer,
                        MetalRingBuffer &download_ring_buffer) noexcept;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const AccelTraceClosestCommand *command) noexcept override;
    void visit(const AccelTraceAnyCommand *command) noexcept override;
    void visit(const AccelUpdateCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;
    void visit(const ShaderDispatchCommand *command) noexcept override;
};

}// namespace luisa::compute::metal
