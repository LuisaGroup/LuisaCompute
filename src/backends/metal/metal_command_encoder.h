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

public:
    MetalCommandEncoder(MetalDevice *device, id<MTLCommandBuffer> cb) noexcept;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const KernelLaunchCommand *command) noexcept override;
};

}// namespace luisa::compute::metal
