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
    MetalCommandEncoder(MetalDevice *device, id<MTLCommandBuffer> cb) noexcept
        : _device{device}, _command_buffer{cb} {}

    void visit(const BufferCopyCommand *command) noexcept override {
        auto blit_encoder = [_command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:_device->buffer(command->src_handle())
                        sourceOffset:command->src_offset()
                            toBuffer:_device->buffer(command->dst_handle())
                   destinationOffset:command->dst_offset()
                                size:command->size()];
        [blit_encoder endEncoding];
    }

    void visit(const BufferUploadCommand *command) noexcept override {

        auto buffer = _device->buffer(command->handle());

        auto t0 = std::chrono::high_resolution_clock::now();
        auto temporary = [_device->handle() newBufferWithBytes:command->data()
                                                        length:command->size()
                                                       options:MTLResourceStorageModeShared];
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        LUISA_VERBOSE_WITH_LOCATION(
            "Allocated temporary shared buffer with size {} in {} ms.",
            command->size(), (t1 - t0) / 1ns * 1e-6);

        auto blit_encoder = [_command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:temporary
                        sourceOffset:0u
                            toBuffer:buffer
                   destinationOffset:command->offset()
                                size:command->size()];
        [blit_encoder endEncoding];
    }

    void visit(const BufferDownloadCommand *command) noexcept override {

        auto buffer = _device->buffer(command->handle());
        auto data = command->data();
        auto size = command->size();

        auto t0 = std::chrono::high_resolution_clock::now();
        auto temporary = [_device->handle() newBufferWithLength:size
                                                        options:MTLResourceStorageModeShared];
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        LUISA_VERBOSE_WITH_LOCATION(
            "Allocated temporary shared buffer with size {} in {} ms.",
            size, (t1 - t0) / 1ns * 1e-6);

        auto blit_encoder = [_command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:buffer
                        sourceOffset:command->offset()
                            toBuffer:temporary
                   destinationOffset:0u
                                size:size];
        [blit_encoder endEncoding];
        [_command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            std::memcpy(data, temporary.contents, size);
        }];
    }

    void visit(const KernelLaunchCommand *command) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Not implemented!");
    }
};

}
