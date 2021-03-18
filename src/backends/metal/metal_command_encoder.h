//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#import <core/platform.h>
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
        auto address = reinterpret_cast<uint64_t>(command->data());
        auto size = command->size();
    
        auto page_size = pagesize();
        auto aligned_begin = address / page_size * page_size;
        auto aligned_end = (address + size + page_size - 1u) / page_size * page_size;
        
        LUISA_VERBOSE_WITH_LOCATION(
            "Aligned address 0x{:016x} with size {} bytes to [0x{:016x}, 0x{:016x}) (pagesize = {} bytes).",
            address, size, aligned_begin, aligned_end, page_size);
    
        auto t0 = std::chrono::high_resolution_clock::now();
        auto temporary = [_device->handle() newBufferWithBytesNoCopy:reinterpret_cast<void *>(aligned_begin)
                                                              length:aligned_end - aligned_begin
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nullptr];
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        LUISA_VERBOSE_WITH_LOCATION(
            "Allocated temporary shared buffer with size {} in {} ms.",
            size, (t1 - t0) / 1ns * 1e-6);

        auto blit_encoder = [_command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:buffer
                        sourceOffset:command->offset()
                            toBuffer:temporary
                   destinationOffset:aligned_begin - address
                                size:size];
        [blit_encoder endEncoding];
    }

    void visit(const KernelLaunchCommand *command) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Not implemented!");
    }
};

}
