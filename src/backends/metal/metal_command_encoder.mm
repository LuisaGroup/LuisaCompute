//
// Created by Mike Smith on 2021/3/19.
//

#import <core/platform.h>
#include <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

MetalCommandEncoder::MetalCommandEncoder(MetalDevice *device, id<MTLCommandBuffer> cb) noexcept
    : _device{device}, _command_buffer{cb} {}

void MetalCommandEncoder::visit(const BufferCopyCommand *command) noexcept {
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_device->buffer(command->src_handle())
                    sourceOffset:command->src_offset()
                        toBuffer:_device->buffer(command->dst_handle())
               destinationOffset:command->dst_offset()
                            size:command->size()];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const BufferUploadCommand *command) noexcept {

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

void MetalCommandEncoder::visit(const BufferDownloadCommand *command) noexcept {

    auto buffer = _device->buffer(command->handle());

    auto address = reinterpret_cast<uint64_t>(command->data());
    auto size = command->size();

    auto page_size = pagesize();
    auto aligned_begin = address / page_size * page_size;
    auto aligned_end = (address + size + page_size - 1u) / page_size * page_size;

    LUISA_VERBOSE_WITH_LOCATION(
        "Aligned address 0x{:012x} with size {} bytes to [0x{:012x}, 0x{:012x}) (pagesize = {} bytes).",
        address, size, aligned_begin, aligned_end, page_size);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto temporary = [_device->handle() newBufferWithBytesNoCopy:reinterpret_cast<void *>(aligned_begin)
                                                          length:aligned_end - aligned_begin
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nullptr];
    auto t1 = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;
    LUISA_VERBOSE_WITH_LOCATION(
        "Wrapped receiver pointer into temporary shared buffer with size {} in {} ms.",
        size, (t1 - t0) / 1ns * 1e-6);

    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer
                    sourceOffset:command->offset()
                        toBuffer:temporary
               destinationOffset:aligned_begin - address
                            size:size];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const KernelLaunchCommand *command) noexcept {

    auto kernel = _device->kernel(command->kernel_uid());
    auto buffer_count = 0u;
    auto texture_count = 0u;

    auto launch_size = command->launch_size();
    auto block_size = command->block_size();
    auto blocks = (launch_size + block_size - 1u) / block_size;
    LUISA_VERBOSE_WITH_LOCATION(
        "Dispatch kernel #{} in ({}, {}, {}) blocks with block_size ({}, {}, {}).",
        command->kernel_uid(),
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z);

    auto compute_encoder = [_command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [compute_encoder setComputePipelineState:kernel];
    command->decode([&](auto argument) noexcept {
        using T = decltype(argument);
        if constexpr (std::is_same_v<T, KernelLaunchCommand::BufferArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding buffer #{} at index {} with offset {}.",
                argument.handle, buffer_count, argument.offset);
            auto buffer = _device->buffer(argument.handle);
            [compute_encoder setBuffer:buffer offset:argument.offset atIndex:buffer_count++];
//            auto usage = [](Command::Resource::Usage u) noexcept -> NSUInteger {
//                switch (u) {
//                    case Command::Resource::Usage::READ: return MTLResourceUsageRead;
//                    case Command::Resource::Usage::WRITE: return MTLResourceUsageWrite;
//                    case Command::Resource::Usage::READ_WRITE: return MTLResourceUsageRead | MTLResourceUsageWrite;
//                    default: return 0u;
//                }
//            }(argument.usage);
//            [compute_encoder useResource:buffer usage:usage];
        } else if constexpr (std::is_same_v<T, KernelLaunchCommand::TextureArgument>) {
            LUISA_ERROR_WITH_LOCATION("Not implemented.");
        } else {// uniform
            [compute_encoder setBytes:argument.data() length:argument.size_bytes() atIndex:buffer_count++];
        }
    });
    [compute_encoder setBytes:&launch_size length:sizeof(launch_size) atIndex:buffer_count];
    [compute_encoder dispatchThreadgroups:MTLSizeMake(blocks.x, blocks.y, blocks.z)
                    threadsPerThreadgroup:MTLSizeMake(block_size.x, block_size.y, block_size.z)];
    [compute_encoder endEncoding];
}

}
