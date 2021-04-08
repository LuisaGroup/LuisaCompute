//
// Created by Mike Smith on 2021/3/19.
//

#import <core/clock.h>
#import <core/platform.h>
#import <ast/function.h>
#import <backends/metal/metal_command_encoder.h>

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

    Clock clock;
    auto temporary = [_device->handle() newBufferWithBytes:command->data()
                                                    length:command->size()
                                                   options:MTLResourceStorageModeShared];

    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated temporary shared buffer with size {} in {} ms.",
        command->size(), clock.toc());

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
    auto [temporary, offset] = _wrap_output_buffer(command->data(), size);

    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer
                    sourceOffset:command->offset()
                        toBuffer:temporary
               destinationOffset:offset
                            size:size];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureUploadCommand *command) noexcept {

    auto offset = command->offset();
    auto size = command->size();
    auto pixel_bytes = pixel_format_size(command->format());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y * size.z;

    auto temporary = _allocate_input_buffer(command->data(), image_bytes);
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:temporary
                    sourceOffset:0u
               sourceBytesPerRow:pitch_bytes
             sourceBytesPerImage:image_bytes
                      sourceSize:MTLSizeMake(size.x, size.y, size.z)
                       toTexture:_device->texture(command->handle())
                destinationSlice:0u
                destinationLevel:command->level()
               destinationOrigin:MTLOriginMake(offset.x, offset.y, offset.z)];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureDownloadCommand *command) noexcept {

    auto offset = command->offset();
    auto size = command->size();
    auto pixel_bytes = pixel_format_size(command->format());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y * size.z;
    auto texture = _device->texture(command->handle());
    auto [buffer, buffer_offset] = _wrap_output_buffer(command->data(), image_bytes);
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:texture
                      sourceSlice:0u
                      sourceLevel:command->level()
                     sourceOrigin:MTLOriginMake(offset.x, offset.y, offset.z)
                       sourceSize:MTLSizeMake(size.x, size.y, size.z)
                         toBuffer:buffer
                destinationOffset:buffer_offset
           destinationBytesPerRow:pitch_bytes
         destinationBytesPerImage:image_bytes];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const KernelLaunchCommand *command) noexcept {

    auto function = Function::kernel(command->kernel_uid());
    auto kernel = _device->kernel(command->kernel_uid());
    auto argument_index = 0u;

    auto launch_size = command->launch_size();
    auto block_size = function.block_size();
    auto blocks = (launch_size + block_size - 1u) / block_size;
    LUISA_VERBOSE_WITH_LOCATION(
        "Dispatch kernel #{} in ({}, {}, {}) blocks with block_size ({}, {}, {}).",
        command->kernel_uid(),
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z);

    auto argument_encoder = kernel.encoder;
    auto argument_buffer_pool = _device->argument_buffer_pool();
    auto argument_buffer = argument_buffer_pool->allocate();
    auto compute_encoder = [_command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [compute_encoder setComputePipelineState:kernel.handle];
    [argument_encoder setArgumentBuffer:argument_buffer.handle() offset:argument_buffer.offset()];
    command->decode([&](auto argument) noexcept {
        using T = decltype(argument);
        auto mark_usage = [compute_encoder](id<MTLResource> res, auto usage) noexcept {
            switch (usage) {
                case Command::Resource::Usage::READ:
                    [compute_encoder useResource:res usage:MTLResourceUsageRead];
                    break;
                case Command::Resource::Usage::WRITE:
                    [compute_encoder useResource:res usage:MTLResourceUsageWrite];
                    break;
                case Command::Resource::Usage::READ_WRITE:
                    [compute_encoder useResource:res usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                default: break;
            }
        };
        if constexpr (std::is_same_v<T, KernelLaunchCommand::BufferArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding buffer #{} at index {} with offset {}.",
                argument.handle, argument_index, argument.offset);
            auto buffer = _device->buffer(argument.handle);
            [argument_encoder setBuffer:buffer
                                 offset:argument.offset
                                atIndex:kernel.arguments[argument_index++].argumentIndex];
            mark_usage(buffer, argument.usage);
        } else if constexpr (std::is_same_v<T, KernelLaunchCommand::TextureArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding texture #{} at index {}.",
                argument.handle, argument_index);
            auto texture = _device->texture(argument.handle);
            [argument_encoder setTexture:texture
                                 atIndex:kernel.arguments[argument_index++].argumentIndex];
            mark_usage(texture, argument.usage);
        } else {// uniform
            auto ptr = [argument_encoder constantDataAtIndex:kernel.arguments[argument_index++].argumentIndex];
            std::memcpy(ptr, argument.data(), argument.size_bytes());
        }
    });
    auto ptr = [argument_encoder constantDataAtIndex:kernel.arguments[argument_index].argumentIndex];
    std::memcpy(ptr, &launch_size, sizeof(launch_size));
    [compute_encoder setBuffer:argument_buffer.handle() offset:argument_buffer.offset() atIndex:0];
    [compute_encoder dispatchThreadgroups:MTLSizeMake(blocks.x, blocks.y, blocks.z)
                    threadsPerThreadgroup:MTLSizeMake(block_size.x, block_size.y, block_size.z)];
    [compute_encoder endEncoding];

    [_command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      auto arg_buffer = argument_buffer;
      argument_buffer_pool->recycle(arg_buffer);
    }];
}

inline std::pair<id<MTLBuffer>, size_t> MetalCommandEncoder::_wrap_output_buffer(void *data, size_t size) noexcept {

    auto address = reinterpret_cast<uint64_t>(data);
    auto page_size = pagesize();
    auto aligned_begin = address / page_size * page_size;
    auto aligned_end = (address + size + page_size - 1u) / page_size * page_size;

    Clock clock;
    auto temporary = [_device->handle() newBufferWithBytesNoCopy:reinterpret_cast<void *>(aligned_begin)
                                                          length:aligned_end - aligned_begin
                                                         options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                                     deallocator:nullptr];
    LUISA_VERBOSE_WITH_LOCATION(
        "Wrapped receiver pointer into temporary shared buffer with size {} in {} ms.",
        size, clock.toc());

    return {temporary, aligned_begin - address};
}

id<MTLBuffer> MetalCommandEncoder::_allocate_input_buffer(const void *data, size_t size) noexcept {
    Clock clock;
    auto temporary = [_device->handle() newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];

    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated temporary shared buffer with size {} in {} ms.",
        size, clock.toc());
    return temporary;
}

void MetalCommandEncoder::visit(const EventSignalCommand *command) noexcept {
    _device->event(command->handle()).signal(_command_buffer);
}

void MetalCommandEncoder::visit(const EventWaitCommand *command) noexcept {
    _device->event(command->handle()).wait(_command_buffer);
}

}
