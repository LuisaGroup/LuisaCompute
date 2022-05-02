//
// Created by Mike Smith on 2021/3/19.
//

#import <core/platform.h>
#import <core/clock.h>
#import <ast/function.h>

#import <backends/metal/metal_mesh.h>
#import <backends/metal/metal_accel.h>
#import <backends/metal/metal_device.h>
#import <backends/metal/metal_stream.h>
#import <backends/metal/metal_bindless_array.h>
#import <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

MetalCommandEncoder::MetalCommandEncoder(
    MetalDevice *device,
    MetalStream *stream) noexcept
    : _device{device},
      _stream{stream},
      _command_buffer{stream->command_buffer()} {}

[[nodiscard]] inline static auto to_buffer(uint64_t handle) noexcept {
    return (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(handle));
}

[[nodiscard]] inline static auto to_texture(uint64_t handle) noexcept {
    return (__bridge id<MTLTexture>)(reinterpret_cast<void *>(handle));
}

[[nodiscard]] inline static auto to_bindless_array(uint64_t handle) noexcept {
    return reinterpret_cast<MetalBindlessArray *>(handle);
}

[[nodiscard]] inline static auto to_accel(uint64_t handle) noexcept {
    return reinterpret_cast<MetalAccel *>(handle);
}

[[nodiscard]] inline static auto to_mesh(uint64_t handle) noexcept {
    return reinterpret_cast<MetalMesh *>(handle);
}

void MetalCommandEncoder::visit(const BufferCopyCommand *command) noexcept {
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:to_buffer(command->src_handle())
                    sourceOffset:command->src_offset()
                        toBuffer:to_buffer(command->dst_handle())
               destinationOffset:command->dst_offset()
                            size:command->size()];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = to_buffer(command->handle());
    auto temp_buffer = _upload(command->data(), command->size());
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:temp_buffer.handle()
                    sourceOffset:temp_buffer.offset()
                        toBuffer:buffer
               destinationOffset:command->offset()
                            size:command->size()];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const BufferDownloadCommand *command) noexcept {
    auto buffer = to_buffer(command->handle());
    auto size = command->size();
    auto temp_buffer = _download(command->data(), size);
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer
                    sourceOffset:command->offset()
                        toBuffer:temp_buffer.handle()
               destinationOffset:temp_buffer.offset()
                            size:size];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto buffer = to_buffer(command->buffer());
    auto texture = to_texture(command->texture());
    auto size = command->size();
    auto pixel_bytes = pixel_storage_size(command->storage());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y;
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer
                    sourceOffset:command->buffer_offset()
               sourceBytesPerRow:pitch_bytes
             sourceBytesPerImage:image_bytes
                      sourceSize:MTLSizeMake(size.x, size.y, size.z)
                       toTexture:texture
                destinationSlice:0u
                destinationLevel:command->level()
               destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureCopyCommand *command) noexcept {
    auto src = to_texture(command->src_handle());
    auto dst = to_texture(command->dst_handle());
    auto size = command->size();
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:src
                      sourceSlice:0u
                      sourceLevel:command->src_level()
                     sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                       sourceSize:MTLSizeMake(size.x, size.y, size.z)
                        toTexture:dst
                 destinationSlice:0u
                 destinationLevel:command->dst_level()
                destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto buffer = to_buffer(command->buffer());
    auto texture = to_texture(command->texture());
    auto size = command->size();
    auto pixel_bytes = pixel_storage_size(command->storage());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y;
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:texture
                      sourceSlice:0u
                      sourceLevel:command->level()
                     sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                       sourceSize:MTLSizeMake(size.x, size.y, size.z)
                         toBuffer:buffer
                destinationOffset:command->buffer_offset()
           destinationBytesPerRow:pitch_bytes
         destinationBytesPerImage:image_bytes];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureUploadCommand *command) noexcept {
    auto size = command->size();
    auto pixel_bytes = pixel_storage_size(command->storage());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y;
    auto texture = to_texture(command->handle());
    auto buffer = _upload(command->data(), image_bytes * size.z);
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer.handle()
                    sourceOffset:buffer.offset()
               sourceBytesPerRow:pitch_bytes
             sourceBytesPerImage:image_bytes
                      sourceSize:MTLSizeMake(size.x, size.y, size.z)
                       toTexture:texture
                destinationSlice:0u
                destinationLevel:command->level()
               destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const TextureDownloadCommand *command) noexcept {
    auto size = command->size();
    auto pixel_bytes = pixel_storage_size(command->storage());
    auto pitch_bytes = pixel_bytes * size.x;
    auto image_bytes = pitch_bytes * size.y;
    auto texture = to_texture(command->handle());
    auto buffer = _download(command->data(), image_bytes * size.z);
    auto blit_encoder = [_command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:texture
                      sourceSlice:0u
                      sourceLevel:command->level()
                     sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                       sourceSize:MTLSizeMake(size.x, size.y, size.z)
                         toBuffer:buffer.handle()
                destinationOffset:buffer.offset()
           destinationBytesPerRow:pitch_bytes
         destinationBytesPerImage:image_bytes];
    [blit_encoder endEncoding];
}

void MetalCommandEncoder::visit(const ShaderDispatchCommand *command) noexcept {

    auto compiled_kernel = _device->compiled_kernel(command->handle());
    auto buffer_index = 0u;
    auto texture_index = 0u;

    auto launch_size = command->dispatch_size();
    auto block_size = command->kernel().block_size();
    auto blocks = (launch_size + block_size - 1u) / block_size;
    LUISA_VERBOSE_WITH_LOCATION(
        "Dispatching shader #{} in ({}, {}, {}) blocks "
        "with block_size ({}, {}, {}).",
        command->handle(),
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z);

    // encode compute shader
    auto compute_encoder = [_command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [compute_encoder setComputePipelineState:compiled_kernel.handle()];
    command->decode([&](auto argument) noexcept -> void {
        using T = decltype(argument);
        if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding buffer #{} at index {} with offset {}.",
                argument.handle, buffer_index, argument.offset);
            auto buffer = to_buffer(argument.handle);
            [compute_encoder setBuffer:buffer
                                offset:argument.offset
                               atIndex:buffer_index++];
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding texture #{} at index {}.",
                argument.handle, texture_index);
            id<MTLTexture> texture = to_texture(argument.handle);
            if (auto level = argument.level; level != 0u) {
                texture = [texture newTextureViewWithPixelFormat:[texture pixelFormat]
                                                     textureType:[texture textureType]
                                                          levels:NSMakeRange(level, 1u)
                                                          slices:NSMakeRange(0u, 1u)];
            }
            [compute_encoder setTexture:texture
                                atIndex:texture_index++];
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding texture heap #{} at index {}.",
                argument.handle, buffer_index);
            auto heap = to_bindless_array(argument.handle);
            heap->traverse_resources([&](auto &&res) noexcept {
                [compute_encoder useResource:res
                                       usage:MTLResourceUsageRead];
            });
            [compute_encoder setBuffer:heap->handle()
                                offset:0u
                               atIndex:buffer_index++];
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding geometry #{} at index {}.",
                argument.handle, buffer_index);
            auto accel = to_accel(argument.handle);
            for (auto resource : accel->resources()) {
                [compute_encoder useResource:resource.handle
                                       usage:MTLResourceUsageRead];
            }
            [compute_encoder setAccelerationStructure:accel->handle()
                                        atBufferIndex:buffer_index++];
            [compute_encoder setBuffer:accel->instance_buffer()
                                offset:0u
                               atIndex:buffer_index++];
        } else {// uniform
            static_assert(std::is_same_v<T, ShaderDispatchCommand::UniformArgument>);
            LUISA_VERBOSE_WITH_LOCATION("Encoding uniform at index {}.", buffer_index);
            [compute_encoder setBytes:argument.data
                               length:argument.size
                              atIndex:buffer_index++];
        }
    });

    LUISA_VERBOSE_WITH_LOCATION(
        "Encoding dispatch size at index {}.",
        buffer_index);
    [compute_encoder setBytes:&launch_size
                       length:sizeof(launch_size)
                      atIndex:buffer_index];
    [compute_encoder dispatchThreadgroups:MTLSizeMake(blocks.x, blocks.y, blocks.z)
                    threadsPerThreadgroup:MTLSizeMake(block_size.x, block_size.y, block_size.z)];
    [compute_encoder endEncoding];
}

MetalBufferView MetalCommandEncoder::_upload(const void *host_ptr, size_t size) noexcept {
    auto rb = &_stream->upload_host_buffer_pool();
    auto buffer = rb->allocate(size);
    std::memcpy(static_cast<std::byte *>(buffer.handle().contents) + buffer.offset(), host_ptr, size);
    if (buffer.is_pooled()) {
        [_command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          rb->recycle(buffer);
        }];
    }
    return buffer;
}

MetalBufferView MetalCommandEncoder::_download(void *host_ptr, size_t size) noexcept {
    auto rb = &_stream->download_host_buffer_pool();
    auto buffer = rb->allocate(size);
    [_command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      std::memcpy(host_ptr, static_cast<const std::byte *>(buffer.handle().contents) + buffer.offset(), size);
      if (buffer.is_pooled()) { rb->recycle(buffer); }
    }];
    return buffer;
}

void MetalCommandEncoder::visit(const BindlessArrayUpdateCommand *command) noexcept {
    auto array = to_bindless_array(command->handle());
    array->update(_stream, _command_buffer);
}

void MetalCommandEncoder::visit(const AccelBuildCommand *command) noexcept {
    auto accel = to_accel(command->handle());
    _command_buffer = accel->build(
        _stream, _command_buffer, command->instance_count(),
        command->request(), command->modifications());
}

void MetalCommandEncoder::visit(const MeshBuildCommand *command) noexcept {
    auto mesh = to_mesh(command->handle());
    _command_buffer = mesh->build(_stream, _command_buffer, command->request());
}

}
