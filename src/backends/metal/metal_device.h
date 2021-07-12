//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#import <vector>
#import <thread>
#import <future>
#import <unordered_map>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>
#import <runtime/device.h>
#import <ast/function.h>
#import <backends/metal/metal_event.h>
#import <backends/metal/metal_stream.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_texture_heap.h>
#import <backends/metal/metal_argument_buffer_pool.h>

namespace luisa::compute::metal {

class MetalDevice : public Device::Interface {

private:
    id<MTLDevice> _handle{nullptr};
    std::unique_ptr<MetalCompiler> _compiler{nullptr};
    std::unique_ptr<MetalArgumentBufferPool> _argument_buffer_pool{nullptr};

    // for buffers
    mutable spin_mutex _buffer_mutex;
    std::vector<id<MTLBuffer>> _buffer_slots;
    std::vector<size_t> _available_buffer_slots;

    // for streams
    mutable spin_mutex _stream_mutex;
    std::vector<std::unique_ptr<MetalStream>> _stream_slots;
    std::vector<size_t> _available_stream_slots;

    // for textures
    mutable spin_mutex _texture_mutex;
    std::vector<id<MTLTexture>> _texture_slots;
    std::vector<size_t> _available_texture_slots;

    // for shaders
    mutable spin_mutex _shader_mutex;
    std::vector<MetalShader> _shader_slots;
    std::vector<size_t> _available_shader_slots;

    // for heaps
    mutable spin_mutex _heap_mutex;
    std::vector<std::unique_ptr<MetalTextureHeap>> _heap_slots;
    std::vector<size_t> _available_heap_slots;

    // for texture samplers
    spin_mutex _texture_sampler_mutex;
    std::unordered_map<TextureSampler, id<MTLSamplerState>, TextureSampler::Hash> _texture_samplers;

    // for events
    mutable spin_mutex _event_mutex;
    std::vector<std::unique_ptr<MetalEvent>> _event_slots;
    std::vector<size_t> _available_event_slots;

public:
    explicit MetalDevice(const Context &ctx, uint32_t index) noexcept;
    ~MetalDevice() noexcept override;
    [[nodiscard]] id<MTLDevice> handle() const noexcept;
    [[nodiscard]] id<MTLBuffer> buffer(uint64_t handle) const noexcept;
    [[nodiscard]] MetalStream *stream(uint64_t handle) const noexcept;
    [[nodiscard]] MetalEvent *event(uint64_t handle) const noexcept;
    [[nodiscard]] id<MTLTexture> texture(uint64_t handle) const noexcept;
    [[nodiscard]] MetalTextureHeap *heap(uint64_t handle) const noexcept;
    [[nodiscard]] MetalArgumentBufferPool *argument_buffer_pool() const noexcept;
    [[nodiscard]] MetalShader compiled_kernel(uint64_t handle) const noexcept;
    [[nodiscard]] id<MTLSamplerState> texture_sampler(TextureSampler sampler) noexcept;

public:
    uint64_t create_texture(PixelFormat format, uint dimension,
                            uint width, uint height, uint depth, uint mipmap_levels,
                            TextureSampler sampler, uint64_t heap_handle, uint32_t index_in_heap) override;
    void destroy_texture(uint64_t handle) noexcept override;
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandBuffer buffer) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    uint64_t create_shader(Function kernel) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    uint64_t create_event() noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    uint64_t create_mesh(uint64_t stream_handle,
                         uint64_t vertex_buffer_handle, size_t vertex_buffer_offset_bytes, size_t vertex_count,
                         uint64_t index_buffer_handle, size_t index_buffer_offset_bytes, size_t triangle_count) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_accel(uint64_t stream_handle,
                          uint64_t mesh_handle_buffer_handle, size_t mesh_handle_buffer_offset_bytes,
                          uint64_t transform_buffer_handle, size_t transform_buffer_offset_bytes,
                          size_t mesh_count) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint64_t create_texture_heap(size_t size) noexcept override;
    size_t query_texture_heap_memory_usage(uint64_t handle) noexcept override;
    void destroy_texture_heap(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::metal
