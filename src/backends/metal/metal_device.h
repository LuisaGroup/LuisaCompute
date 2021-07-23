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
#import <backends/metal/metal_shared_buffer_pool.h>
#import <backends/metal/metal_mesh.h>
#import <backends/metal/metal_accel.h>

namespace luisa::compute::metal {

class MetalDevice : public Device::Interface {

private:
    id<MTLDevice> _handle{nullptr};
    std::unique_ptr<MetalCompiler> _compiler{nullptr};

    std::unique_ptr<MetalSharedBufferPool> _argument_buffer_pool{nullptr};
    std::unique_ptr<MetalSharedBufferPool> _compacted_size_buffer_pool{nullptr};

    // for buffers
    std::vector<id<MTLBuffer>> _buffer_slots;
    std::vector<size_t> _available_buffer_slots;

    // for streams
    std::vector<std::unique_ptr<MetalStream>> _stream_slots;
    std::vector<size_t> _available_stream_slots;

    // for textures
    std::vector<id<MTLTexture>> _texture_slots;
    std::vector<size_t> _available_texture_slots;

    // for shaders
    std::vector<MetalShader> _shader_slots;
    std::vector<size_t> _available_shader_slots;

    // for heaps
    std::vector<std::unique_ptr<MetalTextureHeap>> _heap_slots;
    std::vector<size_t> _available_heap_slots;

    // for meshes
    std::vector<std::unique_ptr<MetalMesh>> _mesh_slots;
    std::vector<size_t> _available_mesh_slots;

    // for acceleration structures
    std::vector<std::unique_ptr<MetalAccel>> _accel_slots;
    std::vector<size_t> _available_accel_slots;

    // for events
    std::vector<std::unique_ptr<MetalEvent>> _event_slots;
    std::vector<size_t> _available_event_slots;

    // mutexes
    mutable spin_mutex _buffer_mutex;
    mutable spin_mutex _stream_mutex;
    mutable spin_mutex _texture_mutex;
    mutable spin_mutex _shader_mutex;
    mutable spin_mutex _heap_mutex;
    mutable spin_mutex _mesh_mutex;
    mutable spin_mutex _accel_mutex;
    mutable spin_mutex _event_mutex;

public:
    explicit MetalDevice(const Context &ctx, uint32_t index) noexcept;
    ~MetalDevice() noexcept override;
    [[nodiscard]] id<MTLDevice> handle() const noexcept;
    [[nodiscard]] id<MTLBuffer> buffer(uint64_t handle) const noexcept;
    [[nodiscard]] MetalStream *stream(uint64_t handle) const noexcept;
    [[nodiscard]] MetalEvent *event(uint64_t handle) const noexcept;
    [[nodiscard]] MetalMesh *mesh(uint64_t handle) const noexcept;
    [[nodiscard]] NSMutableArray<id<MTLAccelerationStructure>> *mesh_handles(std::span<const uint64_t> handles) noexcept;
    [[nodiscard]] MetalAccel *accel(uint64_t handle) const noexcept;
    [[nodiscard]] id<MTLTexture> texture(uint64_t handle) const noexcept;
    [[nodiscard]] MetalTextureHeap *heap(uint64_t handle) const noexcept;
    [[nodiscard]] MetalSharedBufferPool *argument_buffer_pool() const noexcept;
    [[nodiscard]] MetalShader compiled_kernel(uint64_t handle) const noexcept;
    [[nodiscard]] MetalSharedBufferPool *compacted_size_buffer_pool() const noexcept;

public:
    uint64_t create_texture(PixelFormat format, uint dimension,
                            uint width, uint height, uint depth, uint mipmap_levels,
                            TextureSampler sampler, uint64_t heap_handle, uint32_t index_in_heap) override;
    void destroy_texture(uint64_t handle) noexcept override;
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList buffer) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    uint64_t create_shader(Function kernel) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    uint64_t create_event() noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    uint64_t create_mesh() noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_accel() noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint64_t create_texture_heap(size_t size) noexcept override;
    size_t query_texture_heap_memory_usage(uint64_t handle) noexcept override;
    void destroy_texture_heap(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::metal
