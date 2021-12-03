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
#import <core/allocator.h>
#import <runtime/device.h>
#import <ast/function.h>
#import <backends/metal/metal_event.h>
#import <backends/metal/metal_stream.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_bindless_array.h>

#ifdef LUISA_METAL_RAYTRACING_ENABLED
#import <backends/metal/metal_mesh.h>
#import <backends/metal/metal_accel.h>
#endif

namespace luisa::compute::metal {

class MetalDevice final : public Device::Interface {

private:
    id<MTLDevice> _handle{nullptr};
    id<MTLArgumentEncoder> _bindless_array_encoder{nullptr};
    luisa::unique_ptr<MetalCompiler> _compiler{nullptr};

public:
    explicit MetalDevice(const Context &ctx, uint32_t index) noexcept;
    ~MetalDevice() noexcept override;
    [[nodiscard]] id<MTLDevice> handle() const noexcept;
    [[nodiscard]] MetalShader compiled_kernel(uint64_t handle) const noexcept;
    void check_raytracing_supported() const noexcept;

public:
    uint64_t create_texture(PixelFormat format, uint dimension,
                            uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList buffer) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    uint64_t create_event() noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    uint64_t create_mesh(
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint64_t create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    void *buffer_native_handle(uint64_t handle) const noexcept override;
    void *texture_native_handle(uint64_t handle) const noexcept override;
    void *native_handle() const noexcept override;
    void *stream_native_handle(uint64_t handle) const noexcept override;
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    [[nodiscard]] auto bindless_array_encoder() const noexcept { return _bindless_array_encoder; }
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform) noexcept override;
    void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept override;
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;
};

}// namespace luisa::compute::metal
