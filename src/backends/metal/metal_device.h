//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#import <vector>
#import <thread>
#import <future>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>
#import <core/stl.h>
#import <runtime/device.h>
#import <ast/function.h>
#import <backends/metal/metal_event.h>
#import <backends/metal/metal_stream.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_bindless_array.h>
#import <backends/metal/metal_mesh.h>
#import <backends/metal/metal_accel.h>

namespace luisa::compute::metal {

class MetalDevice final : public Device::Interface {

private:
    id<MTLDevice> _handle{nullptr};
    id<MTLArgumentEncoder> _bindless_array_encoder{nullptr};
    id<MTLComputePipelineState> _update_instances_shader{nullptr};
    id<MTLRenderPipelineState> _render_shader{nullptr};
    luisa::unique_ptr<MetalCompiler> _compiler{nullptr};

public:
    explicit MetalDevice(const Context &ctx, uint32_t index) noexcept;
    ~MetalDevice() noexcept override;
    [[nodiscard]] id<MTLDevice> handle() const noexcept;
    [[nodiscard]] MetalShader compiled_kernel(uint64_t handle) const noexcept;
    void check_raytracing_supported() const noexcept;
    [[nodiscard]] auto bindless_array_encoder() const noexcept { return _bindless_array_encoder; }
    [[nodiscard]] auto instance_update_shader() const noexcept { return _update_instances_shader; }
    [[nodiscard]] auto present_shader() const noexcept { return _render_shader; }

public:
    uint64_t create_texture(PixelFormat format, uint dimension,
                            uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    uint64_t create_stream(bool for_present) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void dispatch(uint64_t stream_handle, const CommandList &list) noexcept override;
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
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_accel(AccelUsageHint hint) noexcept override;
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
    bool is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    bool requires_command_reordering() const noexcept override;
    void dispatch(uint64_t stream_handle, luisa::move_only_function<void()> &&func) noexcept override;
    uint64_t create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    void dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept override;
};

}// namespace luisa::compute::metal
