//
// Created by Mike Smith on 2023/4/8.
//

#include <core/logging.h>
#include <backends/metal/metal_device.h>

namespace luisa::compute::metal {

void *MetalDevice::native_handle() const noexcept {
    return nullptr;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    LUISA_INFO("MetalDevice::create_buffer({}, {}) noexcept",
               element->description(), elem_count);
    return BufferCreationInfo();
}

BufferCreationInfo MetalDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    return BufferCreationInfo();
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_bindless_array(size_t size) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_bindless_array(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_stream(StreamTag stream_tag) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_stream(uint64_t handle) noexcept {
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
}

SwapChainCreationInfo MetalDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept {
    return SwapChainCreationInfo();
}

void MetalDevice::destroy_swap_chain(uint64_t handle) noexcept {
}

void MetalDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    return ShaderCreationInfo();
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    return ShaderCreationInfo();
}

ShaderCreationInfo MetalDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return ShaderCreationInfo();
}

Usage MetalDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return Usage::NONE;
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_event() noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_mesh(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_accel(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
}

string MetalDevice::query(luisa::string_view property) noexcept {
    return DeviceInterface::query(property);
}

DeviceExtension *MetalDevice::extension(luisa::string_view name) noexcept {
    return DeviceInterface::extension(name);
}

void MetalDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
}

}// namespace luisa::compute::metal
