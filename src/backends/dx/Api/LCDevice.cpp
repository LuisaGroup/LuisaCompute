#pragma vengine_package vengine_directx
#include <Api/LCDevice.h>
#include <DXRuntime/Device.h>
namespace toolhub::directx {
LCDevice::LCDevice() {
}
void *LCDevice::native_handle() const noexcept {
    return nullptr;
}
uint64_t LCDevice::create_buffer(size_t size_bytes) noexcept {
    return uint64_t();
}
void LCDevice::destroy_buffer(uint64_t handle) noexcept {
}
void *LCDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}
uint64_t LCDevice::create_texture(
    PixelFormat format, 
    uint dimension, 
    uint width,
    uint height, 
    uint depth, 
    uint mipmap_levels) noexcept {
    return uint64_t();
}
void LCDevice::destroy_texture(uint64_t handle) noexcept {
}
void *LCDevice::texture_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}
uint64_t LCDevice::create_bindless_array(size_t size) noexcept {
    return uint64_t();
}
void LCDevice::destroy_bindless_array(uint64_t handle) noexcept {
}
void LCDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
}
void LCDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
void LCDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
bool LCDevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return false;
}
bool LCDevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return false;
}
void LCDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void LCDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void LCDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
uint64_t LCDevice::create_stream() noexcept {
    return uint64_t();
}
void LCDevice::destroy_stream(uint64_t handle) noexcept {
}
void LCDevice::synchronize_stream(uint64_t stream_handle) noexcept {
}
void LCDevice::dispatch(uint64_t stream_handle, CommandList) noexcept {
}
void *LCDevice::stream_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}
uint64_t LCDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    return uint64_t();
}
void LCDevice::destroy_shader(uint64_t handle) noexcept {
}
uint64_t LCDevice::create_event() noexcept {
    return uint64_t();
}
void LCDevice::destroy_event(uint64_t handle) noexcept {
}
void LCDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void LCDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void LCDevice::synchronize_event(uint64_t handle) noexcept {
}
uint64_t LCDevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept {
    return uint64_t();
}
void LCDevice::destroy_mesh(uint64_t handle) noexcept {
}
uint64_t LCDevice::create_accel(AccelBuildHint hint) noexcept {
    return uint64_t();
}
void LCDevice::emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept {
}
void LCDevice::pop_back_instance_from_accel(uint64_t accel) noexcept {
}
void LCDevice::set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept {
}
void LCDevice::set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept {
}
void LCDevice::set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept {
}
bool LCDevice::is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept {
    return false;
}
bool LCDevice::is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept {
    return false;
}
uint64_t LCDevice::get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return uint64_t();
}
uint64_t LCDevice::get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return uint64_t();
}
void LCDevice::destroy_accel(uint64_t handle) noexcept {
}
}// namespace toolhub::directx