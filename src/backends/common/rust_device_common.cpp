#include <core/logging.h>
#include <luisa_compute_ir/bindings.hpp>
#include <luisa_compute_api_types/bindings.hpp>

namespace luisa::compute::backend {
using namespace luisa::compute::api;
using luisa::compute::ir::CArc;
using luisa::compute::ir::KernelModule;
using luisa::compute::ir::Type;
}// namespace luisa::compute::backend

#include "rust_device_common.h"

// must go last to avoid name conflicts
#include <runtime/rhi/resource.h>

namespace luisa::compute::rust {
//using namespace luisa::compute::backend;
//RustDevice::RustDevice(luisa::compute::Context &&ctx, std::string_view name) noexcept
//    : DeviceInterface{std::move(ctx)} {
//    _handle = lc_rs_create_backend(name.data());
//    LUISA_INFO("RustDevice: Created device: {}", name);
//}
//BufferCreationInfo RustDevice::create_buffer(const luisa::compute::Type *element, size_t elem_count) noexcept {
//    LUISA_ERROR_WITH_LOCATION("create_buffer(const Type*, size_t) is deprecated.");
//}
//BufferCreationInfo RustDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
//    auto info = lc_rs_create_buffer(_handle, element, elem_count);
//    BufferCreationInfo ret;
//    ret.handle = info.resource.handle;
//    ret.native_handle = info.resource.native_handle;
//    ret.element_stride = info.element_stride;
//    ret.total_size_bytes = info.total_size_bytes;
//    return ret;
//}
//void RustDevice::destroy_buffer(uint64_t handle) noexcept {
//    lc_rs_destroy_buffer(_handle, api::Buffer{handle});
//}
//ResourceCreationInfo RustDevice::create_texture(
//    luisa::compute::PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
//    auto info = lc_rs_create_texture(_handle, static_cast<api::PixelFormat>(format), dimension, width, height, depth, mipmap_levels);
//    ResourceCreationInfo ret;
//    ret.handle = info.handle;
//    ret.native_handle = info.native_handle;
//    return ret;
//}
//void RustDevice::destroy_texture(uint64_t handle) noexcept {
//    lc_rs_destroy_texture(_handle, api::Texture{handle});
//}
//ResourceCreationInfo RustDevice::create_bindless_array(size_t size) noexcept {
//    auto info = lc_rs_create_bindless_array(_handle, size);
//    ResourceCreationInfo ret;
//    ret.handle = info.handle;
//    ret.native_handle = info.native_handle;
//    return ret;
//}
//void RustDevice::destroy_bindless_array(uint64_t handle) noexcept {
//    lc_rs_destroy_bindless_array(_handle, api::BindlessArray{handle});
//}
//
//// TODO: Implement these functions
//ResourceCreationInfo RustDevice::create_stream(luisa::compute::StreamTag stream_tag) noexcept {
//    return ResourceCreationInfo();
//}
//void RustDevice::destroy_stream(uint64_t handle) noexcept {
//}
//void RustDevice::synchronize_stream(uint64_t stream_handle) noexcept {
//}
//void RustDevice::dispatch(uint64_t stream_handle, luisa::compute::CommandList &&list) noexcept {
//}
//SwapChainCreationInfo RustDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept {
//    return SwapChainCreationInfo();
//}
//void RustDevice::destroy_swap_chain(uint64_t handle) noexcept {
//}
//void RustDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
//}
//ShaderCreationInfo RustDevice::create_shader(const luisa::compute::ShaderOption &option, Function kernel) noexcept {
//    return ShaderCreationInfo();
//}
//ShaderCreationInfo RustDevice::create_shader(const luisa::compute::ShaderOption &option, const ir::KernelModule *kernel) noexcept {
//    return ShaderCreationInfo();
//}
//ShaderCreationInfo RustDevice::load_shader(luisa::string_view name, luisa::span<const luisa::compute::Type *const> arg_types) noexcept {
//    return ShaderCreationInfo();
//}
//void RustDevice::destroy_shader(uint64_t handle) noexcept {
//}
//ResourceCreationInfo RustDevice::create_event() noexcept {
//    return ResourceCreationInfo();
//}
//void RustDevice::destroy_event(uint64_t handle) noexcept {
//}
//void RustDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
//}
//void RustDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
//}
//void RustDevice::synchronize_event(uint64_t handle) noexcept {
//}
//ResourceCreationInfo RustDevice::create_mesh(const luisa::compute::AccelOption &option) noexcept {
//    return ResourceCreationInfo();
//}
//void RustDevice::destroy_mesh(uint64_t handle) noexcept {
//}
//ResourceCreationInfo RustDevice::create_procedural_primitive(const luisa::compute::AccelOption &option) noexcept {
//    return ResourceCreationInfo();
//}
//void RustDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
//}
//ResourceCreationInfo RustDevice::create_accel(const luisa::compute::AccelOption &option) noexcept {
//    return ResourceCreationInfo();
//}
//void RustDevice::destroy_accel(uint64_t handle) noexcept {
//}
//string RustDevice::query(luisa::string_view property) noexcept {
//    return DeviceInterface::query(property);
//}
//DeviceExtension *RustDevice::extension(luisa::string_view name) noexcept {
//    return DeviceInterface::extension(name);
//}
//Usage RustDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
//    return Usage::NONE;
//}
//void RustDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
//}

}// namespace luisa::compute::rust
