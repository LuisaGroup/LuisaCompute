#include <core/logging.h>
#include <luisa_compute_ir/bindings.hpp>
#include <luisa_compute_api_types/bindings.hpp>

namespace luisa::compute::backend {
using namespace luisa::compute::api;
using luisa::compute::ir::CArc;
using luisa::compute::ir::KernelModule;
using luisa::compute::ir::Type;
}// namespace luisa::compute::backend

#include <luisa_compute_backend/bindings.hpp>
#include "rust_device_common.h"

namespace luisa::compute::rust {
using namespace luisa::compute::backend;
RustDevice::RustDevice(luisa::compute::Context &&ctx, std::string_view name) noexcept
    : DeviceInterface{std::move(ctx)} {
    _handle = lc_rs_create_backend(name.data());
    LUISA_INFO("RustDevice: Created device: {}", name);
}
BufferCreationInfo RustDevice::create_buffer(const luisa::compute::Type *element, size_t elem_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("create_buffer(const Type*, size_t) is deprecated.");
}
BufferCreationInfo RustDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    auto info = lc_rs_create_buffer(_handle, element, elem_count);
    BufferCreationInfo ret;
    ret.handle = info.resource.handle;
    ret.native_handle = info.resource.native_handle;
    ret.element_stride = info.element_stride;
    ret.total_size_bytes = info.total_size_bytes;
    return ret;
}
void RustDevice::destroy_buffer(uint64_t handle) noexcept {
    lc_rs_destroy_buffer(_handle, api::Buffer{handle});
}
ResourceCreationInfo RustDevice::create_texture(
    luisa::compute::PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    auto info = lc_rs_create_texture(_handle, static_cast<api::PixelFormat>(format), dimension, width, height, depth, mipmap_levels);
    ResourceCreationInfo ret;
    ret.handle = info.handle;
    ret.native_handle = info.native_handle;
    return ret;
}
void RustDevice::destroy_texture(uint64_t handle) noexcept {
    lc_rs_destroy_texture(_handle, api::Texture{handle});
}
ResourceCreationInfo RustDevice::create_bindless_array(size_t size) noexcept {
    auto info = lc_rs_create_bindless_array(_handle, size);
    ResourceCreationInfo ret;
    ret.handle = info.handle;
    ret.native_handle = info.native_handle;
    return ret;
}
void RustDevice::destroy_bindless_array(uint64_t handle) noexcept {
    lc_rs_destroy_bindless_array(_handle, api::BindlessArray{handle});
}

}// namespace luisa::compute::rust
