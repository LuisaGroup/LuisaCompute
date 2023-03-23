//
// Created by Mike Smith on 2021/10/17.
//

#pragma once

#include "luisa_compute_api_types/bindings.hpp"
#include <api/common.h>
#include <api/device.h>

namespace api = luisa::compute::api;
namespace ir = luisa::compute::ir;

LUISA_EXPORT_API void luisa_compute_init() LUISA_NOEXCEPT;
LUISA_EXPORT_API api::AppContext luisa_compute_app_context() LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_set_app_context(api::AppContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::Context luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(api::Context ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_add_search_path(api::Context ctx, const char *path) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(api::Context ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(api::Context ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::Device luisa_compute_device_create(api::Context ctx, const char *name, const char *properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(api::Device device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_retain(api::Device device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_release(api::Device device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void * luisa_compute_device_native_handle(api::Device device) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedBufferInfo luisa_compute_buffer_create(api::Device device, const ir::CArc<ir::Type> *element, size_t elem_count) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(api::Device device, api::Buffer buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void * luisa_compute_buffer_native_handle(api::Device device, api::Buffer buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_texture_create(api::Device device, api::PixelFormat format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(api::Device device, api::Texture texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_stream_create(api::Device device, api::StreamTag stream_tag) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(api::Device device, api::Stream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(api::Device device, api::Stream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(api::Device device, api::Stream stream, api::CommandList cmd_list) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_native_handle(api::Device device, api::Stream stream, void *handle) LUISA_NOEXCEPT;


LUISA_EXPORT_API api::CreatedShaderInfo luisa_compute_shader_create(api::Device device, api::KernelModule func, const api::ShaderOption &option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(api::Device device, api::Shader shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_event_create(api::Device device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(api::Device device, api::Event event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(api::Device device, api::Event event, api::Stream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(api::Device device, api::Event event, api::Stream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(api::Device device, api::Event event) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_bindless_array_create(api::Device device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(api::Device device, api::BindlessArray array) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_buffer(api::Device device, api::BindlessArray array, size_t index, api::Buffer buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex2d(api::Device device, api::BindlessArray array, size_t index, api::Texture texture, api::Sampler sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex3d(api::Device device, api::BindlessArray array, size_t index, api::Texture texture, api::Sampler sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_buffer(api::Device device, api::BindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex2d(api::Device device, api::BindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex3d(api::Device device, api::BindlessArray array, size_t index) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_mesh_create(api::Device device, const api::AccelOption &option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(api::Device device, api::Mesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_accel_create(api::Device device, const api::AccelOption &option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(api::Device device, api::Accel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace_back(api::Accel accel, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace(api::Accel accel, size_t index, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_transform(api::Accel accel, size_t index, const void *transform) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_visibility(api::Accel accel, size_t index, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_pop_back(api::Accel accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API api::PixelStorage luisa_compute_pixel_format_to_storage(api::PixelFormat format) LUISA_NOEXCEPT;
