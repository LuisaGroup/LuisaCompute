//
// Created by Mike Smith on 2021/10/17.
//

#pragma once

#include <api/common.h>
#include <api/device.h>


LUISA_EXPORT_API void luisa_compute_init() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCAppContext luisa_compute_app_context() LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_set_app_context(LCAppContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCContext luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_add_search_path(LCContext ctx, const char *path) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(LCContext ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx, const char *name, const char *properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_retain(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_release(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void * luisa_compute_device_native_handle(LCDevice device) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedBufferInfo luisa_compute_buffer_create(LCDevice device, const void *element, size_t elem_count) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void * luisa_compute_buffer_native_handle(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_texture_create(LCDevice device, LCPixelFormat format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_stream_create(LCDevice device, LCStreamTag stream_tag) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList cmd_list) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_native_handle(LCDevice device, LCStream stream, void *handle) LUISA_NOEXCEPT;


LUISA_EXPORT_API LCCreatedShaderInfo luisa_compute_shader_create(LCDevice device, LCKernelModule func, const LCShaderOption *option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCDevice device, LCBindlessArray array) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_buffer(LCDevice device, LCBindlessArray array, size_t index, LCBuffer buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex2d(LCDevice device, LCBindlessArray array, size_t index, LCTexture texture, LCSampler sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex3d(LCDevice device, LCBindlessArray array, size_t index, LCTexture texture, LCSampler sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_buffer(LCDevice device, LCBindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex2d(LCDevice device, LCBindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex3d(LCDevice device, LCBindlessArray array, size_t index) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_mesh_create(LCDevice device, const LCAccelOption *option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCDevice device, LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_accel_create(LCDevice device, const LCAccelOption *option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace_back(LCAccel accel, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace(LCAccel accel, size_t index, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_transform(LCAccel accel, size_t index, const void *transform) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_visibility(LCAccel accel, size_t index, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_pop_back(LCAccel accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCPixelStorage luisa_compute_pixel_format_to_storage(LCPixelFormat format) LUISA_NOEXCEPT;
