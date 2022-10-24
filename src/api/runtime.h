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
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(LCContext ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx, const char *name, const char *properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_retain(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_release(LCDevice device) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCBuffer luisa_compute_buffer_create(LCDevice device, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCTexture luisa_compute_texture_create(LCDevice device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCStream luisa_compute_stream_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList cmd_list) LUISA_NOEXCEPT;


LUISA_EXPORT_API LCShader luisa_compute_shader_create(LCDevice device, const LCKernelModule *func, const char *options) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCEvent luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCBindlessArray luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCDevice device, LCBindlessArray array) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_buffer(LCBindlessArray array, size_t index, LCBuffer buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex2d(LCBindlessArray array, size_t index, LCTexture texture, uint32_t sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex3d(LCBindlessArray array, size_t index, LCTexture texture, uint32_t sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_buffer(LCBindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex2d(LCBindlessArray array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex3d(LCBindlessArray array, size_t index) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCMesh luisa_compute_mesh_create(
    LCDevice device,
    LCBuffer v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    LCBuffer t_buffer, size_t t_offset, size_t t_count, LCAccelUsageHint hint) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCDevice device, LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCAccel luisa_compute_accel_create(LCDevice device, LCAccelUsageHint hint) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace_back(LCAccel accel, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace(LCAccel accel, size_t index, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_transform(LCAccel accel, size_t index, const void *transform) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_visibility(LCAccel accel, size_t index, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_pop_back(LCAccel accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCPixelStorage luisa_compute_pixel_format_to_storage(LCPixelFormat format) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCBindlessArray luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT;