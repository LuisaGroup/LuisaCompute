//
// Created by Mike Smith on 2021/10/17.
//

#pragma once

#include <core/platform.h>

LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(void *ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(void *ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(void *ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_device_create(void *ctx, const char *name, const char *properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(void *ctx, void *device) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_buffer_create(void *device, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(void *buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint32_t luisa_compute_pixel_format_to_storage(uint32_t format) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_texture_create(void *device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(void *texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_stream_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(void *stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(void *stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(void *stream, void *cmd_list) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_shader_create(void *device, const void *function, const char *options) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(void *shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_event_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(void *event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(void *event, void *stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(void *event, void *stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(void *event) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_bindless_array_create(void *device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(void *array) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_buffer(void *array, size_t index, void *buffer) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex2d(void *array, size_t index, void *texture, uint32_t sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_emplace_tex3d(void *array, size_t index, void *texture, uint32_t sampler) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_buffer(void *array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex2d(void *array, size_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_remove_tex3d(void *array, size_t index) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_mesh_create(
    void *device,
    void *v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    void *t_buffer, size_t t_offset, size_t t_count, uint32_t hint) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(void *mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_accel_create(void *device, uint32_t hint) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(void *accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace_back(void *accel, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace(void *accel, size_t index, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_transform(void *accel, size_t index, const void *transform) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_visibility(void *accel, size_t index, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_pop_back(void *accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_command_list_create() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_list_append(void *list, void *command) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_command_list_empty(void *list) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_command_upload_buffer(
    void *buffer, size_t offset, size_t size, const void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_download_buffer(
    void *buffer, size_t offset, size_t size, void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_copy_buffer_to_buffer(
    void *src, size_t src_offset,
    void *dst, size_t dst_offset, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_copy_buffer_to_texture(
    void *buffer, size_t buffer_offset,
    void *tex, uint32_t tex_storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_copy_texture_to_buffer(
    uint64_t buffer, size_t buffer_offset,
    uint64_t tex, uint32_t tex_storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_copy_texture_to_texture(
    uint64_t src, uint32_t src_level, uint32_t src_offset_x, uint32_t src_offset_y, uint32_t src_offset_z,
    uint64_t dst, uint32_t dst_level, uint32_t dst_offset_x, uint32_t dst_offset_y, uint32_t dst_offset_z,
    uint32_t storage, uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_upload_texture(
    uint64_t handle, uint32_t storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z,
    const void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_download_texture(
    uint64_t handle, uint32_t storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z,
    void *data) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_command_dispatch_shader(uint64_t handle, const void *kernel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_set_size(void *cmd, uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_buffer(
    void *cmd, uint32_t vid, uint64_t buffer, size_t offset, uint32_t usage) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_texture(
    void *cmd, uint32_t vid, uint64_t tex, uint32_t level, uint32_t usage) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_uniform(
    void *cmd, uint32_t vid,
    const void *data, size_t size, size_t alignment) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_bindless_array(void *cmd, uint32_t vid, uint64_t heap) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_accel(void *cmd, uint32_t vid, uint64_t accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_command_build_mesh(uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_update_mesh(uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_build_accel(uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_command_update_accel(uint64_t handle) LUISA_NOEXCEPT;
