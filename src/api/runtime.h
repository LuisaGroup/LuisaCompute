//
// Created by Mike Smith on 2021/10/17.
//

#pragma once

#include <core/platform.h>

LUISA_EXPORT_API void *luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(void *ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(void *ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(void *ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_device_create(void *ctx, const char *name, uint32_t index) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(void *device) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_buffer_create(void *device, size_t size, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(void *device, uint64_t handle, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_texture_create(void *device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, uint32_t sampler, uint64_t heap, uint32_t index_in_heap) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(void *device, uint64_t handle, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_heap_create(void *device, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_heap_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_stream_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(void *device, uint64_t handle, void *cmd_list) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_shader_create(void *device, const void *function) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_event_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint64_t luisa_compute_mesh_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;
LUISA_EXPORT_API uint64_t luisa_compute_accel_create(void *device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_command_list_create() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_list_append(void *list, void *command) LUISA_NOEXCEPT;
