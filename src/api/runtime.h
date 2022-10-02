//
// Created by Mike Smith on 2021/10/17.
//

#pragma once

#include <core/platform.h>
#include <api/common.h>
#include <api/language.h>
_LUISA_API_DECL_TYPE(LCContext);
_LUISA_API_DECL_TYPE(LCDevice);
_LUISA_API_DECL_TYPE(LCShader);
_LUISA_API_DECL_TYPE(LCBuffer);
_LUISA_API_DECL_TYPE(LCTexture);
_LUISA_API_DECL_TYPE(LCStream);
_LUISA_API_DECL_TYPE(LCEvent);
_LUISA_API_DECL_TYPE(LCCommandList);
_LUISA_API_DECL_TYPE(LCCommand);
_LUISA_API_DECL_TYPE(LCBindlessArray);
_LUISA_API_DECL_TYPE(LCMesh);
_LUISA_API_DECL_TYPE(LCAccel);

typedef enum LCAccelUsageHint {
    FAST_TRACE, // build with best quality
    FAST_UPDATE,// optimize for frequent update, usually with compaction
    FAST_BUILD  // optimize for frequent rebuild, maybe without compaction
} LCAccelUsageHint;

typedef enum LCPixelStorage {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4
} LCPixelStorage;

typedef enum LCPixelFormat {

    R8SInt,
    R8UInt,
    R8UNorm,

    RG8SInt,
    RG8UInt,
    RG8UNorm,

    RGBA8SInt,
    RGBA8UInt,
    RGBA8UNorm,

    R16SInt,
    R16UInt,
    R16UNorm,

    RG16SInt,
    RG16UInt,
    RG16UNorm,

    RGBA16SInt,
    RGBA16UInt,
    RGBA16UNorm,

    R32SInt,
    R32UInt,

    RG32SInt,
    RG32UInt,

    RGBA32SInt,
    RGBA32UInt,

    R16F,
    RG16F,
    RGBA16F,

    R32F,
    RG32F,
    RGBA32F
} LCPixelFormat;

typedef struct lc_uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} lc_uint3;
LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCContext luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT;
LUISA_EXPORT_API char *luisa_compute_context_cache_directory(LCContext ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx, const char *name, const char *properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(LCContext ctx, LCDevice device) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCBuffer luisa_compute_buffer_create(LCDevice device, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCBuffer buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API uint32_t luisa_compute_pixel_format_to_storage(uint32_t format) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCTexture luisa_compute_texture_create(LCDevice device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(LCTexture texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCStream luisa_compute_stream_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(LCStream stream, LCCommandList cmd_list) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCShader luisa_compute_shader_create(LCDevice device, LCFunction function, const char *options) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCEvent luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCBindlessArray luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCBindlessArray array) LUISA_NOEXCEPT;
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
LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCAccel luisa_compute_accel_create(LCDevice device, LCAccelUsageHint hint) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(LCAccel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace_back(LCAccel accel, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_emplace(LCAccel accel, size_t index, void *mesh, const void *transform, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_transform(LCAccel accel, size_t index, const void *transform) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_set_visibility(LCAccel accel, size_t index, int visibility) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_pop_back(LCAccel accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCommandList luisa_compute_command_list_create() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_list_append(LCCommandList list, LCCommand command) LUISA_NOEXCEPT;
LUISA_EXPORT_API int luisa_compute_command_list_empty(LCCommandList list) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCommand luisa_compute_command_upload_buffer(
    LCBuffer buffer, size_t offset, size_t size, const void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_download_buffer(
    LCBuffer buffer, size_t offset, size_t size, void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_copy_buffer_to_buffer(
    LCBuffer src, size_t src_offset,
    LCBuffer dst, size_t dst_offset, size_t size) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_copy_buffer_to_texture(
    LCBuffer buffer, size_t buffer_offset,
    LCTexture texture, LCPixelStorage storage,
    uint32_t level, lc_uint3 size) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_copy_texture_to_buffer(
    LCBuffer buffer, size_t buffer_offset,
    LCTexture tex, LCPixelStorage tex_storage, uint32_t level,
    lc_uint3 size) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_copy_texture_to_texture(
    LCTexture src, uint32_t src_level,
    LCTexture dst, uint32_t dst_level,
    LCPixelStorage storage, lc_uint3 size) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_upload_texture(
    LCTexture handle, LCPixelStorage storage, uint32_t level,
    lc_uint3 size, const void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_download_texture(
   LCTexture handle, LCPixelStorage storage, uint32_t level,
    lc_uint3 size,  void *data) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCommand luisa_compute_command_dispatch_shader(LCShader handle, LCKernel kernel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_set_size(LCCommand cmd, uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_buffer(
    LCCommand cmd, uint32_t vid, LCBuffer buffer, size_t offset, uint32_t usage) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_texture(
    LCCommand cmd, uint32_t vid, LCTexture tex, uint32_t level, uint32_t usage) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_uniform(
    LCCommand cmd, uint32_t vid,
    const void *data, size_t size, size_t alignment) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_bindless_array(LCCommand cmd, uint32_t vid, uint64_t heap) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_command_dispatch_shader_encode_accel(LCCommand, uint32_t vid, LCAccel accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCommand luisa_compute_command_build_mesh(LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_update_mesh(LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_build_accel(LCAccel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCommand luisa_compute_command_update_accel(LCAccel accel) LUISA_NOEXCEPT;
