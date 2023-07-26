#pragma once

#include <luisa/api/common.h>
#include <luisa/api/device.h>
// These are for generate bindings, do not change
typedef void (*LCDispatchCallback)(uint8_t *);
typedef void * VoidPtr;
typedef const void * ConstVoidPtr;
typedef const char * ConstCharPtr;
typedef char * CharPtr;
typedef const LCAccelOption * ConstAccelOptionPtr;
typedef const LCShaderOption * ShaderOptionPtr;
typedef uint8_t * BytePtr;
typedef void (*LoggerCallback)(LCLoggerMessage);
LUISA_EXPORT_API void luisa_compute_set_logger_callback(LoggerCallback callback) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_free_c_string(CharPtr cs) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCContext luisa_compute_context_create(ConstCharPtr exe_path) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx, ConstCharPtr name, ConstCharPtr properties) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_retain(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_device_release(LCDevice device) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedBufferInfo luisa_compute_buffer_create(LCDevice device, ConstVoidPtr element, size_t elem_count) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_texture_create(
    LCDevice device, LCPixelFormat format,
    uint32_t dim, uint32_t w, uint32_t h, uint32_t d,
    uint32_t mips, bool allow_simultaneous_access) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_stream_create(LCDevice device, LCStreamTag stream_tag) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList cmd_list, LCDispatchCallback callback, BytePtr callback_ctx) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedShaderInfo luisa_compute_shader_create(LCDevice device, LCKernelModule func, ShaderOptionPtr option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream, uint64_t value) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream, uint64_t value) LUISA_NOEXCEPT;
LUISA_EXPORT_API bool luisa_compute_is_event_completed(LCDevice device, LCEvent event, uint64_t value) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event, uint64_t value) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCDevice device, LCBindlessArray array) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_mesh_create(LCDevice device, ConstAccelOptionPtr option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCDevice device, LCMesh mesh) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_accel_create(LCDevice device, ConstAccelOptionPtr option) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT;
LUISA_EXPORT_API size_t luisa_compute_device_query(LCDevice device, ConstCharPtr query, CharPtr result, size_t maxlen) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCCreatedSwapchainInfo luisa_compute_swapchain_create(LCDevice device, uint64_t window_handle, LCStream stream_handle, uint32_t width, uint32_t height, bool allow_hdr, bool vsync, uint32_t back_buffer_size) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_swapchain_destroy(LCDevice device, LCSwapchain swapchain) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_swapchain_present(LCDevice device, LCStream stream, LCSwapchain swapchain, LCTexture image) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCPixelStorage luisa_compute_pixel_format_to_storage(LCPixelFormat format) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_set_log_level_verbose() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_set_log_level_info() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_set_log_level_warning() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_set_log_level_error() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_verbose(ConstCharPtr msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_info(ConstCharPtr msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_warning(ConstCharPtr msg) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_log_error(ConstCharPtr msg) LUISA_NOEXCEPT;
