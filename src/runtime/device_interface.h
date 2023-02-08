//
// Created by Mike Smith on 2023/2/6.
//

#pragma once

#include <core/basic_types.h>
#include <ast/function.h>
#include <runtime/context.h>
#include <runtime/pixel.h>
#include <runtime/stream_tag.h>
#include <runtime/command.h>
#include <runtime/command_list.h>
#include <runtime/depth_format.h>

namespace luisa::compute {

class MeshFormat;
class RasterState;

class DeviceExtension {
public:
    virtual ~DeviceExtension() noexcept = default;
};

class DeviceInterface : public luisa::enable_shared_from_this<DeviceInterface> {

protected:
    Context _ctx;

public:
    struct BuiltinBuffer {
        uint64_t handle{};
        uint64_t size{};
    };
    explicit DeviceInterface(Context &&ctx) noexcept : _ctx{std::move(ctx)} {}
    virtual ~DeviceInterface() noexcept = default;
    [[nodiscard]] virtual Hash128 device_hash() const noexcept = 0;
    [[nodiscard]] virtual luisa::string cache_name(luisa::string_view file_name) const noexcept = 0;

    [[nodiscard]] const Context &context() const noexcept { return _ctx; }

    // native handle
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
    [[nodiscard]] virtual bool is_c_api() const noexcept { return false; }

    virtual void set_io(BinaryIO *visitor) noexcept = 0;

    // buffer
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    [[nodiscard]] virtual uint64_t create_buffer(void *native_handle) noexcept { return ~0ull; }
    [[nodiscard]] virtual BuiltinBuffer create_dispatch_buffer(uint32_t dimension, size_t capacity) noexcept = 0;
    [[nodiscard]] virtual BuiltinBuffer create_aabb_buffer(size_t capacity) noexcept = 0;
    // [[nodiscard]] virtual BuiltinBuffer create_draw_buffer(const MeshFormat &mesh_format, bool use_index_buffer, size_t capacity) { return {}; }
    // [[nodiscard]] virtual BuiltinBuffer create_vertexbuffer_args_buffer(size_t capacity) { return {}; }
    // [[nodiscard]] virtual BuiltinBuffer create_indexbuffer_args_buffer(size_t capacity) { return {}; }
    // [[nodiscard]] virtual size_t vertexbuffer_args_stride() { return 0; }
    // [[nodiscard]] virtual size_t indexbuffer_args_stride() { return 0; }

    virtual void destroy_buffer(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual void *buffer_native_handle(uint64_t handle) const noexcept = 0;

    // texture
    [[nodiscard]] virtual uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept = 0;
    virtual void destroy_texture(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual void *texture_native_handle(uint64_t handle) const noexcept = 0;

    // bindless array
    [[nodiscard]] virtual uint64_t create_bindless_array(size_t size) noexcept = 0;
    virtual void destroy_bindless_array(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual void *bindless_native_handle(uint64_t handle) const noexcept = 0;

    // depth buffer
    [[nodiscard]] virtual uint64_t create_depth_buffer(DepthFormat format, uint width, uint height) noexcept { return ~0ull; }
    virtual void destroy_depth_buffer(uint64_t handle) noexcept {}
    [[nodiscard]] virtual void *depth_native_handle(uint64_t handle) const noexcept { return nullptr; }

    // stream
    [[nodiscard]] virtual uint64_t create_stream(StreamTag stream_tag) noexcept = 0;
    virtual void destroy_stream(uint64_t handle) noexcept = 0;
    virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandList &&list) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandList &&list, luisa::fixed_vector<luisa::move_only_function<void()>, 1> &&callback) noexcept = 0;
    [[nodiscard]] virtual void *stream_native_handle(uint64_t handle) const noexcept = 0;

    // swap chain
    [[nodiscard]] virtual uint64_t create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
        bool allow_hdr, bool vsync, uint back_buffer_size) noexcept { return ~0ull; }
    virtual void destroy_swap_chain(uint64_t handle) noexcept {}
    [[nodiscard]] virtual void *swapchain_native_handle(uint64_t handle) const noexcept { return nullptr; }
    virtual PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept { return {}; }
    virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {}
    enum class ShaderCacheOption : bool{
        SAVE = true,
        DISCARD = false
    };
    // kernel
    [[nodiscard]] virtual uint64_t create_shader(Function kernel, variant<string_view, ShaderCacheOption> cache_option) noexcept = 0;
    [[nodiscard]] virtual uint64_t load_shader(luisa::string_view ser_path, luisa::span<Type const *const> types) noexcept = 0;
    [[nodiscard]] virtual uint3 shader_block_size(uint64_t handle) const noexcept = 0;
    virtual void save_shader(Function kernel, luisa::string_view serialization_path) noexcept = 0;
    virtual void destroy_shader(uint64_t handle) noexcept = 0;
// FIXME:
// _ex are experiemental apis
#ifdef LC_ENABLE_API
    [[nodiscard]] LC_RUNTIME_API virtual uint64_t create_shader_ex(const LCKernelModule *kernel, std::string_view meta_options) noexcept;
#endif

    // raster kernel  (may not be supported by some backends)
    [[nodiscard]] virtual uint64_t create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        luisa::string_view serialization_path) noexcept { return ~0ull; }
    [[nodiscard]] virtual uint64_t create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        bool use_cache) noexcept { return ~0ull; }
    [[nodiscard]] virtual uint64_t load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept { return ~0ull; }
    virtual void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view serialization_path) noexcept {}
    virtual void destroy_raster_shader(uint64_t handle) noexcept {}

    // event
    [[nodiscard]] virtual uint64_t create_event() noexcept = 0;
    [[nodiscard]] virtual void *event_native_handle(uint64_t handle) const noexcept = 0;
    virtual void destroy_event(uint64_t handle) noexcept = 0;
    virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void synchronize_event(uint64_t handle) noexcept = 0;

    // accel
    enum struct MeshType : uint8_t {
        Mesh,
        ProceduralPrimitive
    };
    [[nodiscard]] virtual uint64_t create_mesh(
        AccelUsageHint hint, MeshType type,
        bool allow_compact, bool allow_update) noexcept = 0;
    virtual void destroy_mesh(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual uint64_t create_accel(AccelUsageHint hint, bool allow_compact, bool allow_update) noexcept = 0;
    virtual void destroy_accel(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual void *mesh_native_handle(uint64_t handle) const noexcept = 0;
    [[nodiscard]] virtual void *accel_native_handle(uint64_t handle) const noexcept = 0;

    // query
    [[nodiscard]] virtual luisa::string query(luisa::string_view property) noexcept { return {}; }
    [[nodiscard]] virtual DeviceExtension *extension(luisa::string_view name) noexcept { return nullptr; }
};

}// namespace luisa::compute
