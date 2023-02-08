//
// Created by Mike Smith on 2023/2/6.
//

#pragma once

#include <core/basic_types.h>
#include <ast/function.h>
#include <runtime/context.h>
#include <runtime/resource.h>
#include <runtime/stream_tag.h>
#include <runtime/command.h>
#include <runtime/command_list.h>
#include <runtime/depth_format.h>

namespace luisa::compute {

namespace ir {
struct KernelModule;
}

class MeshFormat;
class RasterState;
class Type;
struct AccelOption;

class DeviceExtension {
public:
    virtual ~DeviceExtension() noexcept = default;
};

class DeviceInterface : public luisa::enable_shared_from_this<DeviceInterface> {

protected:
    Context _ctx;

public:
    explicit DeviceInterface(Context &&ctx) noexcept : _ctx{std::move(ctx)} {}
    virtual ~DeviceInterface() noexcept = default;
    [[nodiscard]] virtual Hash128 device_hash() const noexcept = 0;
    [[nodiscard]] virtual luisa::string cache_name(luisa::string_view file_name) const noexcept = 0;

    [[nodiscard]] const Context &context() const noexcept { return _ctx; }

    // native handle
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
    [[nodiscard]] virtual bool is_c_api() const noexcept { return false; }

    virtual void set_io(BinaryIO *visitor) noexcept = 0;

public:
    // buffer
    [[nodiscard]] virtual BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo register_external_buffer(void *native_handle, size_t size_bytes) noexcept = 0;
    virtual void destroy_buffer(uint64_t handle) noexcept = 0;

    // indirect dispatch buffer
    [[nodiscard]] virtual ResourceCreationInfo create_indirect_dispatch_buffer(size_t capacity) noexcept = 0;
    virtual void destroy_indirect_dispatch_buffer(uint64_t handle) noexcept = 0;

    // texture
    [[nodiscard]] virtual ResourceCreationInfo create_texture(PixelFormat format, uint dimension,
                                                              uint width, uint height, uint depth,
                                                              uint mipmap_levels) noexcept = 0;
    virtual void destroy_texture(uint64_t handle) noexcept = 0;

    // bindless array
    [[nodiscard]] virtual ResourceCreationInfo create_bindless_array(size_t size) noexcept = 0;
    virtual void destroy_bindless_array(uint64_t handle) noexcept = 0;

    // depth buffer
    [[nodiscard]] virtual ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept = 0;
    virtual void destroy_depth_buffer(uint64_t handle) noexcept = 0;

    // stream
    [[nodiscard]] virtual ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept = 0;
    virtual void destroy_stream(uint64_t handle) noexcept = 0;
    virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandList &&list) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandList &&list, luisa::fixed_vector<luisa::move_only_function<void()>, 1> &&callback) noexcept = 0;

    // swap chain
<<<<<<< HEAD
    [[nodiscard]] virtual uint64_t create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
        bool allow_hdr, bool vsync, uint back_buffer_size) noexcept { return ~0ull; }
    virtual void destroy_swap_chain(uint64_t handle) noexcept {}
    [[nodiscard]] virtual void *swapchain_native_handle(uint64_t handle) const noexcept { return nullptr; }
    virtual PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept { return {}; }
    virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {}
    struct ShaderOption {
        bool enable_cache{true};
        bool enable_debug_info{false};
        bool enable_fast_math{true};
        bool compile_only{false};
        luisa::string_view name;
    };
    // kernel
    [[nodiscard]] virtual uint64_t create_shader(Function kernel, ShaderOption shader_option) noexcept = 0;
    [[nodiscard]] virtual uint64_t load_shader(luisa::string_view ser_path, luisa::span<Type const *const> types) noexcept = 0;
    [[nodiscard]] virtual uint3 shader_block_size(uint64_t handle) const noexcept = 0;
=======
    [[nodiscard]] virtual SwapChainCreationInfo create_swap_chain(uint64_t window_handle, uint64_t stream_handle,
                                                                  uint width, uint height, bool allow_hdr,
                                                                  bool vsync, uint back_buffer_size) noexcept = 0;
    virtual void destroy_swap_chain(uint64_t handle) noexcept = 0;
    virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept = 0;

    // kernel
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept = 0;
>>>>>>> f9bd719a (merge)
    virtual void destroy_shader(uint64_t handle) noexcept = 0;

    // TODO
    // raster kernel  (may not be supported by some backends)
<<<<<<< HEAD
    [[nodiscard]] virtual uint64_t create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        ShaderOption shader_option) noexcept { return ~0ull; }
    [[nodiscard]] virtual void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view name,
        bool enable_debug_info,
        bool enable_fast_math) noexcept {}
    [[nodiscard]] virtual uint64_t load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept { return ~0ull; }
    virtual void destroy_raster_shader(uint64_t handle) noexcept {}
=======
    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(const MeshFormat &mesh_format,
                                                                    const RasterState &raster_state,
                                                                    luisa::span<const PixelFormat> rtv_format,
                                                                    DepthFormat dsv_format,
                                                                    Function vert,
                                                                    Function pixel,
                                                                    luisa::string_view serialization_path) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(const MeshFormat &mesh_format,
                                                                    const RasterState &raster_state,
                                                                    luisa::span<const PixelFormat> rtv_format,
                                                                    DepthFormat dsv_format,
                                                                    Function vert,
                                                                    Function pixel,
                                                                    bool use_cache) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo load_raster_shader(const MeshFormat &mesh_format,
                                                                  const RasterState &raster_state,
                                                                  luisa::span<const PixelFormat> rtv_format,
                                                                  DepthFormat dsv_format,
                                                                  luisa::span<Type const *const> types,
                                                                  luisa::string_view ser_path) noexcept = 0;
    virtual void save_raster_shader(const MeshFormat &mesh_format,
                                    Function vert,
                                    Function pixel,
                                    luisa::string_view serialization_path) noexcept = 0;
    virtual void destroy_raster_shader(uint64_t handle) noexcept = 0;
>>>>>>> f9bd719a (merge)

    // event
    [[nodiscard]] virtual ResourceCreationInfo create_event() noexcept = 0;
    virtual void destroy_event(uint64_t handle) noexcept = 0;
    virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void synchronize_event(uint64_t handle) noexcept = 0;

    // accel
    [[nodiscard]] virtual ResourceCreationInfo create_mesh(const AccelOption &option,
                                                           uint64_t vertex_buffer,
                                                           size_t vertex_buffer_offset,
                                                           size_t vertex_stride,
                                                           size_t vertex_count,
                                                           uint64_t triangle_buffer,
                                                           size_t triangle_buffer_offset,
                                                           size_t triangle_count) noexcept = 0;
    virtual void destroy_mesh(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_procedural_primitive(const AccelOption &option,
                                                                           uint64_t aabb_buffer,
                                                                           size_t aabb_buffer_offset,
                                                                           size_t aabb_count) noexcept = 0;
    virtual void destroy_procedural_primitive(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_accel(const AccelOption &option) noexcept = 0;
    virtual void destroy_accel(uint64_t handle) noexcept = 0;

    // query
    [[nodiscard]] virtual luisa::string query(luisa::string_view property) noexcept { return {}; }
    [[nodiscard]] virtual DeviceExtension *extension(luisa::string_view name) noexcept { return nullptr; }
};

}// namespace luisa::compute
