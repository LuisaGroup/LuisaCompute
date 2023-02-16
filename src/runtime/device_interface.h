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
    [[nodiscard]] virtual ResourceCreationInfo register_external_buffer(void *external_ptr, size_t size_bytes) noexcept {
        return ResourceCreationInfo::make_invalid();
    }
    [[nodiscard]] virtual ResourceCreationInfo create_buffer(size_t size_bytes) noexcept = 0;
    [[nodiscard]] virtual TypedBufferCreationInfo create_typed_buffer(const Type *element, size_t elem_count) noexcept = 0;
    virtual void destroy_buffer(uint64_t handle) noexcept = 0;

    // texture
    [[nodiscard]] virtual ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
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

    [[nodiscard]] virtual SwapChainCreationInfo create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle,
        uint width, uint height, bool allow_hdr,
        bool vsync, uint back_buffer_size) noexcept = 0;
    virtual void destroy_swap_chain(uint64_t handle) noexcept = 0;
    virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept = 0;

    // kernel
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept = 0;
    virtual void destroy_shader(uint64_t handle) noexcept = 0;

    // TODO
    // raster kernel  (may not be supported by some backends)
    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        ShaderOption shader_option) noexcept { return ResourceCreationInfo::make_invalid(); }

    virtual void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view name,
        bool enable_debug_info,
        bool enable_fast_math) noexcept {}

    [[nodiscard]] virtual ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept { return ResourceCreationInfo::make_invalid(); }

    virtual void destroy_raster_shader(uint64_t handle) noexcept {}

    // event
    [[nodiscard]] virtual ResourceCreationInfo create_event() noexcept = 0;
    virtual void destroy_event(uint64_t handle) noexcept = 0;
    virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    virtual void synchronize_event(uint64_t handle) noexcept = 0;

    // accel
    [[nodiscard]] virtual ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept = 0;
    virtual void destroy_mesh(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept = 0;
    virtual void destroy_procedural_primitive(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_accel(const AccelOption &option) noexcept = 0;
    virtual void destroy_accel(uint64_t handle) noexcept = 0;

    // query
    [[nodiscard]] virtual luisa::string query(luisa::string_view property) noexcept { return {}; }
    [[nodiscard]] virtual DeviceExtension *extension(luisa::string_view name) noexcept { return nullptr; }
};

}// namespace luisa::compute
