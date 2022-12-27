#pragma once
#include <vstl/common.h>
#include <runtime/device.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/ShaderPaths.h>
using namespace luisa::compute;
namespace toolhub::directx {
class LCUtil;
using LCDeviceInterface = luisa::compute::DeviceInterface;
class LCDevice : public LCDeviceInterface, public vstd::IOperatorNewBase {
public:
    ShaderPaths shaderPaths;
    Device nativeDevice;
    vstd::unique_ptr<LCUtil> util;
    //std::numeric_limits<size_t>::max();
    LCDevice(Context &&ctx, DeviceConfig const *settings);
    ~LCDevice();
    void *native_handle() const noexcept override;
    luisa::Hash128 device_hash() const noexcept;
    luisa::string cache_name(luisa::string_view file_name) const noexcept override;
    // buffer
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    uint64_t create_buffer(void *ptr) noexcept override;
    BuiltinBuffer create_dispatch_buffer(uint32_t dimension, size_t capacity) noexcept override;
    BuiltinBuffer create_aabb_buffer(size_t capacity) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void *buffer_native_handle(uint64_t handle) const noexcept override;
    void set_io_visitor(BinaryIO *visitor) noexcept override;
    // texture
    uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    void *texture_native_handle(uint64_t handle) const noexcept override;

    // bindless array
    uint64_t create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void remove_buffer_from_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex2d_from_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex3d_from_bindless_array(uint64_t array, size_t index) noexcept override;
    uint64_t create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
    // IUtil *get_util() noexcept override;
    // stream
    uint64_t create_stream(StreamTag stream_tag) noexcept override;

    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list, luisa::fixed_vector<luisa::move_only_function<void()>, 1> &&func) noexcept override;
    void *stream_native_handle(uint64_t handle) const noexcept override;

    // kernel
    uint64_t create_shader(Function kernel, vstd::string_view file_name) noexcept override;
    uint64_t create_shader(Function kernel, bool is_cache) noexcept override;
    uint64_t load_shader(vstd::string_view file_name, vstd::span<Type const *const> types) noexcept override;
    void save_shader(Function kernel, luisa::string_view serialization_path) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    uint64_t create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        luisa::string_view serialization_path) noexcept override;

    uint64_t create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        bool use_cache) noexcept override;
    [[nodiscard]] uint64_t load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept override;
    void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view serialization_path) noexcept override;

    // event
    uint64_t create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    // accel
    uint64_t create_mesh(
        AccelUsageHint hint,
        MeshType type,
        bool allow_compact, bool allow_update) noexcept override;

    void destroy_mesh(uint64_t handle) noexcept override;

    uint64_t create_accel(AccelUsageHint hint, bool allow_compact, bool allow_update) noexcept override;

    void destroy_accel(uint64_t handle) noexcept override;
    // swap chain
    uint64_t create_swap_chain(
        uint64 window_handle,
        uint64 stream_handle,
        uint width,
        uint height,
        bool allow_hdr,
        bool vsync,
        uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
};
}// namespace toolhub::directx