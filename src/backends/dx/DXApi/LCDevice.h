#pragma once
#include <vstl/common.h>
#include <runtime/device.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/ShaderPaths.h>
using namespace luisa;
using namespace luisa::compute;
namespace toolhub::directx {
class LCDevice : public DeviceInterface, public vstd::IOperatorNewBase {
    struct Ext {
        using Ctor = vstd::func_ptr_t<DeviceExtension *(LCDevice *)>;
        using Dtor = vstd::func_ptr_t<void(DeviceExtension *)>;
        DeviceExtension *ext;
        Ctor ctor;
        Dtor dtor;
        Ext(Ctor ctor, Dtor dtor) : ext{nullptr}, ctor{ctor}, dtor{dtor} {}
        Ext(Ext const &) = delete;
        Ext(Ext &&rhs) : ext{rhs.ext}, ctor{rhs.ctor}, dtor{rhs.dtor} {
            rhs.ext = nullptr;
        }
        ~Ext() {
            if (ext) {
                dtor(ext);
            }
        }
    };

public:
    ShaderPaths shaderPaths;
    Device nativeDevice;
    std::mutex extMtx;
    vstd::unordered_map<vstd::string, Ext> exts;
    //std::numeric_limits<size_t>::max();
    LCDevice(Context &&ctx, DeviceConfig const *settings);
    ~LCDevice();
    void *native_handle() const noexcept override;
    // buffer
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void set_io(BinaryIO *visitor) noexcept override;
    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
    // IUtil *get_util() noexcept override;
    // stream
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;

    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override {
        ShaderCreationInfo info;
        info.invalidate();
        return info;
    }
    ShaderCreationInfo load_shader(vstd::string_view file_name, vstd::span<Type const *const> types) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        ShaderOption cache_option) noexcept override;
    [[nodiscard]] virtual void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view name,
        bool enable_debug_info,
        bool enable_fast_math) noexcept override;
    [[nodiscard]] ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        span<Type const *const> types,
        string_view ser_path) noexcept override;

    // event
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    // accel
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;

    void destroy_mesh(uint64_t handle) noexcept override;

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;

    void destroy_accel(uint64_t handle) noexcept override;
    // swap chain
    SwapChainCreationInfo create_swap_chain(
        uint64 window_handle,
        uint64 stream_handle,
        uint width,
        uint height,
        bool allow_hdr,
        bool vsync,
        uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    // TODO: un-implemented
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    // TODO: un-implemented
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    DeviceExtension *extension(string_view name) noexcept override;
};
}// namespace toolhub::directx