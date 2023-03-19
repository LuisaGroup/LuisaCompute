#pragma once
#include <runtime/rhi/device_interface.h>
namespace luisa::compute::rust {
    // @Mike-Leo-Smith: fill-in the blanks pls
    class RustDevice final: public DeviceInterface {
        void * _handle;
    public:
        RustDevice(Context &&ctx, string_view name) noexcept;
        void *native_handle() const noexcept override { return _handle; }
        bool is_c_api() const noexcept override { return false; }
        BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
        BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
        void destroy_buffer(uint64_t handle) noexcept override;
        ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
        void destroy_texture(uint64_t handle) noexcept override;
        ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
        void destroy_bindless_array(uint64_t handle) noexcept override;
        ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
        void destroy_depth_buffer(uint64_t handle) noexcept override;
        ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
        void destroy_stream(uint64_t handle) noexcept override;
        void synchronize_stream(uint64_t stream_handle) noexcept override;
        void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
        SwapChainCreationInfo create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept override;
        void destroy_swap_chain(uint64_t handle) noexcept override;
        void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
        ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
        ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
        ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
        void destroy_shader(uint64_t handle) noexcept override;
        ResourceCreationInfo create_raster_shader(const MeshFormat &mesh_format,
                                                const RasterState &raster_state,
                                                luisa::span<const PixelFormat> rtv_format,
                                                DepthFormat dsv_format,
                                                Function vert, Function pixel,
                                                const ShaderOption &shader_option) noexcept override;
        void save_raster_shader(const MeshFormat &mesh_format,
                                Function vert, Function pixel,
                                luisa::string_view name,
                                bool enable_debug_info,
                                bool enable_fast_math) noexcept override;
        ResourceCreationInfo load_raster_shader(const MeshFormat &mesh_format,
                                                const RasterState &raster_state,
                                                luisa::span<const PixelFormat> rtv_format,
                                                DepthFormat dsv_format,
                                                luisa::span<const Type *const> types,
                                                luisa::string_view ser_path) noexcept override;
        void destroy_raster_shader(uint64_t handle) noexcept override;
        ResourceCreationInfo create_event() noexcept override;
        void destroy_event(uint64_t handle) noexcept override;
        void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
        void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
        void synchronize_event(uint64_t handle) noexcept override;
        ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
        void destroy_mesh(uint64_t handle) noexcept override;
        ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
        void destroy_procedural_primitive(uint64_t handle) noexcept override;
        ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
        void destroy_accel(uint64_t handle) noexcept override;
        string query(luisa::string_view property) noexcept override;
        DeviceExtension *extension(luisa::string_view name) noexcept override;
    };
}