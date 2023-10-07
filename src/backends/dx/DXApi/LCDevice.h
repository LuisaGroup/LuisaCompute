#pragma once
#include <luisa/vstl/common.h>
#include <luisa/runtime/device.h>
#include <DXRuntime/Device.h>
namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;
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
    Device nativeDevice;
    std::mutex extMtx;
    vstd::unordered_map<vstd::string, Ext> exts;
    //std::numeric_limits<size_t>::max();
    LCDevice(Context &&ctx, DeviceConfig const *settings);
    ~LCDevice();
    void *native_handle() const noexcept override;
    // buffer
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // IUtil *get_util() noexcept override;
    // stream
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;

    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(vstd::string_view file_name, vstd::span<Type const *const> types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence) noexcept override;
    // accel
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;

    void destroy_mesh(uint64_t handle) noexcept override;

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;

    void destroy_accel(uint64_t handle) noexcept override;
    // swap chain
    SwapchainCreationInfo create_swapchain(
        uint64 window_handle,
        uint64 stream_handle,
        uint width,
        uint height,
        bool allow_hdr,
        bool vsync,
        uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    DeviceExtension *extension(string_view name) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    [[nodiscard]] SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;

    [[nodiscard]] SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&update_cmds) noexcept override;
    uint compute_warp_size() const noexcept override;
    luisa::string query(luisa::string_view property) noexcept override;
};
}// namespace lc::dx
