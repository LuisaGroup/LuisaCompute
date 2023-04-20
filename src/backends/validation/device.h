#pragma once
#include <vstl/common.h>
#include <runtime/rhi/device_interface.h>
namespace lc::validation {
using namespace luisa;
using namespace luisa::compute;
namespace detail {
template<typename T>
class ext_deleter {
    vstd::func_ptr_t<void(T *)> _deleter;

public:
    ext_deleter(vstd::func_ptr_t<void(T *)> deleter) : _deleter{deleter} {}
    void operator()(T *ptr) const {
        _deleter(ptr);
    }
};
}// namespace detail
class Device : public DeviceInterface, public vstd::IOperatorNewBase {
    luisa::shared_ptr<DeviceInterface> _native;
    using ExtPtr = vstd::unique_ptr<DeviceExtension, detail::ext_deleter<DeviceExtension>>;
    vstd::unordered_map<vstd::string, ExtPtr> exts;

public:
    static std::mutex &global_mtx() noexcept;
    static uint64_t origin_handle(uint64_t handle);
    void *native_handle() const noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    Device(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) noexcept;
    ~Device();
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;

    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // stream
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override;

    // swap chain
    SwapChainCreationInfo create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle,
        uint width, uint height, bool allow_hdr,
        bool vsync, uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    // event
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;

    // accel
    ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;

    ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;

    // query
    luisa::string query(luisa::string_view property) noexcept;
    DeviceExtension *extension(luisa::string_view name) noexcept;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
};
}// namespace lc::validation