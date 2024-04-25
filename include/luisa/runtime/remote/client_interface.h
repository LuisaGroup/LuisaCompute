#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/vstl/lockfree_array_queue.h>
namespace luisa::compute {
class ClientCallback {
protected:
    ~ClientCallback() = default;

public:
    virtual void async_send(luisa::vector<std::byte> data) noexcept = 0;
    virtual void sync_send(luisa::span<const std::byte> send, luisa::vector<std::byte> &received) noexcept = 0;
};
class LC_RUNTIME_API ClientInterface : public DeviceInterface {
private:
    struct DispatchFeedback {
        luisa::vector<void *> readback_data;
        CommandList::CallbackContainer callbacks;
    };
    ClientCallback *_callback;
    luisa::vector<std::byte> _receive_bytes;
    luisa::vector<std::byte> _send_bytes;
    luisa::spin_mutex _stream_map_mtx;
    mutable luisa::spin_mutex _evt_mtx;
    luisa::unordered_map<uint64_t, vstd::SingleThreadArrayQueue<DispatchFeedback>> _unfinished_stream;
    luisa::unordered_map<uint64_t, uint64_t> _events;
    uint64_t _flag{0};
    [[nodiscard]] void *native_handle() const noexcept override { return nullptr; }
    [[nodiscard]] uint compute_warp_size() const noexcept override { return 0; }

public:
    explicit ClientInterface(
        Context ctx,
        ClientCallback *callback) noexcept;
    [[nodiscard]] BufferCreationInfo create_buffer(const Type *element,
                                                   size_t elem_count,
                                                   void *external_memory /* nullptr if now imported from external memory */) noexcept override;
    [[nodiscard]] BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element,
                                                   size_t elem_count,
                                                   void *external_memory /* nullptr if now imported from external memory */) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;

    // texture
    [[nodiscard]] ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    [[nodiscard]] ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // stream
    [[nodiscard]] ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;

    using StreamLogCallback = luisa::function<void(luisa::string_view)>;
    void set_stream_log_callback(uint64_t stream_handle,
                                 const StreamLogCallback &callback) noexcept override;

    // swap chain
    [[nodiscard]] SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    [[nodiscard]] ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    [[nodiscard]] ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    [[nodiscard]] ShaderCreationInfo create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept override;
    [[nodiscard]] ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    [[nodiscard]] ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept override;

    // accel
    [[nodiscard]] ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_curve(const AccelOption &option) noexcept override;
    void destroy_curve(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;

    // query
    [[nodiscard]] luisa::string query(luisa::string_view property) noexcept override;
    [[nodiscard]] DeviceExtension *extension(luisa::string_view name) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;

    // sparse buffer
    [[nodiscard]] SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    [[nodiscard]] ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;

    // sparse texture
    [[nodiscard]] ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    [[nodiscard]] SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
};
}// namespace luisa::compute