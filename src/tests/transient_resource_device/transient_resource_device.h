#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include "managed_first_fit.h"
namespace luisa::utils {
using namespace luisa::compute;
struct TransientTexDesc {
    PixelFormat format;
    uint dimension;
    uint width;
    uint height;
    uint depth;
    uint mipmap_levels;
    uint mask;// (simultaneous_access << 1) & allow_raster_target
    void set_simultaneous_access(bool simultaneous_access) {
        if (simultaneous_access) {
            mask |= (1u << 1ul);
        } else {
            mask &= ~(1u << 1ul);
        }
    }
    void set_allow_raster_target(bool allow_raster_target) {
        if (allow_raster_target) {
            mask |= 1u;
        } else {
            mask &= ~1u;
        }
    }

    bool simultaneous_access() const {
        return (mask & (1u << 1u)) != 0;
    }
    bool allow_raster_target() const {
        return (mask & 1u) != 0;
    }
};
class TransientResourceDevice : public luisa::compute::DeviceInterface {
    struct TexResourceHandle {
        ResourceCreationInfo res_info;
        uint64_t index;// index in _native_resources
        uint64_t last_frame;
    };

    struct TexTimelineDesc {
        TexResourceHandle *handle{};
        int64_t start_command_index{std::numeric_limits<int64_t>::max()};
        int64_t end_command_index{std::numeric_limits<int64_t>::min()};
    };
    struct BufferTimelineDesc {
        ManagedFirstFit::Node *_node{};
        size_t offset;
        int64_t start_command_index{std::numeric_limits<int64_t>::max()};
        int64_t end_command_index{std::numeric_limits<int64_t>::min()};
    };

    struct CommandCache {
        vstd::fixed_vector<std::pair<TransientTexDesc *, TexTimelineDesc *>, 4> _allocate_tex;
        vstd::fixed_vector<std::pair<TransientTexDesc *, TexTimelineDesc *>, 4> _deallocate_tex;
        vstd::fixed_vector<std::pair<size_t *, BufferTimelineDesc *>, 4> _allocate_buffer;
        vstd::fixed_vector<std::pair<size_t *, BufferTimelineDesc *>, 4> _deallocate_buffer;
    };
    DeviceInterface *_impl;
    bool _is_managing{false};
    bool _is_committing{false};
    vstd::Pool<TransientTexDesc, true> _tex_meta_pool;
    vstd::Pool<size_t, true> _buffer_meta_pool;
    vstd::Pool<TexResourceHandle, true> _tex_handle_pool;
    ManagedFirstFit _buffer_fit;

    vstd::vector<CommandCache> _command_caches;
    // in dispatch
    vstd::HashMap<TransientTexDesc, vstd::fixed_vector<TexResourceHandle *, 4>> _ready_texs;

    vstd::vector<TexResourceHandle *> _native_resources;
    vstd::vector<uint64_t> _tex_dispose_queue;
    // in managing
    vstd::string _temp_name;

    vstd::HashMap<TransientTexDesc *, TexTimelineDesc> _tex_desc_to_native;
    vstd::HashMap<size_t *, BufferTimelineDesc> _buffer_desc_to_native;

    vstd::HashMap<vstd::string, TransientTexDesc *> _tex_name_to_desc;
    vstd::HashMap<vstd::string, size_t *> _buffer_name_to_desc;
    std::pair<uint64_t, uint64_t> managing_cmd_range{
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()};
    //
    uint64_t _frame_index{0};
    uint64_t _transient_buffer_handle = invalid_resource_handle;
    size_t _transient_buffer_size = 0;

    uint64_t _get_tex_handle(uint64_t handle);
    std::pair<uint64_t, size_t> _get_buffer_handle_offset(uint64_t handle);
    void _mark_tex(uint64_t handle, uint64_t command_index);
    void _mark_buffer(uint64_t handle, uint64_t command_index);
    void _preprocess(luisa::span<luisa::unique_ptr<Command> const> commands, CommandList &cmdlist);
    TexResourceHandle *_allocate_handle(TransientTexDesc const &desc);
    void _deallocate_handle(TexResourceHandle *handle);

public:
    luisa::move_only_function<void(luisa::string &&)> dump_func;
    void *get_native_handle(uint64_t handle);

    void set_next_res_name(vstd::string &&name) {
        _temp_name = std::move(name);
    }
    void begin_managing(CommandList const &cmdlist);

    uint64_t resource_contain_frame{0};

    TransientResourceDevice(Context &&ctx, DeviceInterface *impl);
    ~TransientResourceDevice();
    bool is_committing() const { return _is_committing; }

    [[nodiscard]] void *native_handle() const noexcept override;
    [[nodiscard]] uint compute_warp_size() const noexcept override;
    [[nodiscard]] uint64_t memory_granularity() const noexcept override;
    [[nodiscard]] BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_memory /* nullptr if not imported from external memory */) noexcept override;
    [[nodiscard]] BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_memory /* nullptr if not imported from external memory */) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;

    // texture
    [[nodiscard]] ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, void *external_native_handle,
        bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    [[nodiscard]] ResourceCreationInfo create_bindless_array(size_t size, BindlessSlotType type) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // stream
    [[nodiscard]] ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;

    using StreamLogCallback = luisa::function<void(luisa::string_view)>;
    void set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept override;

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

    [[nodiscard]] ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;

    // query
    [[nodiscard]] luisa::string query(luisa::string_view property) noexcept override;
    [[nodiscard]] DeviceExtension *extension(luisa::string_view name) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    luisa::string_view get_name(uint64_t resource_handle) const noexcept override;

    // sparse buffer
    [[nodiscard]] SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    [[nodiscard]] ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;

    // sparse texture
    [[nodiscard]] ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    [[nodiscard]] SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
    TransientResourceDevice(TransientResourceDevice &&) = delete;
    TransientResourceDevice(TransientResourceDevice const &) = delete;
};
struct TransientResourceDeviceScope {
    compute::Stream &stream;
    CommandList cmdlist;
    compute::Device &transient_res_device;
    TransientResourceDeviceScope(
        compute::Stream &stream,
        compute::Device &transient_res_device,
        bool log_info);
    ~TransientResourceDeviceScope() {
        static_cast<TransientResourceDevice *>(transient_res_device.impl())->dispatch(stream.handle(), std::move(cmdlist));
    }
    template<typename T>
    Buffer<T> create_transient_buffer(vstd::string name, size_t element_count) {
        auto d = static_cast<TransientResourceDevice *>(transient_res_device.impl());
        d->set_next_res_name(std::move(name));
        return transient_res_device.create_buffer<T>(element_count);
    }
    template<typename T>
    Image<T> create_transient_image(vstd::string name, PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) {
        auto d = static_cast<TransientResourceDevice *>(transient_res_device.impl());
        d->set_next_res_name(std::move(name));
        return transient_res_device.create_image<T>(pixel, width, height, mip_levels, simultaneous_access, allow_raster_target);
    }
    template<typename T>
    Volume<T> create_transient_volume(vstd::string name, PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u, bool simultaneous_access = false) {
        auto d = static_cast<TransientResourceDevice *>(transient_res_device.impl());
        d->set_next_res_name(std::move(name));
        return transient_res_device.create_volume<T>(pixel, width, height, depth, mip_levels, simultaneous_access, false);
    }
    template<typename T>
    Image<T> create_transient_image(vstd::string name, PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) {
        auto d = static_cast<TransientResourceDevice *>(transient_res_device.impl());
        d->set_next_res_name(std::move(name));
        return transient_res_device.create_image<T>(pixel, size, mip_levels, simultaneous_access, allow_raster_target);
    }
    template<typename T>
    Volume<T> create_transient_volume(vstd::string name, PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false) {
        auto d = static_cast<TransientResourceDevice *>(transient_res_device.impl());
        d->set_next_res_name(std::move(name));
        return transient_res_device.create_volume<T>(pixel, size, mip_levels, simultaneous_access, false);
    }
};
};// namespace luisa::utils