#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/platform.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/tile_modification.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/depth_format.h>

namespace luisa {
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Context;

namespace detail {
class ContextImpl;
}// namespace detail

namespace ir {
struct KernelModule;
struct Type;
template<class T>
struct CArc;
}// namespace ir

namespace ir_v2 {
struct KernelModule;
}// namespace ir_v2

class Type;
struct AccelOption;

class DeviceConfigExt {
public:
    virtual ~DeviceConfigExt() noexcept = default;
};

class Profiler {
public:
    virtual ~Profiler() noexcept = default;

public:
    virtual void allocate(
        uint64_t handle,
        uint64_t alignment,
        size_t size,
        luisa::string_view name,
        luisa::vector<TraceItem> &&stacktrace) noexcept = 0;
    virtual void free(
        uint64_t handle) noexcept = 0;
    virtual void before_load_shader_bytecode(
        luisa::string_view shader_name) noexcept = 0;
    virtual void before_load_shader_cache(
        luisa::string_view shader_name,
        luisa::string_view cache_file_name) noexcept = 0;
    virtual void after_load_shader_bytecode(
        luisa::string_view shader_name,
        bool matched) noexcept = 0;
    virtual void after_load_shader_cache(
        luisa::string_view shader_name,
        luisa::string_view cache_file_name,
        bool matched) noexcept = 0;
    virtual void before_compile_shader_bytecode(
        luisa::string_view shader_name) noexcept = 0;
    virtual void after_compile_shader_bytecode(
        luisa::string_view shader_name) noexcept = 0;
    virtual void before_compile_shader_cache(
        luisa::string_view shader_name,
        luisa::string_view cache_file_name) noexcept = 0;
    virtual void after_compile_shader_cache(
        luisa::string_view shader_name,
        luisa::string_view cache_file_name) noexcept = 0;
};

struct DeviceConfig {
    mutable luisa::unique_ptr<DeviceConfigExt> extension;
    const BinaryIO *binary_io{nullptr};
    Profiler *profiler{nullptr};
    size_t device_index{std::numeric_limits<size_t>::max()};
    bool inqueue_buffer_limit{true};
    bool headless{false};
};

class DeviceExtension {
protected:
    ~DeviceExtension() noexcept = default;
};

class LC_RUNTIME_API DeviceInterface : public luisa::enable_shared_from_this<DeviceInterface> {

protected:
    friend class Context;
    luisa::string _backend_name;
    luisa::shared_ptr<detail::ContextImpl> _ctx_impl;

public:
    explicit DeviceInterface(Context &&ctx) noexcept;
    virtual ~DeviceInterface() noexcept;
    DeviceInterface(DeviceInterface &&) = delete;
    DeviceInterface(DeviceInterface const &) = delete;

    [[nodiscard]] Context context() const noexcept;
    [[nodiscard]] auto backend_name() const noexcept { return luisa::string_view{_backend_name}; }

    // native handle
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
    [[nodiscard]] virtual uint compute_warp_size() const noexcept = 0;

public:
    [[nodiscard]] virtual BufferCreationInfo create_buffer(const Type *element,
                                                           size_t elem_count,
                                                           void *external_memory /* nullptr if now imported from external memory */) noexcept = 0;
    [[nodiscard]] virtual BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element,
                                                           size_t elem_count,
                                                           void *external_memory /* nullptr if now imported from external memory */) noexcept = 0;
    virtual void destroy_buffer(uint64_t handle) noexcept = 0;

    // texture
    [[nodiscard]] virtual ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept = 0;
    virtual void destroy_texture(uint64_t handle) noexcept = 0;

    // bindless array
    [[nodiscard]] virtual ResourceCreationInfo create_bindless_array(size_t size) noexcept = 0;
    virtual void destroy_bindless_array(uint64_t handle) noexcept = 0;

    // stream
    [[nodiscard]] virtual ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept = 0;
    virtual void destroy_stream(uint64_t handle) noexcept = 0;
    virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandList &&list) noexcept = 0;

    using StreamLogCallback = luisa::function<void(luisa::string_view)>;
    virtual void set_stream_log_callback(uint64_t stream_handle,
                                         const StreamLogCallback &callback) noexcept;

    // swap chain
    [[nodiscard]] virtual SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept = 0;
    virtual void destroy_swap_chain(uint64_t handle) noexcept = 0;
    virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept = 0;

    // kernel
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept = 0;
    [[nodiscard]] virtual ShaderCreationInfo create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
        fprintf(stderr,
                "DeviceInterface::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) is not implemented.");
        abort();
    }
    [[nodiscard]] virtual ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept = 0;
    virtual Usage shader_argument_usage(uint64_t handle, size_t index) noexcept = 0;
    virtual void destroy_shader(uint64_t handle) noexcept = 0;

    // event
    [[nodiscard]] virtual ResourceCreationInfo create_event() noexcept = 0;
    virtual void destroy_event(uint64_t handle) noexcept = 0;
    virtual void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept = 0;
    virtual void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept = 0;
    virtual bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept = 0;
    virtual void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept = 0;

    // accel
    [[nodiscard]] virtual ResourceCreationInfo create_mesh(const AccelOption &option) noexcept = 0;
    virtual void destroy_mesh(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept = 0;
    virtual void destroy_procedural_primitive(uint64_t handle) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo create_curve(const AccelOption &option) noexcept;
    virtual void destroy_curve(uint64_t handle) noexcept;

    [[nodiscard]] virtual ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept;
    virtual void destroy_motion_instance(uint64_t handle) noexcept;

    [[nodiscard]] virtual ResourceCreationInfo create_accel(const AccelOption &option) noexcept = 0;
    virtual void destroy_accel(uint64_t handle) noexcept = 0;

    // query
    [[nodiscard]] virtual luisa::string query(luisa::string_view property) noexcept { return {}; }
    [[nodiscard]] virtual DeviceExtension *extension(luisa::string_view name) noexcept { return nullptr; }
    virtual void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept = 0;

    // sparse buffer
    [[nodiscard]] virtual SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
        return SparseBufferCreationInfo::make_invalid();
    }
    [[nodiscard]] virtual ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept { return ResourceCreationInfo::make_invalid(); }
    virtual void deallocate_sparse_buffer_heap(uint64_t handle) noexcept {}
    virtual void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept {}
    virtual void destroy_sparse_buffer(uint64_t handle) noexcept {}

    // sparse texture
    [[nodiscard]] virtual ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept { return ResourceCreationInfo::make_invalid(); }
    virtual void deallocate_sparse_texture_heap(uint64_t handle) noexcept {}
    [[nodiscard]] virtual SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept {
        return SparseTextureCreationInfo::make_invalid();
    }
    virtual void destroy_sparse_texture(uint64_t handle) noexcept {}
};

}// namespace luisa::compute
