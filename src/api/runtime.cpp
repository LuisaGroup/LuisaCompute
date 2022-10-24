//
// Created by Mike Smith on 2021/10/17.
//

#include <luisa-compute.h>
#include <ast/function_builder.h>
#include <api/runtime.h>
#define TOMBSTONE 0xdeadbeef
template<class T>
struct RC {
    T *_object;
    std::atomic_uint64_t _ref_count;
    std::function<void(T *)> _deleter;
    uint32_t tombstone;
    RC(T *object, std::function<void(T *)> deleter) : _object{object}, _deleter{deleter}, _ref_count(1) {
        tombstone = 0;
    }
    ~RC() { _deleter(_object); }
    void check() const {
        if (tombstone == TOMBSTONE) {
            LUISA_ERROR_WITH_LOCATION("Object has been destroyed");
        }
    }
    RC *retain() {
        check();
        _ref_count.fetch_add(1, std::memory_order_acquire);
        return this;
    }
    void release() {
        check();
        if (_ref_count.fetch_sub(1, std::memory_order_release) == 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
            tombstone = TOMBSTONE;
            delete this;
        }
    }
    T *object() {
        check();
        return _object;
    }
};
namespace luisa::compute {

struct BufferResource final : public Resource {
    BufferResource(Device::Interface *device, size_t size_bytes) noexcept
        : Resource{device, Tag::BUFFER, device->create_buffer(size_bytes)} {}
};

struct TextureResource final : public Resource {
    TextureResource(
        Device::Interface *device,
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept
        : Resource{device, Tag::TEXTURE, device->create_texture(format, dimension, width, height, depth, mipmap_levels)} {}
};

struct ShaderResource : public Resource {
    ShaderResource(Device::Interface *device, Function f, luisa::string_view opts) noexcept
        : Resource{device, Tag::SHADER, device->create_shader(f, opts)} {}
};

}// namespace luisa::compute

// TODO: rewrite with runtime constructs, e.g., Stream, Event, BindlessArray...

using namespace luisa;
using namespace luisa::compute;
namespace luisa::compute::detail {
class CommandListConverter {
    LCCommandList _list;
    bool _is_c_api;
    luisa::optional<CommandList> _converted;
    void convert() {
    }

public:
    CommandListConverter(const LCCommandList list, bool is_c_api)
        : _list(list), _is_c_api(is_c_api) {
        convert();
    }
    const CommandList &converted() {
        return _converted.value();
    }
    static const LCCommandList *get(const CommandList &list) {
        if (list._c_list.has_value()) {
            return &list._c_list.value();
        }
        return nullptr;
    }
};
}// namespace luisa::compute::detail

LUISA_EXPORT_API LCContext luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return (LCContext)new_with_allocator<Context>(std::filesystem::path{exe_path});
}

LUISA_EXPORT_API void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT {
    delete_with_allocator(reinterpret_cast<Context *>(ctx));
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    auto s = path.string();
    auto cs = static_cast<char *>(malloc(s.size() + 1u));
    memcpy(cs, s.c_str(), s.size() + 1u);
    return cs;
}

LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT { free(cs); }

LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx)->runtime_directory());
}

LUISA_EXPORT_API char *luisa_compute_context_cache_directory(LCContext ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx)->cache_directory());
}

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx, const char *name, const char *properties) LUISA_NOEXCEPT {
    auto device = new Device(std::move(reinterpret_cast<Context *>(ctx)->create_device(name, properties)));
    return (LCDevice)new_with_allocator<RC<Device>>(
        device, [](Device *d) { delete d; });
}

LUISA_EXPORT_API void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device)->release();
}
LUISA_EXPORT_API void luisa_compute_device_retain(LCDevice device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device)->retain();
}
LUISA_EXPORT_API void luisa_compute_device_release(LCDevice device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device)->release();
}

LUISA_EXPORT_API LCBuffer luisa_compute_buffer_create(LCDevice device, size_t size) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    auto handle = d->retain()->object()->impl()->create_buffer(size);
    return reinterpret_cast<LCBuffer>(handle);
}

LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(buffer);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_buffer(handle);
    d->release();
}

LUISA_EXPORT_API LCTexture luisa_compute_texture_create(LCDevice device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT {
    auto dev = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCTexture>(dev->retain()->object()->impl()->create_texture(
        static_cast<PixelFormat>(format),
        dim, w, h, d, mips));
}

LUISA_EXPORT_API void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(texture);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_texture(handle);
    d->release();
}

LUISA_EXPORT_API LCStream luisa_compute_stream_create(LCDevice device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCStream>(d->retain()->object()->impl()->create_stream(false));
}

LUISA_EXPORT_API void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_stream(handle);
    d->release();
}

LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_stream(handle);
}

LUISA_EXPORT_API void luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList c_cmd_list) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    auto converter = luisa::compute::detail::CommandListConverter(c_cmd_list, d->object()->impl()->is_c_api());
    auto &cmd_list = converter.converted();
    d->object()->impl()->dispatch(handle, cmd_list);
}

LUISA_EXPORT_API LCShader luisa_compute_shader_create(LCDevice device, const LCKernelModule *function, const char *options) LUISA_NOEXCEPT {
    // auto d = reinterpret_cast<RC<Device> *>(device);
    // return (LCShader)d->retain()->object()->impl()->create_shader(
    //     luisa::compute::Function{reinterpret_cast<luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> *>(function)->get()},
    //     std::string_view{options});
    abort();
    return nullptr;
}

LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(shader);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_shader(handle);
    d->release();
}

LUISA_EXPORT_API LCEvent luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return (LCEvent)d->retain()->object()->impl()->create_event();
}

LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_event(handle);
    d->release();
}

LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto stream_handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->signal_event(handle, stream_handle);
}

LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto stream_handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->wait_event(handle, stream_handle);
}

LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_event(handle);
}

LUISA_EXPORT_API LCMesh luisa_compute_mesh_create(
    LCDevice device,
    LCBuffer v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    LCBuffer t_buffer, size_t t_offset, size_t t_count, LCAccelUsageHint hint) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCMesh>(d->retain()->object()->impl()->create_mesh(
        reinterpret_cast<uint64_t>(v_buffer), v_offset, v_stride, v_count,
        reinterpret_cast<uint64_t>(t_buffer), t_offset, t_count, static_cast<AccelUsageHint>(hint)));
}

LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCDevice device, LCMesh mesh) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(mesh);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_mesh(handle);
    d->release();
}

LUISA_EXPORT_API LCAccel luisa_compute_accel_create(LCDevice device, LCAccelUsageHint hint) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCAccel>(
        d->retain()->object()->impl()->create_accel(static_cast<AccelUsageHint>(hint)));
}

LUISA_EXPORT_API void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(accel);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_accel(handle);
    d->release();
}

// LCCommand luisa_compute_command_update_mesh(uint64_t handle) LUISA_NOEXCEPT {
//     return (LCCommand)MeshUpdateCommand::create(handle);
// }

// LCCommand luisa_compute_command_update_accel(uint64_t handle) LUISA_NOEXCEPT {
//     return (LCCommand)AccelUpdateCommand::create(handle);
// }

LUISA_EXPORT_API LCPixelStorage luisa_compute_pixel_format_to_storage(LCPixelFormat format) LUISA_NOEXCEPT {
    return static_cast<LCPixelStorage>(
        to_underlying(pixel_format_to_storage(static_cast<PixelFormat>(format))));
}

LUISA_EXPORT_API LCBindlessArray luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    auto bindless_array = luisa::new_with_allocator<BindlessArray>(d->retain()->object()->create_bindless_array(n));
    return reinterpret_cast<LCBindlessArray>(bindless_array);
}

LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCDevice device, LCBindlessArray array) LUISA_NOEXCEPT {
    auto bindless_array = reinterpret_cast<BindlessArray *>(array);
    luisa::delete_with_allocator(bindless_array);
    reinterpret_cast<RC<Device> *>(device)->release();
}

class ExternDevice : public Device::Interface {
    LCDeviceInterface *impl;

public:
    ExternDevice(LCContext ctx, LCDeviceInterface *impl) : Interface(Context(*(Context *)ctx)), impl{impl} {}
    ~ExternDevice() override {
        impl->dtor(impl);
    }
    // native handle
    [[nodiscard]] void *native_handle() const noexcept override {
        return impl;
    }

    // buffer
    [[nodiscard]] uint64_t create_buffer(size_t size_bytes) noexcept override {
        return impl->create_buffer(impl, size_bytes);
    }
    void destroy_buffer(uint64_t handle) noexcept override {
        impl->destroy_buffer(impl, handle);
    }
    [[nodiscard]] void *buffer_native_handle(uint64_t handle) const noexcept override {
        return impl->buffer_native_handle(impl, handle);
    }

    // texture
    [[nodiscard]] uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override {
        return impl->create_texture(impl, static_cast<LCPixelFormat>(format), dimension, width, height, depth, mipmap_levels);
    }
    void destroy_texture(uint64_t handle) noexcept override {
        impl->destroy_texture(impl, handle);
    }
    [[nodiscard]] void *texture_native_handle(uint64_t handle) const noexcept override {
        return impl->texture_native_handle(impl, handle);
    }
    // bindless array
    [[nodiscard]] uint64_t create_bindless_array(size_t size) noexcept override {
        return impl->create_bindless_array(impl, size);
    }
    void destroy_bindless_array(uint64_t handle) noexcept override {
        impl->destroy_bindless_array(impl, handle);
    }
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override {
        impl->emplace_buffer_in_bindless_array(impl, array, index, handle, offset_bytes);
    }
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override {
        impl->emplace_tex2d_in_bindless_array(impl, array, index, handle, LCSampler{.filter = (LCSamplerFilter)sampler.filter(), .address = (LCSamplerAddress)sampler.address()});
    }
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override {
        impl->emplace_tex3d_in_bindless_array(impl, array, index, handle, LCSampler{.filter = (LCSamplerFilter)sampler.filter(), .address = (LCSamplerAddress)sampler.address()});
    }
    bool is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override {
        return impl->is_resource_in_bindless_array(impl, array, handle);
    }
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override {
        impl->remove_buffer_in_bindless_array(impl, array, index);
    }
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override {
        impl->remove_tex2d_in_bindless_array(impl, array, index);
    }
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override {
        impl->remove_tex3d_in_bindless_array(impl, array, index);
    }

    // stream
    [[nodiscard]] uint64_t create_stream(bool for_present) noexcept override {
        return impl->create_stream(impl, for_present);
    }
    void destroy_stream(uint64_t handle) noexcept override {
        impl->destroy_stream(impl, handle);
    }
    void synchronize_stream(uint64_t stream_handle) noexcept override {
        impl->synchronize_stream(impl, stream_handle);
    }
    void dispatch(uint64_t stream_handle, const CommandList &list) noexcept override {
        auto c_cmd_list = compute::detail::CommandListConverter::get(list);
        LUISA_ASSERT(c_cmd_list != nullptr, "null!");
        impl->dispatch(impl, stream_handle, *c_cmd_list);
    }
    void dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
        for (auto &&list : lists) { dispatch(stream_handle, list); }
    }
    void dispatch(uint64_t stream_handle, luisa::move_only_function<void()> &&func) noexcept override {
        abort();
    }
    [[nodiscard]] void *stream_native_handle(uint64_t handle) const noexcept override {
        return impl->stream_native_handle(impl, handle);
    }
    // swap chain
    [[nodiscard]] uint64_t create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
        bool allow_hdr, uint back_buffer_size) noexcept override {
        return impl->create_swap_chain(impl, window_handle, stream_handle, width, height, allow_hdr, back_buffer_size);
    }
    void destroy_swap_chain(uint64_t handle) noexcept override {
        impl->destroy_swap_chain(impl, handle);
    }
    PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept override {
        return static_cast<PixelStorage>(impl->swap_chain_pixel_storage(impl, handle));
    }
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override {
        impl->present_display_in_stream(impl, stream_handle, swapchain_handle, image_handle);
    }
    // kernel
    [[nodiscard]] uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override {
        LUISA_ERROR_WITH_LOCATION("create_shader() is deprecated.");
    }
    [[nodiscard]] virtual uint64_t create_shader_ex(const LCKernelModule *kernel, std::string_view meta_options) noexcept {
        return impl->create_shader_ex(impl, kernel, meta_options.data());
    }

    void destroy_shader(uint64_t handle) noexcept override {
        impl->destroy_shader(impl, handle);
    }

    // event
    [[nodiscard]] uint64_t create_event() noexcept override {
        return impl->create_event(impl);
    }
    void destroy_event(uint64_t handle) noexcept override {
        impl->destroy_event(impl, handle);
    }
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override {
        impl->signal_event(impl, handle, stream_handle);
    }
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override {
        impl->wait_event(impl, handle, stream_handle);
    }
    void synchronize_event(uint64_t handle) noexcept override {
        impl->synchronize_event(impl, handle);
    }

    // accel
    [[nodiscard]] uint64_t create_mesh(
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept override {
        return impl->create_mesh(impl, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_count, static_cast<LCAccelUsageHint>(hint));
    }
    void destroy_mesh(uint64_t handle) noexcept override {
        impl->destroy_mesh(impl, handle);
    }
    [[nodiscard]] uint64_t create_accel(AccelUsageHint hint) noexcept override {
        return impl->create_accel(impl, static_cast<LCAccelUsageHint>(hint));
    }
    void destroy_accel(uint64_t handle) noexcept override {
        impl->destroy_accel(impl, handle);
    }

    // query
    [[nodiscard]] luisa::string query(std::string_view meta_expr) noexcept override { return {}; }
    [[nodiscard]] bool requires_command_reordering() const noexcept override { return impl->requires_command_reordering(impl); }

    [[nodiscard]] bool is_c_api() const noexcept override final { return true; }
};

LUISA_EXPORT_API LCDevice luisa_compute_create_external_device(LCContext ctx, LCDeviceInterface *impl) {
    auto ext_device = luisa::make_shared<ExternDevice>(ctx, impl);
    auto device = new Device{Device::Handle{ext_device}};
    return (LCDevice) new RC<Device>(device, [](Device *d) { delete d; });
}
