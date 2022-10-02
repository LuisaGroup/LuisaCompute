//
// Created by Mike Smith on 2021/10/17.
//

#include <luisa-compute.h>
#include <ast/function_builder.h>
#include <api/runtime.h>
#define TOMBSTONE 0xdeadbeef
template<class T>
struct RC {
    T _object;
    std::atomic_uint64_t _ref_count;
    std::function<void(T *)> _deleter;
    uint32_t tombstone;
    RC(T object, std::function<void(T *)> deleter) : _object{object}, _deleter{deleter}, _ref_count(1) {
        tombstone = 0;
    }
    ~RC() { _deleter(&_object); }
    void check() {
        if (tombstone == TOMBSTONE) {
            LUISA_ERROR_WITH_LOCATION("Object has been destroyed");
        }
    }
    RC *retain() const {
        check();
        _ref_count.fetch_add(1, std::memory_order_acquire);
        return this;
    }
    void release() const {
        check();
        if (_ref_count.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            tombstone = TOMBSTONE;
            delete this;
        }
    }
    T *object() {
        check();
        return &_object;
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

LCContext luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return (LCContext)new_with_allocator<Context>(std::filesystem::path{exe_path});
}

void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT {
    delete_with_allocator(reinterpret_cast<Context *>(ctx));
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    auto s = path.string();
    auto cs = static_cast<char *>(malloc(s.size() + 1u));
    memcpy(cs, s.c_str(), s.size() + 1u);
    return cs;
}

void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT { free(cs); }

char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx)->runtime_directory());
}

char *luisa_compute_context_cache_directory(LCContext ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx)->cache_directory());
}

LCDevice luisa_compute_device_create(LCContext ctx, const char *name, const char *properties) LUISA_NOEXCEPT {
    auto device = reinterpret_cast<Context *>(ctx)->create_device(
        name, nlohmann::json::parse(properties));
    return (LCDevice)new_with_allocator<RC<Device>>(
        device, [](Device *d) { delete_with_allocator(d); });
}

void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device)->release();
}

LCBuffer luisa_compute_buffer_create(LCDevice device, size_t size) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    auto handle = d->retain()->object()->impl()->create_buffer(size);
    return reinterpret_cast<LCBuffer>(handle);
}

void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(buffer);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_buffer(handle);
    d->release();
}

LCTexture luisa_compute_texture_create(LCDevice device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips) LUISA_NOEXCEPT {
    auto dev = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCTexture>(dev->retain()->object()->impl()->create_texture(
        static_cast<PixelFormat>(format),
        dim, w, h, d, mips));
}

void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(texture);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_texture(handle);
    d->release();
}

LCStream luisa_compute_stream_create(LCDevice device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCStream>(d->retain()->object()->impl()->create_stream(false));
}

void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_stream(handle);
    d->release();
}

void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_stream(handle);
}

void luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList cmd_list) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->dispatch(handle, std::move(*reinterpret_cast<CommandList *>(cmd_list)));
    delete_with_allocator(reinterpret_cast<CommandList *>(cmd_list));
}

LCShader luisa_compute_shader_create(LCDevice device, LCFunction function, const char *options) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return (LCShader)d->retain()->object()->impl()->create_shader(
        luisa::compute::Function{reinterpret_cast<luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> *>(function)->get()},
        std::string_view{options});
}

void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(shader);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_shader(handle);
    d->release();
}

LCEvent luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return (LCEvent)d->retain()->object()->impl()->create_event();
}

void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_event(handle);
    d->release();
}

void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto stream_handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->signal_event(handle, stream_handle);
}

void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto stream_handle = reinterpret_cast<uint64_t>(stream);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->wait_event(handle, stream_handle);
}

void luisa_compute_event_synchronize(LCDevice device, LCEvent event) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(event);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_event(handle);
}

LCMesh luisa_compute_mesh_create(
    LCDevice device,
    LCBuffer v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    LCBuffer t_buffer, size_t t_offset, size_t t_count, LCAccelUsageHint hint) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCMesh>(d->retain()->object()->impl()->create_mesh(
        reinterpret_cast<uint64_t>(v_buffer), v_offset, v_stride, v_count,
        reinterpret_cast<uint64_t>(t_buffer), t_offset, t_count, static_cast<AccelUsageHint>(hint)));
}

void luisa_compute_mesh_destroy(LCDevice device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_mesh(handle);
    d->release();
}

LCAccel luisa_compute_accel_create(LCDevice device, LCAccelUsageHint hint) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device);
    return reinterpret_cast<LCAccel>(
        d->retain()->object()->impl()->create_accel(static_cast<AccelUsageHint>(hint)));
}

void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(accel);
    auto d = reinterpret_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_accel(handle);
    d->release();
}

LCCommandList luisa_compute_command_list_create() LUISA_NOEXCEPT {
    return (LCCommandList)new_with_allocator<CommandList>();
}

void luisa_compute_command_list_append(LCCommandList list, LCCommand command) LUISA_NOEXCEPT {
    reinterpret_cast<CommandList *>(list)->append(reinterpret_cast<Command *>(command));
}

int luisa_compute_command_list_empty(LCCommandList list) LUISA_NOEXCEPT {
    return reinterpret_cast<CommandList *>(list)->empty();
}

LCCommand luisa_compute_command_upload_buffer(uint64_t buffer, size_t offset, size_t size, const void *data) LUISA_NOEXCEPT {
    return (LCCommand)BufferUploadCommand::create(buffer, offset, size, data);
}

LCCommand luisa_compute_command_download_buffer(uint64_t buffer, size_t offset, size_t size, void *data) LUISA_NOEXCEPT {
    return (LCCommand)BufferDownloadCommand::create(buffer, offset, size, data);
}

LCCommand luisa_compute_command_copy_buffer_to_buffer(uint64_t src, size_t src_offset, uint64_t dst, size_t dst_offset, size_t size) LUISA_NOEXCEPT {
    return (LCCommand)BufferCopyCommand::create(src, dst, src_offset, dst_offset, size);
}

LCCommand luisa_compute_command_copy_buffer_to_texture(
    LCBuffer buffer, size_t buffer_offset,
    LCTexture texture, LCPixelStorage storage,
    uint32_t level, lc_uint3 size) LUISA_NOEXCEPT {
    auto buffer_handle = reinterpret_cast<uint64_t>(buffer);
    auto texture_handle = reinterpret_cast<uint64_t>(texture);
    return reinterpret_cast<LCCommand>(BufferToTextureCopyCommand::create(
        buffer_handle, buffer_offset, texture_handle,
        static_cast<PixelStorage>(storage), level, make_uint3(size.x, size.y, size.z)));
}

LCCommand luisa_compute_command_copy_texture_to_buffer(
    LCBuffer buffer, size_t buffer_offset,
    LCTexture texture, LCPixelStorage storage, uint32_t level,
    lc_uint3 size) LUISA_NOEXCEPT {
    auto buffer_handle = reinterpret_cast<uint64_t>(buffer);
    auto texture_handle = reinterpret_cast<uint64_t>(texture);
    return reinterpret_cast<LCCommand>(TextureToBufferCopyCommand::create(
        buffer_handle, buffer_offset, texture_handle,
        static_cast<PixelStorage>(storage), level, make_uint3(size.x, size.y, size.z)));
}

LCCommand luisa_compute_command_copy_texture_to_texture(
    LCTexture src, uint32_t src_level,
    LCTexture dst, uint32_t dst_level,
    LCPixelStorage storage, lc_uint3 size) LUISA_NOEXCEPT {
    auto src_handle = reinterpret_cast<uint64_t>(src);
    auto dst_handle = reinterpret_cast<uint64_t>(dst);
    return reinterpret_cast<LCCommand>(TextureCopyCommand::create(
        static_cast<PixelStorage>(storage),
        src_handle, dst_handle, src_level, dst_level, make_uint3(size.x, size.y, size.z)));
}

LCCommand luisa_compute_command_upload_texture(
    LCTexture handle, LCPixelStorage storage, uint32_t level,
    lc_uint3 size,  const void *data) LUISA_NOEXCEPT {
    auto texture_handle = reinterpret_cast<uint64_t>(handle);
    return reinterpret_cast<LCCommand>(TextureUploadCommand::create(
        texture_handle, static_cast<PixelStorage>(storage), level,
        make_uint3(size.x, size.y, size.z), data));
}

LCCommand luisa_compute_command_download_texture(
   LCTexture handle, LCPixelStorage storage, uint32_t level,
    lc_uint3 size,  void *data) LUISA_NOEXCEPT {
    auto texture_handle = reinterpret_cast<uint64_t>(handle);
    return reinterpret_cast<LCCommand>(TextureDownloadCommand::create(
        texture_handle, static_cast<PixelStorage>(storage), level,
        make_uint3(size.x, size.y, size.z), data));
}

LCCommand luisa_compute_command_dispatch_shader(LCShader shader, LCKernel kernel) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<uint64_t>(shader);
    return reinterpret_cast<LCCommand>(
        ShaderDispatchCommand::create(handle, Function{reinterpret_cast<luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> *>(kernel)->get()}));
}

void luisa_compute_command_dispatch_shader_set_size(LCCommand cmd, uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->set_dispatch_size(make_uint3(sx, sy, sz));
}

void luisa_compute_command_dispatch_shader_encode_buffer(LCCommand cmd, uint32_t vid, uint64_t buffer, size_t offset, uint32_t usage) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->encode_buffer(vid, buffer, offset, static_cast<Usage>(usage));
}

void luisa_compute_command_dispatch_shader_encode_texture(LCCommand cmd, uint32_t vid, uint64_t tex, uint32_t level, uint32_t usage) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->encode_texture(vid, tex, level, static_cast<Usage>(usage));
}

void luisa_compute_command_dispatch_shader_encode_uniform(LCCommand cmd, uint32_t vid, const void *data, size_t size, size_t alignment) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->encode_uniform(vid, data, size, alignment);
}

void luisa_compute_command_dispatch_shader_encode_heap(LCCommand cmd, uint32_t vid, uint64_t heap) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->encode_bindless_array(vid, heap);
}

void luisa_compute_command_dispatch_shader_encode_accel(LCCommand cmd, uint32_t vid, uint64_t accel) LUISA_NOEXCEPT {
    reinterpret_cast<ShaderDispatchCommand *>(cmd)->encode_accel(vid, accel);
}

LCCommand luisa_compute_command_build_mesh(uint64_t handle) LUISA_NOEXCEPT {
    return (LCCommand)MeshBuildCommand::create(handle);
}

LCCommand luisa_compute_command_update_mesh(uint64_t handle) LUISA_NOEXCEPT {
    return (LCCommand)MeshUpdateCommand::create(handle);
}

LCCommand luisa_compute_command_build_accel(uint64_t handle) LUISA_NOEXCEPT {
    return (LCCommand)AccelBuildCommand::create(handle);
}

LCCommand luisa_compute_command_update_accel(uint64_t handle) LUISA_NOEXCEPT {
    return (LCCommand)AccelUpdateCommand::create(handle);
}

uint32_t luisa_compute_pixel_format_to_storage(uint32_t format) LUISA_NOEXCEPT {
    return to_underlying(pixel_format_to_storage(static_cast<PixelFormat>(format)));
}

uint64_t luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    auto bindless_array = luisa::new_with_allocator<BindlessArray>(d->retain()->create_bindless_array(n));
    return reinterpret_cast<uint64_t>(bindless_array);
}

void luisa_compute_bindless_array_destroy(LCDevice device, uint64_t handle) LUISA_NOEXCEPT {
    auto bindless_array = reinterpret_cast<BindlessArray *>(handle);
    luisa::delete_with_allocator(bindless_array);
    static_cast<RC<Device> *>(device)->release();
}
