//
// Created by Mike Smith on 2021/10/17.
//

#include <core/rc.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/sampler.h>
#include <runtime/command_list.h>
#include <api/runtime.h>

using namespace luisa;
using namespace luisa::compute;

void *luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return RC<Context>::create(Context{std::filesystem::path{exe_path}});
}

void luisa_compute_context_destroy(void *ctx) LUISA_NOEXCEPT {
    static_cast<RC<Context> *>(ctx)->release();
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    auto s = path.string();
    auto cs = static_cast<char *>(malloc(s.size() + 1u));
    memcpy(cs, s.c_str(), s.size() + 1u);
    return cs;// TODO: free?
}

char *luisa_compute_context_runtime_directory(void *ctx) LUISA_NOEXCEPT {
    return path_to_c_str(static_cast<RC<Context> *>(ctx)->object()->runtime_directory());
}

char *luisa_compute_context_cache_directory(void *ctx) LUISA_NOEXCEPT {
    return path_to_c_str(static_cast<RC<Context> *>(ctx)->object()->cache_directory());
}

void *luisa_compute_device_create(void *ctx, const char *name, uint32_t index) LUISA_NOEXCEPT {
    return RC<Device>::create(
        static_cast<RC<Context> *>(ctx)->retain()->create_device(name, index));
}

void luisa_compute_device_destroy(void *ctx, void *device) LUISA_NOEXCEPT {
    static_cast<RC<Device> *>(device)->release();
    static_cast<RC<Context> *>(ctx)->release();
}

uint64_t luisa_compute_buffer_create(void *device, size_t size, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_buffer(size, heap_handle, index_in_heap);
}

void luisa_compute_buffer_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_buffer(handle);
    d->release();
}

uint64_t luisa_compute_texture_create(void *device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, uint32_t sampler, uint64_t heap, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto dev = static_cast<RC<Device> *>(device);
    return dev->retain()->impl()->create_texture(
        static_cast<PixelFormat>(format),
        dim, w, h, d, mips,
        Sampler::decode(sampler),
        heap, index_in_heap);
}

void luisa_compute_texture_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_texture(handle);
    d->release();
}

uint64_t luisa_compute_heap_create(void *device, size_t size) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_heap(size);
}

void luisa_compute_heap_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_heap(handle);
    d->release();
}

uint64_t luisa_compute_stream_create(void *device) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_stream();
}

void luisa_compute_stream_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_stream(handle);
    d->release();
}

void luisa_compute_stream_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_stream(handle);
}

void luisa_compute_stream_dispatch(void *device, uint64_t handle, void *cmd_list) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->dispatch(handle, std::move(*static_cast<CommandList *>(cmd_list)));
    delete_with_allocator(static_cast<CommandList *>(cmd_list));
}

uint64_t luisa_compute_shader_create(void *device, const void *function) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_shader(
        Function{static_cast<const luisa::compute::detail::FunctionBuilder *>(function)});
}

void luisa_compute_shader_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_shader(handle);
    d->release();
}

uint64_t luisa_compute_event_create(void *device) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_event();
}

void luisa_compute_event_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_event(handle);
    d->release();
}

void luisa_compute_event_signal(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->signal_event(handle, stream);
}

void luisa_compute_event_wait(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->wait_event(handle, stream);
}

void luisa_compute_event_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->synchronize_event(handle);
}

uint64_t luisa_compute_mesh_create(void *device) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_mesh();
}

void luisa_compute_mesh_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_mesh(handle);
    d->release();
}

uint64_t luisa_compute_accel_create(void *device) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    return d->retain()->impl()->create_accel();
}

void luisa_compute_accel_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto d = static_cast<RC<Device> *>(device);
    d->object()->impl()->destroy_accel(handle);
    d->release();
}

void *luisa_compute_command_list_create() LUISA_NOEXCEPT {
    return new_with_allocator<CommandList>();
}

void luisa_compute_command_list_append(void *list, void *command) LUISA_NOEXCEPT {
    static_cast<CommandList *>(list)->append(static_cast<Command *>(command));
}

int luisa_compute_command_list_empty(void *list) LUISA_NOEXCEPT {
    return static_cast<CommandList *>(list)->empty();
}

void *luisa_compute_command_upload_buffer(uint64_t buffer, size_t offset, size_t size, const void *data) LUISA_NOEXCEPT {
    return BufferUploadCommand::create(buffer, offset, size, data);
}

void *luisa_compute_command_download_buffer(uint64_t buffer, size_t offset, size_t size, void *data) LUISA_NOEXCEPT {
    return BufferDownloadCommand::create(buffer, offset, size, data);
}

void *luisa_compute_command_copy_buffer_to_buffer(uint64_t src, size_t src_offset, uint64_t dst, size_t dst_offset, size_t size) LUISA_NOEXCEPT {
    return BufferCopyCommand::create(src, dst, src_offset, dst_offset, size);
}

void *luisa_compute_command_copy_buffer_to_texture(
    uint64_t buffer, size_t buffer_offset,
    uint64_t tex, uint32_t tex_storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT {
    return BufferToTextureCopyCommand::create(
        buffer, buffer_offset, tex,
        static_cast<PixelStorage>(tex_storage), level,
        make_uint3(offset_x, offset_y, offset_z),
        make_uint3(size_x, size_y, size_z));
}

void *luisa_compute_command_copy_texture_to_buffer(
    uint64_t buffer, size_t buffer_offset,
    uint64_t tex, uint32_t tex_storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT {
    return TextureToBufferCopyCommand::create(
        buffer, buffer_offset, tex,
        static_cast<PixelStorage>(tex_storage), level,
        make_uint3(offset_x, offset_y, offset_z),
        make_uint3(size_x, size_y, size_z));
}

void *luisa_compute_command_copy_texture_to_texture(
    uint64_t src, uint32_t src_level,
    uint32_t src_offset_x, uint32_t src_offset_y, uint32_t src_offset_z,
    uint64_t dst, uint32_t dst_level,
    uint32_t dst_offset_x, uint32_t dst_offset_y, uint32_t dst_offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z) LUISA_NOEXCEPT {
    return TextureCopyCommand::create(
        src, dst, src_level, dst_level,
        make_uint3(src_offset_x, src_offset_y, src_offset_z),
        make_uint3(dst_offset_x, dst_offset_y, dst_offset_z),
        make_uint3(size_x, size_y, size_z));
}

void *luisa_compute_command_upload_texture(
    uint64_t handle, uint32_t storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z,
    const void *data) LUISA_NOEXCEPT {
    return TextureUploadCommand::create(
        handle, static_cast<PixelStorage>(storage), level,
        make_uint3(offset_x, offset_y, offset_z),
        make_uint3(size_x, size_y, size_z),
        data);
}

void *luisa_compute_command_download_texture(
    uint64_t handle, uint32_t storage, uint32_t level,
    uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
    uint32_t size_x, uint32_t size_y, uint32_t size_z,
    void *data) LUISA_NOEXCEPT {
    return TextureDownloadCommand::create(
        handle, static_cast<PixelStorage>(storage), level,
        make_uint3(offset_x, offset_y, offset_z),
        make_uint3(size_x, size_y, size_z),
        data);
}

void *luisa_compute_command_dispatch_shader(uint64_t handle, const void *kernel) LUISA_NOEXCEPT {
    return ShaderDispatchCommand::create(handle, Function{static_cast<const luisa::compute::detail::FunctionBuilder *>(kernel)});
}

void luisa_compute_command_dispatch_shader_set_size(void *cmd, uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->set_dispatch_size(make_uint3(sx, sy, sz));
}

void luisa_compute_command_dispatch_shader_encode_buffer(void *cmd, uint32_t vid, uint64_t buffer, size_t offset, uint32_t usage) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->encode_buffer(vid, buffer, offset, static_cast<Usage>(usage));
}

void luisa_compute_command_dispatch_shader_encode_texture(void *cmd, uint32_t vid, uint64_t tex, uint32_t usage) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->encode_texture(vid, tex, static_cast<Usage>(usage));
}

void luisa_compute_command_dispatch_shader_encode_uniform(void *cmd, uint32_t vid, const void *data, size_t size, size_t alignment) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->encode_uniform(vid, data, size, alignment);
}

void luisa_compute_command_dispatch_shader_encode_heap(void *cmd, uint32_t vid, uint64_t heap) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->encode_heap(vid, heap);
}

void luisa_compute_command_dispatch_shader_encode_accel(void *cmd, uint32_t vid, uint64_t accel) LUISA_NOEXCEPT {
    static_cast<ShaderDispatchCommand *>(cmd)->encode_accel(vid, accel);
}

void *luisa_compute_command_build_mesh(
    uint64_t handle, uint32_t hint,
    uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    uint64_t t_buffer, size_t t_offset, size_t t_count) LUISA_NOEXCEPT {
    return MeshBuildCommand::create(
        handle, static_cast<AccelBuildHint>(hint),
        v_buffer, v_offset, v_stride, v_count,
        t_buffer, t_offset, t_count);
}

void *luisa_compute_command_update_mesh(uint64_t handle) LUISA_NOEXCEPT {
    return MeshUpdateCommand::create(handle);
}

void *luisa_compute_command_build_accel(
    uint64_t handle, uint32_t hint,
    const void *instance_mesh_handles,
    const void *instance_transforms,
    size_t instance_count) LUISA_NOEXCEPT {
    return AccelBuildCommand::create(
        handle, static_cast<AccelBuildHint>(hint),
        std::span{static_cast<const uint64_t *>(instance_mesh_handles), instance_count},
        std::span{static_cast<const float4x4 *>(instance_transforms), instance_count});
}

void *luisa_compute_command_update_accel(uint64_t handle, const void *transforms, size_t offset, size_t count) LUISA_NOEXCEPT {
    if (transforms == nullptr) {
        return AccelUpdateCommand::create(handle);
    }
    std::span s{static_cast<const float4x4 *>(transforms) + offset, count};
    return AccelUpdateCommand::create(handle, s, offset);
}

uint32_t luisa_compute_pixel_format_to_storage(uint32_t format) LUISA_NOEXCEPT {
    return to_underlying(pixel_format_to_storage(static_cast<PixelFormat>(format)));
}

size_t luisa_compute_heap_query_usage(void *device, uint64_t handle) LUISA_NOEXCEPT {
    return static_cast<RC<Device> *>(device)->object()->impl()->query_heap_memory_usage(handle);
}
