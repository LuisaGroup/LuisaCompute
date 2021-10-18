//
// Created by Mike Smith on 2021/10/17.
//

#include <core/allocator.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/texture.h>
#include <runtime/command_list.h>
#include <api/runtime.h>

using namespace luisa;
using namespace luisa::compute;

void *luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return new_with_allocator<Context>(std::filesystem::path{exe_path});
}

void luisa_compute_context_destroy(void *ctx) LUISA_NOEXCEPT {
    delete_with_allocator(static_cast<Context *>(ctx));
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    return strdup(path.string().c_str());
}

char *luisa_compute_context_runtime_directory(void *ctx) LUISA_NOEXCEPT {
    return path_to_c_str(static_cast<Context *>(ctx)->runtime_directory());
}

char *luisa_compute_context_cache_directory(void *ctx) LUISA_NOEXCEPT {
    return path_to_c_str(static_cast<Context *>(ctx)->cache_directory());
}

void *luisa_compute_device_create(void *ctx, const char *name, uint32_t index) LUISA_NOEXCEPT {
    return new_with_allocator<Device>(static_cast<Context *>(ctx)->create_device(name, index));
}

void luisa_compute_device_destroy(void *device) LUISA_NOEXCEPT {
    delete_with_allocator(static_cast<Device *>(device));
}

uint64_t luisa_compute_buffer_create(void *device, size_t size, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_buffer(size, heap_handle, index_in_heap);
}

void luisa_compute_buffer_destroy(void *device, uint64_t handle, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_buffer(handle);// TODO
}

uint64_t luisa_compute_texture_create(void *device, uint32_t format, uint32_t dim, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, uint32_t sampler, uint64_t heap, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_texture(
        static_cast<PixelFormat>(format),
        dim, w, h, d, mips,
        TextureSampler::decode(sampler),
        heap, index_in_heap);
}

void luisa_compute_texture_destroy(void *device, uint64_t handle, uint64_t heap_handle, uint32_t index_in_heap) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_texture(handle);//TODO
}

uint64_t luisa_compute_heap_create(void *device, size_t size) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_heap(size);
}

void luisa_compute_heap_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_heap(handle);
}

uint64_t luisa_compute_stream_create(void *device) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_stream();
}

void luisa_compute_stream_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_stream(handle);
}

void luisa_compute_stream_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->synchronize_stream(handle);
}

void luisa_compute_stream_dispatch(void *device, uint64_t handle, void *cmd_list) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->dispatch(handle, std::move(*static_cast<CommandList *>(cmd_list)));
    delete_with_allocator(static_cast<CommandList *>(cmd_list));
}

uint64_t luisa_compute_shader_create(void *device, const void *function) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_shader(Function{static_cast<const luisa::compute::detail::FunctionBuilder *>(function)});
}

void luisa_compute_shader_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_shader(handle);
}

uint64_t luisa_compute_event_create(void *device) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_event();
}

void luisa_compute_event_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_event(handle);
}

void luisa_compute_event_signal(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->signal_event(handle, stream);
}

void luisa_compute_event_wait(void *device, uint64_t handle, uint64_t stream) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->wait_event(handle, stream);
}

void luisa_compute_event_synchronize(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->synchronize_event(handle);
}

uint64_t luisa_compute_mesh_create(void *device) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_mesh();
}

void luisa_compute_mesh_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_mesh(handle);
}

uint64_t luisa_compute_accel_create(void *device) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    return impl->create_accel();
}

void luisa_compute_accel_destroy(void *device, uint64_t handle) LUISA_NOEXCEPT {
    auto impl = static_cast<Device *>(device)->impl();
    impl->destroy_accel(handle);
}

void *luisa_compute_command_list_create() LUISA_NOEXCEPT {
    return new_with_allocator<CommandList>();
}

void luisa_compute_command_list_append(void *list, void *command) LUISA_NOEXCEPT {
    static_cast<CommandList *>(list)->append(static_cast<Command *>(command));
}
