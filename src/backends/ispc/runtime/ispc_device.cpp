#pragma vengine_package ispc_vsproject

#include <runtime/sampler.h>
#include <backends/ispc/runtime/ispc_device.h>
#include <backends/ispc/runtime/ispc_codegen.h>
#include <backends/ispc/runtime/ispc_shader.h>
#include <backends/ispc/runtime/ispc_runtime.h>
#include "ispc_event.h"

namespace lc::ispc {
void *ISPCDevice::native_handle() const noexcept {
    return nullptr;
}

// buffer
uint64_t ISPCDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64>(vengine_malloc(size_bytes));
}
void ISPCDevice::destroy_buffer(uint64_t handle) noexcept {
    vengine_free(reinterpret_cast<void *>(handle));
}
void *ISPCDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

// texture
uint64_t ISPCDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels) noexcept { return 0; }
void ISPCDevice::destroy_texture(uint64_t handle) noexcept {}
void *ISPCDevice::texture_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}
// stream
uint64_t ISPCDevice::create_stream() noexcept {
    return reinterpret_cast<uint64>(new CommandExecutor(&tPool));
}
void ISPCDevice::destroy_stream(uint64_t handle) noexcept {
    delete reinterpret_cast<CommandExecutor *>(handle);
}
void ISPCDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    auto cmd = reinterpret_cast<CommandExecutor *>(stream_handle);
    cmd->WaitThread();
}
void ISPCDevice::dispatch(uint64_t stream_handle, CommandList cmdList) noexcept {
    auto cmd = reinterpret_cast<CommandExecutor *>(stream_handle);
    for (auto &&i : cmdList) {
        i->accept(*cmd);
    }
    cmd->ExecuteDispatch();
}

void *ISPCDevice::stream_native_handle(uint64_t handle) const noexcept {
    return (void *)handle;
}

// kernel
uint64_t ISPCDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    return reinterpret_cast<uint64>(new Shader(context(), kernel));
}
void ISPCDevice::destroy_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<Shader *>(handle);
}

// event
uint64_t ISPCDevice::create_event() noexcept {
    return reinterpret_cast<uint64>(vengine_new<Event>());
}
void ISPCDevice::destroy_event(uint64_t handle) noexcept {
    vengine_delete(reinterpret_cast<Event *>(handle));
}
void ISPCDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto evt = reinterpret_cast<Event *>(handle);
    auto cmd = reinterpret_cast<CommandExecutor *>(stream_handle);
    {
        std::unique_lock lck(evt->mtx);
        evt->targetFence++;
    }
    cmd->AddTask(CommandExecutor::Signal{evt});
    cmd->ExecuteDispatch();
}
void ISPCDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto evt = reinterpret_cast<Event *>(handle);
    auto cmd = reinterpret_cast<CommandExecutor *>(stream_handle);
    cmd->AddTask(CommandExecutor::Wait{evt});
    cmd->ExecuteDispatch();
}
void Event::Sync() {
    std::unique_lock lck(mtx);
    while (targetFence) {
        cv.wait(lck);
    }
}

void ISPCDevice::synchronize_event(uint64_t handle) noexcept {
    auto evt = reinterpret_cast<Event *>(handle);
    evt->Sync();
}

// accel
uint64_t ISPCDevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept {
    return 0;
}
void ISPCDevice::destroy_mesh(uint64_t handle) noexcept {}
uint64_t ISPCDevice::create_accel() noexcept { return 0; }
void ISPCDevice::destroy_accel(uint64_t handle) noexcept {}
uint64_t ISPCDevice::create_bindless_array(size_t size) noexcept {
    return 0;
}
void ISPCDevice::destroy_bindless_array(uint64_t handle) noexcept {
}
void ISPCDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
}
void ISPCDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
void ISPCDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
void ISPCDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void ISPCDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void ISPCDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
bool ISPCDevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) noexcept {
    return false;
}
bool ISPCDevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) noexcept {
    return false;
}

}// namespace lc::ispc

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view properties) noexcept {
    return luisa::new_with_allocator<lc::ispc::ISPCDevice>(ctx, 0);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
