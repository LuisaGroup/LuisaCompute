#include <luisa/runtime/remote/server_interface.h>
#include <luisa/core/logging.h>
#include "device_func.h"
#include "serde.hpp"
namespace luisa::compute {
ServerInterface::ServerInterface(
    Handle device_impl,
    SendMsgFunc &&send_msg) noexcept
    : _impl{std::move(device_impl)},
      _send_msg{std::move(send_msg)} {}
uint64_t ServerInterface::native_handle(uint64_t handle) const {
    std::lock_guard lck{_handle_mtx};
    auto iter = _handle_map.find(handle);
    LUISA_ASSERT(iter != _handle_map.end(), "invalid handle.");
    return iter->second;
}
void ServerInterface::insert_handle(uint64_t frontend_handle, uint64_t backend_handle) {
    std::lock_guard lck{_handle_mtx};
    _handle_map.try_emplace(frontend_handle, backend_handle);
}
[[nodiscard]] uint64_t ServerInterface::remove_handle(uint64_t frontend_handle) {
    std::lock_guard lck{_handle_mtx};
    auto iter = _handle_map.find(frontend_handle);
    LUISA_ASSERT(iter != _handle_map.end(), "invalid handle.");
    auto v = iter->second;
    _handle_map.erase(iter);
    return v;
}

void ServerInterface::execute(luisa::span<const std::byte> data, luisa::vector<std::byte> &result) noexcept {
    auto const *ptr = data.data();
    auto func = SerDe::deser_value<DeviceFunc>(ptr);
    switch (func) {
        case DeviceFunc::CreateBufferAst: create_buffer_ast(ptr, result); break;
        // case DeviceFunc::CreateBufferIR: create_buffer_ir(ptr, result); break;
        case DeviceFunc::DestroyBuffer: destroy_buffer(ptr, result); break;
        case DeviceFunc::CreateTexture: create_texture(ptr, result); break;
        case DeviceFunc::DestroyTexture: destroy_texture(ptr, result); break;
        case DeviceFunc::CreateBindlessArray: create_bindless_array(ptr, result); break;
        case DeviceFunc::DestroyBindlessArray: destroy_bindless_array(ptr, result); break;
        case DeviceFunc::CreateStream: create_stream(ptr, result); break;
        case DeviceFunc::DestroyStream: destroy_stream(ptr, result); break;
        case DeviceFunc::Dispatch: dispatch(ptr, result); break;
        case DeviceFunc::CreateSwapChain: create_swap_chain(ptr, result); break;
        case DeviceFunc::CreateShaderAst: create_shader_ast(ptr, result); break;
        // case DeviceFunc::CreateShaderIR: create_shader_ir(ptr, result); break;
        // case DeviceFunc::CreateShaderIRV2: create_shader_ir_v2(ptr, result); break;
        case DeviceFunc::LoadShader: load_shader(ptr, result); break;
        case DeviceFunc::ShaderArgUsage: shader_arg_usage(ptr, result); break;
        case DeviceFunc::DestroyShader: destroy_shader(ptr, result); break;
        case DeviceFunc::CreateEvent: create_event(ptr, result); break;
        case DeviceFunc::DestroyEvent: destroy_event(ptr, result); break;
        case DeviceFunc::SignalEvent: signal_event(ptr, result); break;
        case DeviceFunc::WaitEvent: wait_event(ptr, result); break;
        case DeviceFunc::SyncEvent: sync_event(ptr, result); break;
        case DeviceFunc::CreateSwapchain: create_swapchain(ptr, result); break;
        case DeviceFunc::DestroySwapchain: destroy_swapchain(ptr, result); break;
        case DeviceFunc::CreateMesh: create_mesh(ptr, result); break;
        case DeviceFunc::DestroyMesh: destroy_mesh(ptr, result); break;
        case DeviceFunc::CreateProcedrualPrim: create_procedrual_prim(ptr, result); break;
        case DeviceFunc::DestroyProcedrualPrim: destroy_procedrual_prim(ptr, result); break;
        case DeviceFunc::CreateCurve: create_curve(ptr, result); break;
        case DeviceFunc::DestroyCurve: destroy_curve(ptr, result); break;
        case DeviceFunc::CreateAccel: create_accel(ptr, result); break;
        case DeviceFunc::DestroyAccel: destroy_accel(ptr, result); break;
        case DeviceFunc::CreateSparseBuffer: create_sparse_buffer(ptr, result); break;
        case DeviceFunc::DestroySparseBuffer: destroy_sparse_buffer(ptr, result); break;
        case DeviceFunc::CreateSparseTexture: create_sparse_texture(ptr, result); break;
        case DeviceFunc::DestroySparseTexture: destroy_sparse_texture(ptr, result); break;
        case DeviceFunc::AllocSparseBufferHeap: alloc_sparse_buffer_heap(ptr, result); break;
        case DeviceFunc::DeAllocSparseBufferHeap: dealloc_sparse_buffer_heap(ptr, result); break;
        case DeviceFunc::AllocSparseTextureHeap: alloc_sparse_texture_heap(ptr, result); break;
        case DeviceFunc::DeAllocSparseTextureHeap: dealloc_sparse_texture_heap(ptr, result); break;
        case DeviceFunc::UpdateSparseResource: update_sparse_resource(ptr, result); break;
        default: break;
    }
}
void ServerInterface::create_buffer_ast(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto type = Type::from(SerDe::deser_value<luisa::string>(ptr));
    auto elem_count = SerDe::deser_value<size_t>(ptr);
    auto res = _impl->create_buffer(
        type,
        elem_count,
        nullptr);
    insert_handle(frontend_handle, res.handle);
}
// void ServerInterface::create_buffer_ir(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_buffer(handle);
}
void ServerInterface::create_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto format = SerDe::deser_value<PixelFormat>(ptr);
    auto dimension = SerDe::deser_value<uint>(ptr);
    auto width = SerDe::deser_value<uint>(ptr);
    auto height = SerDe::deser_value<uint>(ptr);
    auto depth = SerDe::deser_value<uint>(ptr);
    auto mipmap_levels = SerDe::deser_value<uint>(ptr);
    auto simultaneous_access = SerDe::deser_value<uint>(ptr);
    auto raster = SerDe::deser_value<uint>(ptr);
    auto res = _impl->create_texture(
        format,
        dimension,
        width, height,
        depth, mipmap_levels,
        simultaneous_access,
        raster);
    insert_handle(frontend_handle, res.handle);
}
void ServerInterface::destroy_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_texture(handle);
}
void ServerInterface::create_bindless_array(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto size = SerDe::deser_value<size_t>(ptr);
    auto res = _impl->create_bindless_array(size);
    insert_handle(frontend_handle, res.handle);
}
void ServerInterface::destroy_bindless_array(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_bindless_array(handle);
}
void ServerInterface::create_stream(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto stream_tag = SerDe::deser_value<StreamTag>(ptr);
    auto res = _impl->create_stream(stream_tag);
    insert_handle(frontend_handle, res.handle);
}
void ServerInterface::destroy_stream(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_stream(handle);
}
void ServerInterface::dispatch(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    // TODO
}
void ServerInterface::create_swap_chain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::create_shader_ast(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
// void ServerInterface::create_shader_ir(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
// void ServerInterface::create_shader_ir_v2(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::load_shader(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::shader_arg_usage(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_shader(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_shader(handle);
}
void ServerInterface::create_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_event(handle);
}
void ServerInterface::signal_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::wait_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::sync_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::create_swapchain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_swapchain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_swap_chain(handle);
}
void ServerInterface::create_mesh(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_mesh(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_mesh(handle);
}
void ServerInterface::create_procedrual_prim(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_procedrual_prim(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_procedural_primitive(handle);
}
void ServerInterface::create_curve(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_curve(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_curve(handle);
}
void ServerInterface::create_accel(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_accel(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_accel(handle);
}
void ServerInterface::create_sparse_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_sparse_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_sparse_buffer(handle);
}
void ServerInterface::create_sparse_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::destroy_sparse_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->destroy_sparse_texture(handle);
}
void ServerInterface::alloc_sparse_buffer_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::dealloc_sparse_buffer_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->deallocate_sparse_buffer_heap(handle);
}
void ServerInterface::alloc_sparse_texture_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
void ServerInterface::dealloc_sparse_texture_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {
    auto frontend_handle = SerDe::deser_value<uint64_t>(ptr);
    auto handle = remove_handle(frontend_handle);
    _impl->deallocate_sparse_texture_heap(handle);
}
void ServerInterface::update_sparse_resource(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept {}
}// namespace luisa::compute