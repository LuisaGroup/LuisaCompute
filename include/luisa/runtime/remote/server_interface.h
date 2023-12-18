#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/logging.h>
namespace luisa::compute {
class LC_RUNTIME_API ServerInterface {
public:
    using Handle = luisa::shared_ptr<DeviceInterface>;
    using SendMsgFunc = luisa::move_only_function<void(luisa::vector<std::byte>)>;

private:
    Handle _impl;
    mutable luisa::spin_mutex _handle_mtx;
    luisa::unordered_map<uint64_t, uint64_t> _handle_map;
    SendMsgFunc _send_msg;
    [[nodiscard]] uint64_t native_handle(uint64_t handle) const;
    void insert_handle(uint64_t frontend_handle, uint64_t backend_handle);
    [[nodiscard]] uint64_t remove_handle(uint64_t frontend_handle);

public:
    explicit ServerInterface(
        Handle device_impl,
        SendMsgFunc &&send_msg) noexcept;
    void execute(luisa::span<const std::byte> data, luisa::vector<std::byte> &result) noexcept;
    void create_buffer_ast(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    // void create_buffer_ir(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_bindless_array(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_bindless_array(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_stream(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_stream(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void dispatch(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_swap_chain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_shader_ast(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    // void create_shader_ir(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    // void create_shader_ir_v2(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void load_shader(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void shader_arg_usage(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_shader(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void signal_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void wait_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void sync_event(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_swapchain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_swapchain(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_mesh(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_mesh(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_procedrual_prim(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_procedrual_prim(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_curve(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_curve(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_accel(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_accel(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_sparse_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_sparse_buffer(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void create_sparse_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void destroy_sparse_texture(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void alloc_sparse_buffer_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void dealloc_sparse_buffer_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void alloc_sparse_texture_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void dealloc_sparse_texture_heap(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
    void update_sparse_resource(std::byte const *&ptr, luisa::vector<std::byte> &result) noexcept;
};
}// namespace luisa::compute