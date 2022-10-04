#pragma once
#include <api/common.h>

typedef struct LCDeviceInterface {
    LCContext ctx;
    void (*dtor)(LCDeviceInterface *self);

    // buffer
    // [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    // virtual void destroy_buffer(uint64_t handle) noexcept = 0;
    // [[nodiscard]] virtual void *buffer_native_handle(uint64_t handle) const noexcept = 0;
    uint64_t (*create_buffer)(LCDeviceInterface *self, size_t size_bytes);
    void (*destroy_buffer)(LCDeviceInterface *self, uint64_t handle);
    void *(*buffer_native_handle)(LCDeviceInterface *self, uint64_t handle);

    // texture
    // [[nodiscard]] virtual uint64_t create_texture(
    //     PixelFormat format, uint32_t dimension,
    //     uint32_t width, uint32_t height, uint32_t depth,
    //     uint32_t mipmap_levels) noexcept = 0;
    // virtual void destroy_texture(uint64_t handle) noexcept = 0;
    // [[nodiscard]] virtual void *texture_native_handle(uint64_t handle) const noexcept = 0;

    uint64_t (*create_texture)(LCDeviceInterface *self, LCPixelFormat format, uint32_t dimension, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipmap_levels);
    void (*destroy_texture)(LCDeviceInterface *self, uint64_t handle);
    void *(*texture_native_handle)(LCDeviceInterface *self, uint64_t handle);

    // bindless array
    // [[nodiscard]] virtual uint64_t create_bindless_array(size_t size) noexcept = 0;
    // virtual void destroy_bindless_array(uint64_t handle) noexcept = 0;
    // virtual void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept = 0;
    // virtual void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept = 0;
    // virtual void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept = 0;
    // virtual bool is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept = 0;
    // virtual void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
    // virtual void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
    // virtual void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;

    uint64_t (*create_bindless_array)(LCDeviceInterface *self, size_t size);
    void (*destroy_bindless_array)(LCDeviceInterface *self, uint64_t handle);
    void (*emplace_buffer_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index, uint64_t handle, size_t offset_bytes);
    void (*emplace_tex2d_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index, uint64_t handle, LCSampler sampler);
    void (*emplace_tex3d_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index, uint64_t handle, LCSampler sampler);
    bool (*is_resource_in_bindless_array)(LCDeviceInterface *self, uint64_t array, uint64_t handle);
    void (*remove_buffer_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index);
    void (*remove_tex2d_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index);
    void (*remove_tex3d_in_bindless_array)(LCDeviceInterface *self, uint64_t array, size_t index);

    // stream
    // [[nodiscard]] virtual uint64_t create_stream(bool for_present) noexcept = 0;
    // virtual void destroy_stream(uint64_t handle) noexcept = 0;
    // virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
    // virtual void dispatch(uint64_t stream_handle, const CommandList &list) noexcept = 0;
    // virtual void dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
    //     for (auto &&list : lists) { dispatch(stream_handle, list); }
    // }
    // virtual void dispatch(uint64_t stream_handle, luisa::move_only_function<void()> &&func) noexcept = 0;
    // [[nodiscard]] virtual void *stream_native_handle(uint64_t handle) const noexcept = 0;

    uint64_t (*create_stream)(LCDeviceInterface *self, bool for_present);
    void (*destroy_stream)(LCDeviceInterface *self, uint64_t handle);
    void (*synchronize_stream)(LCDeviceInterface *self, uint64_t stream_handle);
    void (*dispatch)(LCDeviceInterface *self, uint64_t stream_handle, LCCommandList list);
    void (*dispatch_span)(LCDeviceInterface *self, uint64_t stream_handle, LCCommandList lists, size_t count);
    void (*dispatch_func)(LCDeviceInterface *self, uint64_t stream_handle, void (*func)(void *), void *arg);
    void *(*stream_native_handle)(LCDeviceInterface *self, uint64_t handle);

    // swap chain
    // [[nodiscard]] virtual uint64_t create_swap_chain(
    //     uint64_t window_handle, uint64_t stream_handle, uint32_t width, uint32_t height,
    //     bool allow_hdr, uint32_t back_buffer_size) noexcept = 0;
    // virtual void destroy_swap_chain(uint64_t handle) noexcept = 0;
    // virtual PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept = 0;
    // virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept = 0;

    uint64_t (*create_swap_chain)(LCDeviceInterface *self, uint64_t window_handle, uint64_t stream_handle, uint32_t width, uint32_t height, bool allow_hdr, uint32_t back_buffer_size);
    void (*destroy_swap_chain)(LCDeviceInterface *self, uint64_t handle);
    LCPixelStorage (*swap_chain_pixel_storage)(LCDeviceInterface *self, uint64_t handle);
    void (*present_display_in_stream)(LCDeviceInterface *self, uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle);

    // kernel
    // [[nodiscard]] virtual uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept = 0;
    // virtual void destroy_shader(uint64_t handle) noexcept = 0;

    // kernel is an opqaue handle, see Function::opaque_func
    uint64_t (*create_shader)(LCDeviceInterface *self, void * kernel, const char *meta_options);
    void (*destroy_shader)(LCDeviceInterface *self, uint64_t handle);

    // event
    // [[nodiscard]] virtual uint64_t create_event() noexcept = 0;
    // virtual void destroy_event(uint64_t handle) noexcept = 0;
    // virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    // virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
    // virtual void synchronize_event(uint64_t handle) noexcept = 0;

    uint64_t (*create_event)(LCDeviceInterface *self);
    void (*destroy_event)(LCDeviceInterface *self, uint64_t handle);
    void (*signal_event)(LCDeviceInterface *self, uint64_t handle, uint64_t stream_handle);
    void (*wait_event)(LCDeviceInterface *self, uint64_t handle, uint64_t stream_handle);
    void (*synchronize_event)(LCDeviceInterface *self, uint64_t handle);

    // accel
    // [[nodiscard]] virtual uint64_t create_mesh(
    //     uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    //     uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept = 0;
    // virtual void destroy_mesh(uint64_t handle) noexcept = 0;
    // [[nodiscard]] virtual uint64_t create_accel(AccelUsageHint hint) noexcept = 0;
    // virtual void destroy_accel(uint64_t handle) noexcept = 0;

    uint64_t (*create_mesh)(LCDeviceInterface *self, uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, LCAccelUsageHint hint);
    void (*destroy_mesh)(LCDeviceInterface *self, uint64_t handle);
    uint64_t (*create_accel)(LCDeviceInterface *self, LCAccelUsageHint hint);
    void (*destroy_accel)(LCDeviceInterface *self, uint64_t handle);

    // query
    // [[nodiscard]] virtual luisa::string query(std::string_view meta_expr) noexcept { return {}; }
    // [[nodiscard]] virtual bool requires_command_reordering() const noexcept { return true; }

    const char *(*query)(LCDeviceInterface *self, const char *meta_expr);
    bool (*requires_command_reordering)(LCDeviceInterface *self);

    // [[nodiscard]] virtual Extension *extension() noexcept { return nullptr; }
} LCDeviceInterface;

LUISA_EXPORT_API LCDevice luisa_compute_create_external_device(LCContext ctx, LCDeviceInterface *impl);