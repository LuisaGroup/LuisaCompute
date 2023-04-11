#pragma once
#include <vulkan/vulkan.h>
#include <runtime/device.h>
#include "../vks/VulkanDevice.h"
#include <vstl/common.h>

namespace lc::vk {
struct Settings {
    bool validation;
    bool fullscreen{false};
    bool vsync{false};
    bool overlay{true};
};
using namespace luisa;
using namespace luisa::compute;
class Device : public DeviceInterface {
    VkPhysicalDevice physical_device{};
    vstd::unique_ptr<vks::VulkanDevice> vk_device;
    VkPhysicalDeviceProperties device_properties{};
    VkPhysicalDeviceFeatures device_features{};
    VkPhysicalDeviceMemoryProperties device_memory_properties{};
    vstd::vector<VkPhysicalDevice> physical_devices;
    vstd::vector<const char *> instance_exts = {VK_KHR_SURFACE_EXTENSION_NAME};
    vstd::vector<vstd::string> supported_instance_exts;
    vstd::vector<const char *> enable_device_exts;
    vstd::vector<const char *> enable_inst_ext;
    VkQueue graphics_queue{};
    VkQueue compute_queue{};
    VkQueue copy_queue{};

public:
    VkInstance create_instance(bool enableValidation, Settings &settings);
    bool init_device(Settings &settings, uint32_t selectedDevice);
    Device(Context &&ctx);
    ~Device();
    void *native_handle() const noexcept override;
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;

    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // stream
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override;

    // swap chain
    SwapChainCreationInfo create_swap_chain(
        uint64_t window_handle, uint64_t stream_handle,
        uint width, uint height, bool allow_hdr,
        bool vsync, uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;

    // accel
    ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;

    ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;

    // query
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
};
}// namespace lc::vk