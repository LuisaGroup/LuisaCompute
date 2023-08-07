#pragma once
#include <vulkan/vulkan.h>
#include <luisa/runtime/device.h>
#include "VulkanDevice.h"
#include <luisa/vstl/common.h>
#include "../common/default_binary_io.h"
#include "vk_allocator.h"
namespace lc::hlsl {
class ShaderCompiler;
}// namespace lc::hlsl
namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;
class Device : public DeviceInterface, public vstd::IOperatorNewBase {
    vstd::optional<vks::VulkanDevice> _vk_device;
    VkPhysicalDeviceProperties _device_properties{};
    VkPhysicalDeviceFeatures _device_features{};
    VkPhysicalDeviceMemoryProperties _device_memory_properties{};
    vstd::vector<vstd::string> _enable_device_exts;
    VkQueue _graphics_queue{};
    VkQueue _compute_queue{};
    VkQueue _copy_queue{};
    VkPipelineCacheHeaderVersionOne _pso_header{};
    vstd::optional<VkAllocator> _allocator;
    BinaryIO const *_binary_io{};
    vstd::unique_ptr<DefaultBinaryIO> _default_file_io;
    void _init_device(uint32_t selectedDevice);

public:
    static hlsl::ShaderCompiler *Compiler();
    VkInstance instance() const;
    auto &allocator() { return *_allocator; }
    auto physical_device() const { return _vk_device->physicalDevice; }
    auto logic_device() const { return _vk_device->logicalDevice; }
    auto const &pso_header() const { return _pso_header; }
    bool is_pso_same(VkPipelineCacheHeaderVersionOne const &pso);
    auto const &properties() const { return _vk_device->properties; }
    auto const &features() const { return _vk_device->features; }
    auto graphics_queue_index() const { return _vk_device->queueFamilyIndices.graphics; }
    auto compute_queue_index() const { return _vk_device->queueFamilyIndices.compute; }
    auto copy_queue_index() const { return _vk_device->queueFamilyIndices.transfer; }
    Device(Context &&ctx, DeviceConfig const *configs);
    ~Device();
    void *native_handle() const noexcept override;
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;

    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
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
    SwapchainCreationInfo create_swapchain(
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
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept override;

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
