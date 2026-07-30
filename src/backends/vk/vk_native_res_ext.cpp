#include "vk_native_res_ext.h"
#include "buffer.h"
#include "texture.h"
#include "device.h"
namespace lc::vk {
VkNativeResourceExt::VkNativeResourceExt(Device *device) : NativeResourceExt(device) {
    _device = device;
}
BufferCreationInfo VkNativeResourceExt::register_external_buffer(
    void *buffer_ptr,
    const Type *element,
    size_t elem_count,
    // custom data see backends' header
    void *custom_data) noexcept {
    size_t elem_size = (element == Type::of<void>()) ? 1 : element->size();
    size_t size = elem_size * elem_count;
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(new ExternalBuffer(
        _device,
        static_cast<VkBuffer>(buffer_ptr),
        size));
    info.native_handle = buffer_ptr;
    info.element_stride = elem_size;
    info.total_size_bytes = size;
    return info;
}
ResourceCreationInfo VkNativeResourceExt::register_external_texture(
    void *image_ptr,
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    // custom data see backends' header
    void *custom_data) noexcept {
    auto image = static_cast<VkImage>(image_ptr);
    auto size = uint3(width, height, depth);
    auto tex = custom_data ?
                   new Texture(
                       _device, image, dimension,
                       *static_cast<VkFormat *>(custom_data), size,
                       mipmap_levels, false) :
                   new Texture(
                       _device, image, dimension, format, size,
                       mipmap_levels, false);
    ResourceCreationInfo info{
        .handle = reinterpret_cast<uint64_t>(tex),
        .native_handle = image_ptr};
    return info;
}

ResourceCreationInfo VkNativeResourceExt::register_external_depth_buffer(
    void *depth_buffer_ptr,
    DepthFormat format,
    uint width,
    uint height,
    // custom data see backends' header
    void *custom_data) noexcept {
    // NOTE: external depth buffer registration not yet implemented.
    return ResourceCreationInfo::make_invalid();
}

SwapchainCreationInfo VkNativeResourceExt::register_external_swapchain(
    void *swapchain_ptr,
    bool vsync) noexcept {
    // NOTE: external swapchain registration not yet implemented.
    SwapchainCreationInfo s{};
    s.invalidate();
    return s;
}

uint64_t VkNativeResourceExt::get_native_resource_device_address(
    void *native_handle) noexcept {
    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = static_cast<VkBuffer>(native_handle);
    return vkGetBufferDeviceAddress(_device->logic_device(), &buffer_device_address_info);
}

}// namespace lc::vk
