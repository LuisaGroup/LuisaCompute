#include "swapchain.h"
#include <volk.h>
#include "log.h"
#include "device.h"
#include "stream.h"

#include "../common/moltenvk_surface.h"

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_APPLE)
#include <vulkan/vulkan_macos.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#if LUISA_ENABLE_WAYLAND
#include <dlfcn.h>
#include <vulkan/vulkan_wayland.h>
#include <wayland-client.h>
#endif
#else
#error "Unsupported platform"
#endif

namespace lc::vk {
VkCompositeAlphaFlagBitsKHR _choose_composite_alpha(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);
    // Priority: Post-multiplied > Pre-multiplied > Inherit > Opaque
    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
        return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;// Best for transparency
    }
    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
        return VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    }
    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
        return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;// Use native window settings
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// Fallback (no transparency)
}

static const std::array vulkan_swapchain_screen_shader_vertex_bytecode = {
    0x07230203u, 0x00010300u, 0x000d000bu, 0x00000026u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0008000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x00000012u, 0x00000020u,
    0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u,
    0x0000000bu, 0x00000001u, 0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
    0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00030047u, 0x0000000bu, 0x00000002u, 0x00040047u,
    0x00000012u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000020u, 0x0000001eu, 0x00000000u, 0x00020013u,
    0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u,
    0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu,
    0x00000008u, 0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u, 0x0006001eu,
    0x0000000bu, 0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000003u,
    0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu, 0x00000020u,
    0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u, 0x00000006u,
    0x00000002u, 0x00040020u, 0x00000011u, 0x00000001u, 0x00000010u, 0x0004003bu, 0x00000011u, 0x00000012u,
    0x00000001u, 0x0004002bu, 0x00000006u, 0x00000014u, 0x40000000u, 0x0004002bu, 0x00000006u, 0x00000016u,
    0x3f800000u, 0x0004002bu, 0x00000006u, 0x00000019u, 0x00000000u, 0x00040020u, 0x0000001du, 0x00000003u,
    0x00000007u, 0x00040020u, 0x0000001fu, 0x00000003u, 0x00000010u, 0x0004003bu, 0x0000001fu, 0x00000020u,
    0x00000003u, 0x0005002cu, 0x00000010u, 0x00000025u, 0x00000016u, 0x00000016u, 0x00050036u, 0x00000002u,
    0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x00000010u, 0x00000013u,
    0x00000012u, 0x0005008eu, 0x00000010u, 0x00000015u, 0x00000013u, 0x00000014u, 0x00050083u, 0x00000010u,
    0x00000018u, 0x00000015u, 0x00000025u, 0x00050051u, 0x00000006u, 0x0000001au, 0x00000018u, 0x00000000u,
    0x00050051u, 0x00000006u, 0x0000001bu, 0x00000018u, 0x00000001u, 0x00070050u, 0x00000007u, 0x0000001cu,
    0x0000001au, 0x0000001bu, 0x00000019u, 0x00000016u, 0x00050041u, 0x0000001du, 0x0000001eu, 0x0000000du,
    0x0000000fu, 0x0003003eu, 0x0000001eu, 0x0000001cu, 0x0003003eu, 0x00000020u, 0x00000013u, 0x000100fdu,
    0x00010038u};

// source:
// #version 450
// layout(binding = 0) uniform texture2D tex;
// layout(binding = 1) uniform sampler sam;
// layout(location = 0) in vec2 fragTexCoord;
// layout(location = 0) out vec4 outColor;
// void main() {
//     outColor = texture(sampler2D(tex, sam), fragTexCoord);
// }
static const std::array vulkan_swapchain_screen_shader_fragment_bytecode = {
    0x07230203u, 0x00010300u, 0x000d000bu, 0x00000019u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000016u, 0x00030010u,
    0x00000004u, 0x00000007u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000cu,
    0x00000022u, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000021u, 0x00000000u, 0x00040047u, 0x00000010u,
    0x00000022u, 0x00000000u, 0x00040047u, 0x00000010u, 0x00000021u, 0x00000001u, 0x00040047u, 0x00000016u,
    0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
    0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au,
    0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u,
    0x0000000bu, 0x00000000u, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x0002001au,
    0x0000000eu, 0x00040020u, 0x0000000fu, 0x00000000u, 0x0000000eu, 0x0004003bu, 0x0000000fu, 0x00000010u,
    0x00000000u, 0x0003001bu, 0x00000012u, 0x0000000au, 0x00040017u, 0x00000014u, 0x00000006u, 0x00000002u,
    0x00040020u, 0x00000015u, 0x00000001u, 0x00000014u, 0x0004003bu, 0x00000015u, 0x00000016u, 0x00000001u,
    0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
    0x0000000au, 0x0000000du, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x00000011u, 0x00000010u, 0x00050056u,
    0x00000012u, 0x00000013u, 0x0000000du, 0x00000011u, 0x0004003du, 0x00000014u, 0x00000017u, 0x00000016u,
    0x00050057u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u,
    0x000100fdu, 0x00010038u};
struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    luisa::vector<VkSurfaceFormatKHR> formats;
    luisa::vector<VkPresentModeKHR> present_modes;
};
[[nodiscard]] auto _query_swapchain_support(
    VkPhysicalDevice device, VkSurfaceKHR surface) noexcept {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    auto format_count = 0u;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0u) {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }
    auto present_mode_count = 0u;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
    if (present_mode_count != 0u) {
        details.present_modes.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
    }
    return details;
}
void _create_surface(
    uint64_t display_handle, uint64_t window_handle, VkSurfaceKHR &surface, VkInstance instance) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS)
    VkWin32SurfaceCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    create_info.hwnd = reinterpret_cast<HWND>(window_handle);
    create_info.hinstance = GetModuleHandle(nullptr);
    auto vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
    VK_CHECK_RESULT(vkCreateWin32SurfaceKHR(instance, &create_info, Device::alloc_callbacks(), &surface));
#elif defined(LUISA_PLATFORM_APPLE)
    VkMacOSSurfaceCreateInfoMVK create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
    create_info.pView = cocoa_window_content_view(window_handle);
    auto vkCreateMacOSSurfaceMVK = (PFN_vkCreateMacOSSurfaceMVK)vkGetInstanceProcAddr(instance, "vkCreateMacOSSurfaceMVK");
    VK_CHECK_RESULT(vkCreateMacOSSurfaceMVK(instance, &create_info, Device::alloc_callbacks(), &surface));
#else
    static std::once_flag set_xlib_error_handler;
    std::call_once(set_xlib_error_handler, [] {
        XSetErrorHandler([](Display *display, XErrorEvent *error) noexcept {
            char buffer[256] = {};
            XGetErrorText(display, error->error_code, buffer, sizeof(buffer));
            LUISA_WARNING_WITH_LOCATION("Xlib error: {}", buffer);
            return 0;
        });
    });
    auto create_surface_xlib = [&] {
        VkXlibSurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        create_info.dpy = display_handle ? reinterpret_cast<Display *>(display_handle) : XOpenDisplay(nullptr);
        create_info.window = static_cast<Window>(window_handle);
        auto vkCreateXlibSurfaceKHR = (PFN_vkCreateXlibSurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR");
        if (!vkCreateXlibSurfaceKHR) {
            LUISA_ERROR_WITH_LOCATION("vkCreateXlibSurfaceKHR is not available - VK_KHR_xlib_surface extension not enabled");
        }
        VK_CHECK_RESULT(vkCreateXlibSurfaceKHR(instance, &create_info, Device::alloc_callbacks(), &surface));
    };
#if LUISA_ENABLE_WAYLAND
    if (window_handle & 0xffff'ffff'0000'0000ull) {// 64-bit pointer, so likely wayland
        VkWaylandSurfaceCreateInfoKHR create_info_wl{};
        create_info_wl.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        create_info_wl.display = display_handle ? reinterpret_cast<wl_display *>(display_handle) : wl_display_connect(nullptr);
        create_info_wl.surface = reinterpret_cast<wl_surface *>(window_handle);
        auto vkCreateWaylandSurfaceKHR = (PFN_vkCreateWaylandSurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWaylandSurfaceKHR");
        if (!vkCreateWaylandSurfaceKHR) {
            LUISA_WARNING_WITH_LOCATION("vkCreateWaylandSurfaceKHR not available, falling back to Xlib surface.");
            create_surface_xlib();
            return;
        }
        VK_CHECK_RESULT(vkCreateWaylandSurfaceKHR(instance, &create_info_wl, Device::alloc_callbacks(), &surface));
    } else {// X uses 32-bit IDs
        create_surface_xlib();
    }
#else
    create_surface_xlib();
#endif
#endif
}
void _create_render_pass(
    VkSurfaceFormatKHR swapchain_format,
    VkDevice device,
    VkRenderPass &render_pass) noexcept {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = swapchain_format.format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    VK_CHECK_RESULT(vkCreateRenderPass(device, &render_pass_info, Device::alloc_callbacks(), &render_pass));
}

void _create_descriptor_set_layout(
    VkDevice device,
    VkSampler &texture_sampler,
    VkDescriptorSetLayout &descriptor_set_layout) noexcept {

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 0;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler_info, Device::alloc_callbacks(), &texture_sampler));

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

    // binding 0: image
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 1: immutable sampler
    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].pImmutableSamplers = &texture_sampler;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layout_info, Device::alloc_callbacks(), &descriptor_set_layout));
}

void _create_framebuffers(
    VkDevice device,
    luisa::vector<VkImageView> &swapchain_image_views,
    luisa::vector<VkFramebuffer> &swapchain_framebuffers,
    VkRenderPass render_pass,
    VkExtent2D swapchain_extent) noexcept {
    swapchain_framebuffers.resize(swapchain_image_views.size());
    for (auto i = 0u; i < swapchain_image_views.size(); i++) {
        VkImageView attachments[] = {swapchain_image_views[i]};
        VkFramebufferCreateInfo framebuffer_create_info{};
        framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_create_info.renderPass = render_pass;
        framebuffer_create_info.attachmentCount = 1u;
        framebuffer_create_info.pAttachments = attachments;
        framebuffer_create_info.width = std::max(1u, swapchain_extent.width);
        framebuffer_create_info.height = std::max(1u, swapchain_extent.height);
        framebuffer_create_info.layers = 1u;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebuffer_create_info, Device::alloc_callbacks(), &swapchain_framebuffers[i]));
    }
}
void _create_vertex_buffer(
    CommandBuffer &cmdbuffer,
    vstd::vector<vstd::function<void()>> &after_render_callback,
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkBuffer &vertex_buffer,
    VkDeviceMemory &vertex_buffer_memory) noexcept {

    const std::array vertices = {1.f, 0.f,
                                 0.f, 0.f,
                                 0.f, 1.f,
                                 1.f, 0.f,
                                 0.f, 1.f,
                                 1.f, 1.f};
    auto buffer_size = sizeof(vertices);

    auto find_memory_type = [&](uint type_filter, VkMemoryPropertyFlags properties) noexcept {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
            if ((type_filter & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        LUISA_ERROR_WITH_LOCATION("Failed to find suitable memory type.");
    };

    auto create_buffer = [device, &find_memory_type](VkDeviceSize size,
                                                     VkBufferUsageFlags usage,
                                                     VkMemoryPropertyFlags properties) noexcept {
        VkBuffer buffer;
        VkDeviceMemory buffer_memory;
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK_RESULT(vkCreateBuffer(device, &buffer_info, Device::alloc_callbacks(), &buffer));
        VkMemoryRequirements memory_requirements;
        vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = memory_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(memory_requirements.memoryTypeBits, properties);
        VK_CHECK_RESULT(vkAllocateMemory(device, &alloc_info, Device::alloc_callbacks(), &buffer_memory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, buffer_memory, 0));
        return std::make_pair(buffer, buffer_memory);
    };

    auto [staging_buffer, staging_buffer_memory] = create_buffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void *data = nullptr;
    VK_CHECK_RESULT(vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data));
    std::memcpy(data, vertices.data(), buffer_size);
    vkUnmapMemory(device, staging_buffer_memory);

    std::tie(vertex_buffer, vertex_buffer_memory) = create_buffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    auto command_buffer = cmdbuffer.cmdbuffer();
    VkBufferCopy copy_region{};
    copy_region.size = buffer_size;

    {
        VkBufferMemoryBarrier2 vertex_copy_dst_barrier{};
        vertex_copy_dst_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vertex_copy_dst_barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        vertex_copy_dst_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        vertex_copy_dst_barrier.srcAccessMask = VK_ACCESS_2_NONE;
        vertex_copy_dst_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        vertex_copy_dst_barrier.srcQueueFamilyIndex = cmdbuffer.resource_barrier->queue_index;
        vertex_copy_dst_barrier.dstQueueFamilyIndex = vertex_copy_dst_barrier.srcQueueFamilyIndex;
        vertex_copy_dst_barrier.buffer = vertex_buffer;
        vertex_copy_dst_barrier.offset = 0;
        vertex_copy_dst_barrier.size = buffer_size;
        VkDependencyInfo vertex_copy_dst_info{};
        vertex_copy_dst_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        vertex_copy_dst_info.bufferMemoryBarrierCount = 1;
        vertex_copy_dst_info.pBufferMemoryBarriers = &vertex_copy_dst_barrier;
        vkCmdPipelineBarrier2(cmdbuffer.cmdbuffer(), &vertex_copy_dst_info);
    }

    vkCmdCopyBuffer(command_buffer, staging_buffer, vertex_buffer, 1, &copy_region);
    {
        VkBufferMemoryBarrier2 vertex_copy_dst_barrier{};
        vertex_copy_dst_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vertex_copy_dst_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        vertex_copy_dst_barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        vertex_copy_dst_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        vertex_copy_dst_barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        vertex_copy_dst_barrier.srcQueueFamilyIndex = cmdbuffer.resource_barrier->queue_index;
        vertex_copy_dst_barrier.dstQueueFamilyIndex = vertex_copy_dst_barrier.srcQueueFamilyIndex;
        vertex_copy_dst_barrier.buffer = vertex_buffer;
        vertex_copy_dst_barrier.offset = 0;
        vertex_copy_dst_barrier.size = buffer_size;
        VkDependencyInfo vertex_copy_dst_info{};
        vertex_copy_dst_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        vertex_copy_dst_info.bufferMemoryBarrierCount = 1;
        vertex_copy_dst_info.pBufferMemoryBarriers = &vertex_copy_dst_barrier;
        vkCmdPipelineBarrier2(cmdbuffer.cmdbuffer(), &vertex_copy_dst_info);
    }
    // after commit
    after_render_callback.emplace_back(
        [device, staging_buffer, staging_buffer_memory]() {
            vkDestroyBuffer(device, staging_buffer, Device::alloc_callbacks());
            vkFreeMemory(device, staging_buffer_memory, Device::alloc_callbacks());
        });
}

void _record_command_buffer(
    VkCommandBuffer command_buffer,
    VkRenderPass render_pass,
    luisa::vector<VkFramebuffer> &swapchain_framebuffers,
    VkExtent2D const &swapchain_extent,
    VkBuffer vertex_buffer,
    VkDescriptorSet &descriptor_set,
    VkPipeline graphics_pipeline,
    VkPipelineLayout pipeline_layout,
    uint current_frame,
    uint image_index) {

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass;
    render_pass_info.framebuffer = swapchain_framebuffers[image_index];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swapchain_extent;
    VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;
    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = std::max(1.f, static_cast<float>(swapchain_extent.width));
    viewport.height = std::max(1.f, static_cast<float>(swapchain_extent.height));
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchain_extent;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    auto vertex_buffer_offset = static_cast<VkDeviceSize>(0u);
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &vertex_buffer_offset);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    vkCmdDraw(command_buffer, 6u, 1, 0, 0);
    vkCmdEndRenderPass(command_buffer);
}

void _create_descriptor_sets(
    VkDevice device,
    luisa::vector<VkImage> &swapchain_images,
    VkDescriptorSet &descriptor_set,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDescriptorPool desc_pool) noexcept {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set));
}

void _create_pipeline(
    VkDevice device,
    VkPipelineLayout &pipeline_layout,
    VkDescriptorSetLayout &descriptor_set_layout,
    VkPipeline &graphics_pipeline,
    VkRenderPass &render_pass) noexcept {

    auto create_shader_module = [&](auto code) noexcept {
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size_bytes();
        create_info.pCode = code.data();
        VkShaderModule shader_module;
        VK_CHECK_RESULT(vkCreateShaderModule(device, &create_info, Device::alloc_callbacks(), &shader_module));
        return shader_module;
    };
    auto vert_shader = create_shader_module(luisa::span{vulkan_swapchain_screen_shader_vertex_bytecode});
    auto frag_shader = create_shader_module(luisa::span{vulkan_swapchain_screen_shader_fragment_bytecode});

    VkPipelineShaderStageCreateInfo vertex_stage_info{};
    vertex_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertex_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertex_stage_info.module = vert_shader;
    vertex_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo fragment_stage_info{};
    fragment_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragment_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragment_stage_info.module = frag_shader;
    fragment_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = {vertex_stage_info, fragment_stage_info};

    VkVertexInputBindingDescription vertex_description{};
    vertex_description.binding = 0;
    vertex_description.stride = sizeof(float) * 2;
    vertex_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribute_description{};
    attribute_description.binding = 0;
    attribute_description.location = 0;
    attribute_description.format = VK_FORMAT_R32G32_SFLOAT;
    attribute_description.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.vertexAttributeDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &vertex_description;
    vertex_input_info.pVertexAttributeDescriptions = &attribute_description;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamicState.pDynamicStates = dynamic_states.data();

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipeline_layout_info, Device::alloc_callbacks(), &pipeline_layout));

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamicState;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, Device::alloc_callbacks(), &graphics_pipeline));

    vkDestroyShaderModule(device, vert_shader, Device::alloc_callbacks());
    vkDestroyShaderModule(device, frag_shader, Device::alloc_callbacks());
}

[[nodiscard]] static auto _is_hdr_colorspace(VkColorSpaceKHR colorspace) noexcept {
    return colorspace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
}

[[nodiscard]] static auto _colorspace_name(VkColorSpaceKHR colorspace) noexcept {
    switch (colorspace) {
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return "sRGB (non-linear)";
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
            return "Display P3 (non-linear)";
        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
            return "Extended sRGB (linear)";
        case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
            return "Display P3 (linear)";
        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            return "DCI P3 (non-linear)";
        case VK_COLOR_SPACE_BT709_LINEAR_EXT:
            return "BT709 (linear)";
        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            return "BT709 (non-linear)";
        case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
            return "BT2020 (linear)";
        case VK_COLOR_SPACE_HDR10_ST2084_EXT:
            return "HDR10 (ST2084)";
        case VK_COLOR_SPACE_DOLBYVISION_EXT:
            return "Dolby Vision";
        case VK_COLOR_SPACE_HDR10_HLG_EXT:
            return "HDR10 (HLG)";
        case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
            return "Adobe RGB (linear)";
        case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
            return "Adobe RGB (non-linear)";
        case VK_COLOR_SPACE_PASS_THROUGH_EXT:
            return "Pass-through";
        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
            return "Extended sRGB (non-linear)";
        case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
            return "Display Native";
        default:
            break;
    }
    return "Unknown";
}

void Swapchain::create_swapchain(
    uint64_t display_handle,
    uint64_t window_handle,
    uint width, uint height, uint back_buffers,
    bool is_recreation, bool allow_hdr, bool vsync,
    bool transparent) {
    _display_handle = display_handle;
    _window_handle = window_handle;
    _requested_size = uint2(width, height);
    _requested_hdr = allow_hdr;
    _requested_vsync = vsync;

    // Only create surface on first creation, not during recreation
    // Surface is destroyed in destructor
    if (!is_recreation) {
        _create_surface(
            display_handle,
            window_handle,
            _surface,
            Device::instance());
    }

    auto support = _query_swapchain_support(
        device()->physical_device(), _surface);
    if (support.capabilities.maxImageCount == 0u) { support.capabilities.maxImageCount = std::max(back_buffers, support.capabilities.minImageCount); }
    if (!is_recreation) {// only allow change back buffer count and swapchain format on first creation
        back_buffers = std::clamp(
            back_buffers,
            support.capabilities.minImageCount,
            support.capabilities.maxImageCount);
        _swapchain_format = [&formats = support.formats, allow_hdr] {
            for (auto f : formats) {
                LUISA_VERBOSE_WITH_LOCATION(
                    "Supported swapchain format: "
                    "colorspace = {}, format = {}",
                    luisa::to_string(f.colorSpace),
                    luisa::to_string(f.format));
            }
            if (allow_hdr) {
                for (auto format : formats) {
                    if (format.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
                        return format;
                    }
                }
                for (auto format : formats) {
                    if (format.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                        return format;
                    }
                }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && format.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
                    return format;
                }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                    (format.format == VK_FORMAT_R8G8B8A8_UNORM ||
                     format.format == VK_FORMAT_B8G8R8A8_UNORM)) {
                    return format;
                }
            }
            for (auto format : formats) {
                if (format.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                    return format;
                }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                    (format.format == VK_FORMAT_R8G8B8A8_SRGB ||
                     format.format == VK_FORMAT_B8G8R8A8_SRGB)) {
                    return format;
                }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return format;
                }
            }
            return formats.front();
        }();
    }

    _requested_transparent = transparent;
    _swapchain_extent = [&capabilities = support.capabilities, width, height] {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() &&
            capabilities.currentExtent.height != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        VkExtent2D actual_extent{width, height};
        actual_extent.width = std::clamp(actual_extent.width,
                                         capabilities.minImageExtent.width,
                                         capabilities.maxImageExtent.width);
        actual_extent.height = std::clamp(actual_extent.height,
                                          capabilities.minImageExtent.height,
                                          capabilities.maxImageExtent.height);
        return actual_extent;
    }();

    auto present_mode = [&present_modes = support.present_modes, vsync] {
        if (!vsync) {// try to use mailbox mode if vsync is disabled
            for (auto mode : present_modes) {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR) { return mode; }
            }
            for (auto mode : present_modes) {
                if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) { return mode; }
            }
        }
        LUISA_ASSERT(std::find(present_modes.cbegin(),
                               present_modes.cend(),
                               VK_PRESENT_MODE_FIFO_KHR) != present_modes.cend(),
                     "FIFO present mode is not supported.");
        return VK_PRESENT_MODE_FIFO_KHR;
    }();

    // create the swapchain
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = _surface;
    create_info.minImageCount = back_buffers;
    create_info.imageFormat = _swapchain_format.format;
    create_info.imageColorSpace = _swapchain_format.colorSpace;
    create_info.imageExtent = _swapchain_extent;
    create_info.imageArrayLayers = 1u;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    if (transparent) {
        create_info.compositeAlpha = _choose_composite_alpha(device()->physical_device(), _surface);
    }
    auto logic_device = device()->logic_device();
    if (!vkCreateSwapchainKHR) {
        LUISA_ERROR_WITH_LOCATION("vkCreateSwapchainKHR is null - volkLoadDevice not called or VK_KHR_swapchain not enabled");
    }
    VK_CHECK_RESULT(vkCreateSwapchainKHR(logic_device, &create_info, Device::alloc_callbacks(), &_swapchain));

    // get the swapchain images - first query count, then resize, then fetch
    uint32_t image_count = 0;
    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(logic_device, _swapchain, &image_count, nullptr));
    if (image_count == 0) {
        LUISA_ERROR_WITH_LOCATION("Swapchain image count is zero");
    }
    if (image_count != back_buffers) {
        LUISA_WARNING_WITH_LOCATION("Swapchain image count mismatch: requested = {}, actual = {}.", back_buffers, image_count);
    }
    _swapchain_images.resize(image_count);
    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(logic_device, _swapchain, &image_count, _swapchain_images.data()));

    // create the swapchain image views
    _swapchain_image_views.resize(image_count);
    for (auto i = 0u; i < image_count; i++) {
        VkImageViewCreateInfo image_view_create_info{};
        image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        image_view_create_info.image = _swapchain_images[i];
        image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        image_view_create_info.format = _swapchain_format.format;
        image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_view_create_info.subresourceRange.baseMipLevel = 0u;
        image_view_create_info.subresourceRange.levelCount = 1u;
        image_view_create_info.subresourceRange.baseArrayLayer = 0u;
        image_view_create_info.subresourceRange.layerCount = 1u;
        VK_CHECK_RESULT(vkCreateImageView(logic_device, &image_view_create_info, Device::alloc_callbacks(), &_swapchain_image_views[i]));
    }
    LUISA_INFO("Created swapchain: {}x{} with {} back buffer(s) in {} (format = {}, mode = {}).",
               _swapchain_extent.width, _swapchain_extent.height, _swapchain_images.size(),
               _colorspace_name(_swapchain_format.colorSpace),
               luisa::to_string(_swapchain_format.format),
               luisa::to_string(present_mode));

    // Only create persistent resources on first creation, not during recreation
    if (!is_recreation) {
        _create_render_pass(_swapchain_format, device()->logic_device(), _render_pass);
        _create_descriptor_set_layout(device()->logic_device(), _texture_sampler, _descriptor_set_layout);
        _create_pipeline(device()->logic_device(), _pipeline_layout, _descriptor_set_layout, _graphics_pipeline, _render_pass);
    }

    if (_image_available_semaphores.size() != _swapchain_images.size()) {
        for (auto &sem : _image_available_semaphores) {
            vkDestroySemaphore(device()->logic_device(), sem, Device::alloc_callbacks());
        }
        for (auto &sem : _render_finished_semaphores) {
            vkDestroySemaphore(device()->logic_device(), sem, Device::alloc_callbacks());
        }
        for (auto &f : _in_flight_fences) {
            vkDestroyFence(device()->logic_device(), f, Device::alloc_callbacks());
        }

        _image_available_semaphores.resize(_swapchain_images.size());
        _render_finished_semaphores.resize(_swapchain_images.size());
        _in_flight_fences.resize(_swapchain_images.size());

        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (auto i : vstd::range((int64_t)_swapchain_images.size())) {
            VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), &_in_flight_fences[i]));
            VK_CHECK_RESULT(vkCreateSemaphore(device()->logic_device(), &semaphore_info, Device::alloc_callbacks(), &_image_available_semaphores[i]));
            VK_CHECK_RESULT(vkCreateSemaphore(device()->logic_device(), &semaphore_info, Device::alloc_callbacks(), &_render_finished_semaphores[i]));
        }
    }

    // Framebuffers are always recreated (they depend on swapchain images)
    _create_framebuffers(
        device()->logic_device(),
        _swapchain_image_views,
        _swapchain_framebuffers,
        _render_pass,
        _swapchain_extent);
}

Swapchain::Swapchain(Device *device)
    : Resource(device) {
    if (!device->enable_surface_feature()) [[unlikely]] {
        LUISA_ERROR("Surface not enabled, Swapcchain can not be created.");
    }
}
void Swapchain::_recreate_swapchain() {
    auto back_buffers = _swapchain_framebuffers.size();
    auto device = this->device()->logic_device();
    vkDeviceWaitIdle(device);

    _destroy_swapchain();

    create_swapchain(
        _display_handle, _window_handle,
        _requested_size.x, _requested_size.y,
        back_buffers, true,
        _requested_hdr, _requested_vsync,
        _requested_transparent);

    _current_frame = 0;
}

bool Swapchain::is_hdr() const noexcept {
    return _is_hdr_colorspace(_swapchain_format.colorSpace);
}

void Swapchain::_destroy_swapchain() {
    auto device = this->device()->logic_device();
    // Destroy framebuffers BEFORE image views (framebuffers depend on views)
    for (auto &i : _swapchain_framebuffers) {
        vkDestroyFramebuffer(device, i, Device::alloc_callbacks());
    }
    for (auto &i : _swapchain_image_views) {
        vkDestroyImageView(
            device,
            i,
            Device::alloc_callbacks());
    }
    _swapchain_framebuffers.clear();
    _swapchain_image_views.clear();
    _swapchain_images.clear();
    vkDestroySwapchainKHR(device, _swapchain, Device::alloc_callbacks());
}

Swapchain::~Swapchain() {
    auto device = this->device()->logic_device();
    vkDeviceWaitIdle(device);
    _destroy_swapchain();
    vkDestroySurfaceKHR(
        Device::instance(),
        _surface,
        Device::alloc_callbacks());
    for (auto &i : _image_available_semaphores) {
        vkDestroySemaphore(device, i, Device::alloc_callbacks());
    }
    for (auto &i : _render_finished_semaphores) {
        vkDestroySemaphore(device, i, Device::alloc_callbacks());
    }
    for (auto &i : _in_flight_fences) {
        vkDestroyFence(device, i, Device::alloc_callbacks());
    }

    vkDestroyPipeline(device, _graphics_pipeline, Device::alloc_callbacks());
    vkDestroyPipelineLayout(device, _pipeline_layout, Device::alloc_callbacks());
    vkDestroyBuffer(device, _vertex_buffer, Device::alloc_callbacks());
    vkFreeMemory(device, _vertex_buffer_memory, Device::alloc_callbacks());
    vkDestroySampler(device, _texture_sampler, Device::alloc_callbacks());
    vkDestroyDescriptorSetLayout(device, _descriptor_set_layout, Device::alloc_callbacks());
    vkDestroyRenderPass(device, _render_pass, Device::alloc_callbacks());
}

void Swapchain::handle_present_error() {
    _recreate_swapchain();
}

void Swapchain::present(
    CommandBuffer &cmdbuffer,
    VkSemaphore &wait, VkSemaphore &signal,
    VkPipelineStageFlags &wait_stage,
    // present info
    VkSemaphore &present_wait,
    uint &image_index,
    Texture const *tex,
    VkFence &fence,
    uint mip) {
    fence = nullptr;
    image_index = 0u;
    VK_CHECK_RESULT(vkWaitForFences(
        device()->logic_device(), 1, &_in_flight_fences[_current_frame],
        VK_TRUE, UINT64_MAX));
    if (auto ret = vkAcquireNextImageKHR(
            device()->logic_device(), _swapchain, UINT64_MAX,
            _image_available_semaphores[_current_frame],
            VK_NULL_HANDLE, &image_index);
        ret == VK_ERROR_OUT_OF_DATE_KHR) {
        _recreate_swapchain();
        return;
    } else if (ret != VK_SUCCESS && ret != VK_SUBOPTIMAL_KHR) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to acquire swapchain image: {}.",
            luisa::to_string(ret));
    }
    VK_CHECK_RESULT(vkResetFences(device()->logic_device(), 1, &_in_flight_fences[_current_frame]));
    // Create vertex buffer only after successful acquisition
    if (!_vertex_buffer) {
        _create_vertex_buffer(cmdbuffer, cmdbuffer.states()->callbacks, device()->physical_device(), device()->logic_device(), _vertex_buffer, _vertex_buffer_memory);
    }
    VkDescriptorSet descriptor_set;
    _create_descriptor_sets(
        device()->logic_device(),
        _swapchain_images,
        descriptor_set,
        _descriptor_set_layout,
        cmdbuffer.states()->desc_pool);
    auto image = tex->vk_image();
    auto image_format = Texture::to_vk_format(tex->format());
    cmdbuffer.resource_barrier->record(
        TexView{tex, mip},
        ResourceBarrier::Usage::kRasterRead);
    cmdbuffer.resource_barrier->update_states(cmdbuffer.cmdbuffer());

    auto image_layout = cmdbuffer.resource_barrier->get_layout(tex, mip);
    VkImageViewCreateInfo image_view_create_info{};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = image,
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = image_format;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = mip;
    image_view_create_info.subresourceRange.levelCount = 1u;
    image_view_create_info.subresourceRange.baseArrayLayer = 0u;
    image_view_create_info.subresourceRange.layerCount = 1u;
    VkImageView img_view;
    VK_CHECK_RESULT(vkCreateImageView(
        device()->logic_device(),
        &image_view_create_info,
        Device::alloc_callbacks(),
        &img_view));
    cmdbuffer.states()->callbacks.emplace_back([img_view, device = this->device()->logic_device()]() {
        vkDestroyImageView(device, img_view, Device::alloc_callbacks());
    });

    // update descriptor set if necessary
    VkDescriptorImageInfo image_info{};
    image_info.imageView = img_view;
    image_info.imageLayout = image_layout;
    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstBinding = 0u;
    descriptor_write.dstArrayElement = 0u;
    descriptor_write.dstSet = descriptor_set;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_write.descriptorCount = 1u;
    descriptor_write.pImageInfo = &image_info;
    vkUpdateDescriptorSets(device()->logic_device(), 1u, &descriptor_write, 0u, nullptr);

    _record_command_buffer(
        cmdbuffer.cmdbuffer(),
        _render_pass,
        _swapchain_framebuffers,
        _swapchain_extent,
        _vertex_buffer,
        descriptor_set,
        _graphics_pipeline,
        _pipeline_layout,
        _current_frame,
        image_index);
    // submit command buffer
    wait = _image_available_semaphores[_current_frame];
    fence = _in_flight_fences[_current_frame];
    signal = _render_finished_semaphores[image_index];
    wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    present_wait = _render_finished_semaphores[image_index];
    _current_frame = (_current_frame + 1u) % _swapchain_images.size();
}
}// namespace lc::vk