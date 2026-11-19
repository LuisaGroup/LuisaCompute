//
// Created by mike on 1/11/26.
//

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <volk.h>

#if defined(LUISA_PLATFORM_WINDOWS)
#include "../common/windows_security_attributes.h"
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#else
#error "Unsupported platform"
#endif

#include "../common/vulkan_instance.h"
#include <luisa/backends/common/vulkan_swapchain.h>
#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_device.h"
#include "hip_stream.h"
#include "hip_texture.h"
#include "hip_swapchain.h"

namespace luisa::compute::hip {

struct HIPVulkanSyncContext {
    VkDevice device;
    VkSemaphore semaphore;
    uint64_t value;

    [[nodiscard]] static auto &pool() noexcept {
        static Pool<HIPVulkanSyncContext> pool;
        return pool;
    }

    template<typename... Args>
    [[nodiscard]] static auto create(Args &&...args) noexcept {
        return pool().create(std::forward<Args>(args)...);
    }

    void recycle() noexcept {
        pool().destroy(this);
    }
};
static void luisa_hip_vulkan_sync_callback(void *ptr) noexcept {
    auto ctx = static_cast<HIPVulkanSyncContext *>(ptr);
    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = ctx->semaphore;
    signal_info.value = ctx->value;
    LUISA_CHECK_VULKAN(vkSignalSemaphore(ctx->device, &signal_info));
    ctx->recycle();
}

class HIPSwapchain::Impl {

private:
    static constexpr std::array required_extensions{
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef LUISA_PLATFORM_WINDOWS
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
    };

private:
    VulkanSwapchain _base;
    uint2 _size;
    uint _current_frame{0u};
    spin_mutex _present_mutex;
    spin_mutex _name_mutex;
    luisa::string _name;

    // vulkan objects
    VkImage _image{nullptr};
    VkDeviceMemory _image_memory{nullptr};
    VkDeviceSize _image_memory_size{};
    VkImageView _image_view{nullptr};

    // HIP objects
    hipExternalMemory_t _hip_ext_image_memory{};
    void *_hip_ext_image_buffer{nullptr};
    size_t _hip_ext_image_stride{0u};
    VkSemaphore _sync_semaphore{nullptr};
    uint64_t _sync_value{0u};

    [[nodiscard]] auto _find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) noexcept {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(_base.physical_device(), &memory_properties);
        for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
            if ((type_filter & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        LUISA_ERROR_WITH_LOCATION("Failed to find suitable memory type.");
    }

    [[nodiscard]] auto _choose_image_format() const noexcept {
        return _base.is_hdr() ?
                   VK_FORMAT_R16G16B16A16_SFLOAT :
                   VK_FORMAT_R8G8B8A8_SRGB;
    }

    void _create_image() noexcept {

        VkExternalMemoryImageCreateInfo external_memory_info{};
        external_memory_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef LUISA_PLATFORM_WINDOWS
        external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = _size.x;
        image_info.extent.height = _size.y;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = _choose_image_format();
        image_info.tiling = VK_IMAGE_TILING_LINEAR;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.pNext = &external_memory_info;
        LUISA_CHECK_VULKAN(vkCreateImage(_base.device(), &image_info, nullptr, &_image));

        // compute memory requirements
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(_base.device(), _image, &mem_requirements);
        _image_memory_size = mem_requirements.size;

        VkImageSubresource subresource{};
        subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource.mipLevel = 0;
        subresource.arrayLayer = 0;
        VkSubresourceLayout layout;
        vkGetImageSubresourceLayout(_base.device(), _image, &subresource, &layout);
        _hip_ext_image_stride = layout.rowPitch;

#ifdef LUISA_PLATFORM_WINDOWS
        WindowsSecurityAttributes security_attributes;
        VkExportMemoryWin32HandleInfoKHR export_memory_info{};
        export_memory_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        export_memory_info.pAttributes = security_attributes.get();
        export_memory_info.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
        export_memory_info.name = nullptr;
#endif

        VkExportMemoryAllocateInfo export_allocate_info{};
        export_allocate_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;

#ifdef LUISA_PLATFORM_WINDOWS
        export_allocate_info.pNext = IsWindows8OrGreater() ? &export_memory_info : nullptr;
        export_allocate_info.handleTypes =
            IsWindows8OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
        export_allocate_info.pNext = nullptr;
        export_allocate_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = _find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        alloc_info.pNext = &export_allocate_info;
        LUISA_CHECK_VULKAN(vkAllocateMemory(_base.device(), &alloc_info, nullptr, &_image_memory));
        LUISA_CHECK_VULKAN(vkBindImageMemory(_base.device(), _image, _image_memory, 0));
    }

    void _transition_image_layout(VkImageLayout old_layout,
                                  VkImageLayout new_layout) noexcept {

        // create a single-use command buffer
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = _base.command_pool();
        alloc_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer;
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_base.device(), &alloc_info, &command_buffer));

        // begin recording
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        // transition image layout
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = _image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage;
        VkPipelineStageFlags dst_stage;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = 0;
            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            LUISA_ERROR_WITH_LOCATION("Unsupported layout transition.");
        }
        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        // end recording
        LUISA_CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        // submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_base.queue(), 1, &submit_info, VK_NULL_HANDLE));
        LUISA_CHECK_VULKAN(vkQueueWaitIdle(_base.queue()));

        // free command buffer
        vkFreeCommandBuffers(_base.device(), _base.command_pool(), 1, &command_buffer);
    }

    void _create_image_view() noexcept {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = _image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = _choose_image_format();
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        LUISA_CHECK_VULKAN(vkCreateImageView(_base.device(), &view_info, nullptr, &_image_view));
    }

    void _hip_import_image() noexcept {

        auto vulkan_image_memory_handle = [this](auto type) noexcept {
            auto device = _base.device();
#ifdef LUISA_PLATFORM_WINDOWS
            auto fp_vkGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
            LUISA_ASSERT(fp_vkGetMemoryWin32HandleKHR != nullptr,
                         "Failed to load vkGetMemoryWin32HandleKHR function.");
            HANDLE handle{};
            VkMemoryGetWin32HandleInfoKHR handle_info{};
            handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            handle_info.pNext = nullptr;
            handle_info.memory = _image_memory;
            handle_info.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(type);
            LUISA_CHECK_VULKAN(fp_vkGetMemoryWin32HandleKHR(device, &handle_info, &handle));
            return handle;
#else
            auto fp_vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
            LUISA_ASSERT(fp_vkGetMemoryFdKHR != nullptr,
                         "Failed to load vkGetMemoryFdKHR function.");
            auto fd = 0;
            VkMemoryGetFdInfoKHR fd_info{};
            fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            fd_info.pNext = nullptr;
            fd_info.memory = _image_memory;
            fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
            LUISA_CHECK_VULKAN(fp_vkGetMemoryFdKHR(device, &fd_info, &fd));
            return fd;
#endif
        };

        hipExternalMemoryHandleDesc hip_ext_memory_handle{};
#ifdef LUISA_PLATFORM_WINDOWS
        hip_ext_memory_handle.type = IsWindows8OrGreater() ?
                                         hipExternalMemoryHandleTypeOpaqueWin32 :
                                         hipExternalMemoryHandleTypeOpaqueWin32Kmt;
        hip_ext_memory_handle.handle.win32.handle = vulkan_image_memory_handle(
            IsWindows8OrGreater() ?
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        hip_ext_memory_handle.type = hipExternalMemoryHandleTypeOpaqueFd;
        hip_ext_memory_handle.handle.fd = vulkan_image_memory_handle(
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
        hip_ext_memory_handle.size = _image_memory_size;
        LUISA_CHECK_HIP(hipImportExternalMemory(&_hip_ext_image_memory, &hip_ext_memory_handle));

        hipExternalMemoryBufferDesc hip_ext_buffer_desc{};
        hip_ext_buffer_desc.offset = 0;
        hip_ext_buffer_desc.size = _image_memory_size;
        hip_ext_buffer_desc.flags = 0;
        LUISA_CHECK_HIP(hipExternalMemoryGetMappedBuffer(
            &_hip_ext_image_buffer, _hip_ext_image_memory, &hip_ext_buffer_desc));
    }

private:
    void _initialize() noexcept {
        // vulkan objects
        _create_image();
        _transition_image_layout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        _transition_image_layout(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        _create_image_view();
        // cuda objects
        _hip_import_image();
        _sync_value = 0u;
        VkSemaphoreTypeCreateInfo type_info{};
        type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type_info.initialValue = 0u;
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphore_info.pNext = &type_info;
        LUISA_CHECK_VULKAN(vkCreateSemaphore(_base.device(), &semaphore_info, nullptr, &_sync_semaphore));
    }

    void _cleanup() noexcept {
        auto device = _base.device();
        // Finish both HIP (including host-func semaphore signaling) and Vulkan
        // (presenting from the shared image and waiting on the timeline
        // semaphore) before releasing any shared or Vulkan-side resource.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
        LUISA_CHECK_VULKAN(vkDeviceWaitIdle(device));
        // HIP objects
        if (_hip_ext_image_buffer != nullptr) {
            LUISA_CHECK_HIP(hipFree(_hip_ext_image_buffer));
            _hip_ext_image_buffer = nullptr;
        }
        LUISA_CHECK_HIP(hipDestroyExternalMemory(_hip_ext_image_memory));
        // Vulkan objects
        vkDestroySemaphore(device, _sync_semaphore, nullptr);
        vkDestroyImageView(device, _image_view, nullptr);
        vkDestroyImage(device, _image, nullptr);
        vkFreeMemory(device, _image_memory, nullptr);
    }

public:
    Impl(hipUUID_t device_uuid,
         uint64_t display_handle, uint64_t window_handle,
         uint width, uint height, bool allow_hdr,
         bool vsync, uint back_buffer_size) noexcept
        : _base{luisa::bit_cast<VulkanDeviceUUID>(device_uuid),
                display_handle, window_handle, width, height,
                allow_hdr, vsync, back_buffer_size, required_extensions},
          _size{make_uint2(width, height)} { _initialize(); }

    ~Impl() noexcept { _cleanup(); }

    [[nodiscard]] VulkanSwapchain *native_handle() noexcept { return &_base; }

    [[nodiscard]] auto pixel_storage() const noexcept {
        return _base.is_hdr() ? PixelStorage::HALF4 : PixelStorage::BYTE4;
    }

    [[nodiscard]] auto size() const noexcept { return _size; }

    void set_name(luisa::string name) noexcept {
        std::scoped_lock lock{_name_mutex};
        _name = std::move(name);
    }

    void present(hipStream_t stream, hipArray_t image) noexcept {

        auto name = [this] {
            std::scoped_lock lock{_name_mutex};
            return _name;
        }();

        std::scoped_lock lock{_present_mutex};

        // wait for the frame to be ready
        _base.wait_for_fence();

        // copy image to swapchain image
        HIP_MEMCPY3D copy{};
        copy.srcMemoryType = hipMemoryTypeArray;
        copy.srcArray = image;
        copy.dstMemoryType = hipMemoryTypeDevice;
        copy.dstDevice = _hip_ext_image_buffer;
        copy.dstPitch = _hip_ext_image_stride;
        copy.dstHeight = _size.y;
        copy.WidthInBytes = pixel_storage_size(pixel_storage(), make_uint3(_size.x, 1u, 1u));
        copy.Height = _size.y;
        copy.Depth = 1u;
        LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, stream));

        // ROCm doesn't support external semaphores on Linux yet,
        // so we have to use timeline semaphores signaled from the host.
        // The same path is used on Windows because the external
        // semaphore interop seems to be broken.
        auto sync_value = ++_sync_value;
        auto ctx = HIPVulkanSyncContext::create(
            _base.device(), _sync_semaphore, sync_value);
        LUISA_CHECK_HIP(hipLaunchHostFunc(stream, luisa_hip_vulkan_sync_callback, ctx));
        _base.present(nullptr, nullptr, _image_view,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      _sync_semaphore, sync_value);

        // update current frame index
        _current_frame = (_current_frame + 1u) % _base.back_buffer_count();
    }
};

HIPSwapchain::HIPSwapchain(HIPDevice *device, SwapchainOption o) noexcept
    : _impl{luisa::make_unique<Impl>(device->device_uuid_for_vulkan(),
                                     o.display, o.window, o.size.x, o.size.y,
                                     o.wants_hdr, o.wants_vsync, o.back_buffer_count)} {}

HIPSwapchain::~HIPSwapchain() noexcept = default;

VulkanSwapchain *HIPSwapchain::native_handle() const noexcept { return _impl->native_handle(); }
PixelStorage HIPSwapchain::pixel_storage() const noexcept { return _impl->pixel_storage(); }

void HIPSwapchain::present(HIPStream *stream, HIPTexture *image) noexcept {
    LUISA_ASSERT(image->storage() == _impl->pixel_storage(),
                 "Image pixel format must match the swapchain.");
    LUISA_ASSERT(all(image->size() == make_uint3(_impl->size(), 1u)),
                 "Image size and pixel format must match the swapchain.");
    _impl->present(stream->handle(), image->level(0u));
}

void HIPSwapchain::set_name(luisa::string name) noexcept {
    _impl->set_name(std::move(name));
}

}// namespace luisa::compute::hip

#endif
