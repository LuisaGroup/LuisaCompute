#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <volk.h>

#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/platform.h>

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
#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_texture.h"
#include "cuda_swapchain.h"

namespace luisa::compute::cuda {

class CUDASwapchain::Impl {

private:
    static constexpr std::array required_extensions{
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef LUISA_PLATFORM_WINDOWS
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
    };

private:
    VulkanSwapchain _base;
    uint2 _size;
    uint _current_frame{0u};
    bool _has_presented_frame{false};
    spin_mutex _present_mutex;
    spin_mutex _name_mutex;
    luisa::string _name;

private:
    // vulkan objects
    VkImage _image{nullptr};
    VkDeviceMemory _image_memory{nullptr};
    VkDeviceSize _image_memory_size{};
    VkImageView _image_view{nullptr};
    luisa::vector<VkSemaphore> _semaphores{};
    VkSemaphore _released_semaphore{};
    VkCommandBuffer _acquire_image_command{};
    VkCommandBuffer _release_image_command{};

private:
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
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
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

    void _initialize_image_ownership() noexcept {

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

        // Initialize the image on the graphics queue before releasing it to CUDA.
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = _image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = _base.queue_family_index();
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
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

    [[nodiscard]] VkCommandBuffer _create_image_handoff_command(bool acquire) noexcept {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = _base.command_pool();
        alloc_info.commandBufferCount = 1u;
        VkCommandBuffer command_buffer{};
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_base.device(), &alloc_info, &command_buffer));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // The semaphore chain serializes image accesses, but a later frame can
        // submit these immutable command buffers while an earlier one is pending.
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = acquire ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = acquire ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : _base.queue_family_index();
        barrier.dstQueueFamilyIndex = acquire ? _base.queue_family_index() : VK_QUEUE_FAMILY_EXTERNAL;
        barrier.srcAccessMask = acquire ? 0u : VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = acquire ? VK_ACCESS_SHADER_READ_BIT : 0u;
        barrier.image = _image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.layerCount = 1u;
        vkCmdPipelineBarrier(command_buffer,
                             acquire ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             acquire ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        LUISA_CHECK_VULKAN(vkEndCommandBuffer(command_buffer));
        return command_buffer;
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

    void _create_semaphores() noexcept {

        VkSemaphoreCreateInfo semaphore_info = {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkExportSemaphoreCreateInfoKHR export_info = {};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef LUISA_PLATFORM_WINDOWS
        export_info.handleTypes = IsWindows8OrGreater() ?
                                      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                                      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
        export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        semaphore_info.pNext = &export_info;

        auto device = _base.device();
        auto n = _base.back_buffer_count();
        _semaphores.resize(n);
        for (uint32_t i = 0u; i < n; i++) {
            LUISA_CHECK_VULKAN(vkCreateSemaphore(device, &semaphore_info, nullptr, &_semaphores[i]));
        }
        LUISA_CHECK_VULKAN(vkCreateSemaphore(device, &semaphore_info, nullptr, &_released_semaphore));
    }

private:
    // cuda objects
    CUexternalMemory _cuda_ext_image_memory{};
    CUmipmappedArray _cuda_ext_image_mipmapped_array{};
    CUarray _cuda_ext_image_array{};
    luisa::vector<CUexternalSemaphore> _cuda_ext_semaphores;
    CUexternalSemaphore _cuda_released_semaphore{};

private:
    void _cuda_import_image() noexcept {

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

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC cuda_ext_memory_handle{};
#ifdef LUISA_PLATFORM_WINDOWS
        cuda_ext_memory_handle.type = IsWindows8OrGreater() ?
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 :
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT;
        cuda_ext_memory_handle.handle.win32.handle = vulkan_image_memory_handle(
            IsWindows8OrGreater() ?
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        cuda_ext_memory_handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        cuda_ext_memory_handle.handle.fd = vulkan_image_memory_handle(
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
        cuda_ext_memory_handle.size = _image_memory_size;
        LUISA_CHECK_CUDA(cuImportExternalMemory(&_cuda_ext_image_memory, &cuda_ext_memory_handle));

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC cuda_ext_mipmapped_array_desc{};
        cuda_ext_mipmapped_array_desc.offset = 0;
        cuda_ext_mipmapped_array_desc.arrayDesc.Width = _size.x;
        cuda_ext_mipmapped_array_desc.arrayDesc.Height = _size.y;
        cuda_ext_mipmapped_array_desc.arrayDesc.Depth = 0;
        cuda_ext_mipmapped_array_desc.arrayDesc.Format = _base.is_hdr() ?
                                                             CU_AD_FORMAT_HALF :
                                                             CU_AD_FORMAT_UNSIGNED_INT8;
        cuda_ext_mipmapped_array_desc.arrayDesc.NumChannels = 4;
        cuda_ext_mipmapped_array_desc.numLevels = 1;
        LUISA_CHECK_CUDA(cuExternalMemoryGetMappedMipmappedArray(
            &_cuda_ext_image_mipmapped_array, _cuda_ext_image_memory,
            &cuda_ext_mipmapped_array_desc));
        LUISA_CHECK_CUDA(cuMipmappedArrayGetLevel(
            &_cuda_ext_image_array, _cuda_ext_image_mipmapped_array, 0));
    }

    void _cuda_import_semaphore(VkSemaphore vk_semaphore,
                                CUexternalSemaphore &ext_semaphore) noexcept {

        auto vulkan_semaphore_handle = [this, vk_semaphore](auto type) noexcept {
            auto device = _base.device();
#ifdef LUISA_PLATFORM_WINDOWS
            auto fp_vkGetSemaphoreWin32HandleKHR = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
                vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
            LUISA_ASSERT(fp_vkGetSemaphoreWin32HandleKHR != nullptr,
                         "Failed to load vkGetSemaphoreWin32HandleKHR function.");
            HANDLE handle{};
            VkSemaphoreGetWin32HandleInfoKHR handle_info{};
            handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
            handle_info.pNext = nullptr;
            handle_info.semaphore = vk_semaphore;
            handle_info.handleType = type;
            LUISA_CHECK_VULKAN(fp_vkGetSemaphoreWin32HandleKHR(device, &handle_info, &handle));
            return handle;
#else
            auto fp_vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
                vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
            LUISA_ASSERT(fp_vkGetSemaphoreFdKHR != nullptr,
                         "Failed to load vkGetSemaphoreFdKHR function.");
            auto fd = 0;
            VkSemaphoreGetFdInfoKHR fd_info{};
            fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
            fd_info.pNext = nullptr;
            fd_info.semaphore = vk_semaphore;
            fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
            LUISA_CHECK_VULKAN(fp_vkGetSemaphoreFdKHR(device, &fd_info, &fd));
            return fd;
#endif
        };

        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC cuda_ext_semaphore_handle_desc{};
#ifdef LUISA_PLATFORM_WINDOWS
        cuda_ext_semaphore_handle_desc.type =
            IsWindows8OrGreater() ?
                CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 :
                CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT;
        cuda_ext_semaphore_handle_desc.handle.win32.handle = vulkan_semaphore_handle(
            IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                                    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        cuda_ext_semaphore_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
        cuda_ext_semaphore_handle_desc.handle.fd = vulkan_semaphore_handle(
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
#endif

        LUISA_CHECK_CUDA(cuImportExternalSemaphore(&ext_semaphore, &cuda_ext_semaphore_handle_desc));
        LUISA_ASSERT(ext_semaphore != nullptr, "Failed to import external semaphore.");
    }

private:
    void _initialize() noexcept {
        // vulkan objects
        _create_image();
        _initialize_image_ownership();
        _acquire_image_command = _create_image_handoff_command(true);
        _release_image_command = _create_image_handoff_command(false);
        _create_image_view();
        _create_semaphores();
        // cuda objects
        _cuda_import_image();
        auto n = _base.back_buffer_count();
        _cuda_ext_semaphores.resize(n);
        for (auto i = 0u; i < n; i++) {
            _cuda_import_semaphore(_semaphores[i], _cuda_ext_semaphores[i]);
        }
        _cuda_import_semaphore(_released_semaphore, _cuda_released_semaphore);
    }

    void _cleanup() noexcept {
        auto device = _base.device();
        // Finish both APIs' accesses before releasing their shared mappings and
        // synchronization objects. The external memory must outlive its mapping.
        LUISA_CHECK_CUDA(cuCtxSynchronize());
        LUISA_CHECK_VULKAN(vkDeviceWaitIdle(device));
        // cuda objects
        LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(_cuda_ext_image_mipmapped_array));
        LUISA_CHECK_CUDA(cuDestroyExternalMemory(_cuda_ext_image_memory));
        for (auto semaphore : _cuda_ext_semaphores) {
            LUISA_CHECK_CUDA(cuDestroyExternalSemaphore(semaphore));
        }
        LUISA_CHECK_CUDA(cuDestroyExternalSemaphore(_cuda_released_semaphore));
        // vulkan objects
        vkFreeCommandBuffers(device, _base.command_pool(), 1u, &_acquire_image_command);
        vkFreeCommandBuffers(device, _base.command_pool(), 1u, &_release_image_command);
        vkDestroyImageView(device, _image_view, nullptr);
        vkDestroyImage(device, _image, nullptr);
        vkFreeMemory(device, _image_memory, nullptr);
        for (auto semaphore : _semaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        vkDestroySemaphore(device, _released_semaphore, nullptr);
    }

public:
    Impl(CUuuid device_uuid,
         uint64_t display_handle, uint64_t window_handle,
         uint width, uint height, bool allow_hdr,
         bool vsync, uint back_buffer_size,
         bool transparent) noexcept
        : _base{luisa::bit_cast<VulkanDeviceUUID>(device_uuid),
                display_handle, window_handle, width, height,
                allow_hdr, vsync, back_buffer_size, required_extensions, transparent},
          _size{make_uint2(width, height)} { _initialize(); }
    ~Impl() noexcept { _cleanup(); }
    [[nodiscard]] auto native_handle() noexcept { return &_base; }
    [[nodiscard]] auto native_handle() const noexcept { return &_base; }
    [[nodiscard]] auto pixel_storage() const noexcept {
        return _base.is_hdr() ? PixelStorage::HALF4 : PixelStorage::BYTE4;
    }
    [[nodiscard]] auto size() const noexcept { return _size; }

    void present(CUstream stream, CUarray image) noexcept {
        auto name = [this] {
            std::scoped_lock lock{_name_mutex};
            return _name;
        }();

        std::scoped_lock lock{_present_mutex};
        LUISA_ASSERT(_current_frame < _semaphores.size(), "Invalid frame index.");

        if (!name.empty()) { nvtxRangePushA(luisa::format("{}::present", name).c_str()); }

        // wait for the frame to be ready
        _base.wait_for_fence();
        // All frames share one imported image. Waiting for this frame's Vulkan
        // fence alone does not protect it from the previous frame's sampling.
        if (_has_presented_frame) {
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait_params{};
            LUISA_CHECK_CUDA(cuWaitExternalSemaphoresAsync(
                &_cuda_released_semaphore, &wait_params, 1u, stream));
        }

        // copy image to swapchain image
        if (!name.empty()) { nvtxRangePushA("copy"); }
        CUDA_MEMCPY3D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.srcArray = image;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = _cuda_ext_image_array;
        copy.WidthInBytes = pixel_storage_size(pixel_storage(), make_uint3(_size.x, 1u, 1u));
        copy.Height = pixel_storage_size(pixel_storage(), make_uint3(_size.xy(), 1u)) / copy.WidthInBytes;
        copy.Depth = 1u;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, stream));
        if (!name.empty()) { nvtxRangePop(); }

        // signal that the frame is ready
        if (!name.empty()) { nvtxRangePushA(luisa::format("signal", name).c_str()); }
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal_params{};
        LUISA_ASSERT(_current_frame < _cuda_ext_semaphores.size(), "Invalid frame index.");
        auto current_semaphore = _cuda_ext_semaphores[_current_frame];
        LUISA_CHECK_CUDA(cuSignalExternalSemaphoresAsync(&current_semaphore, &signal_params, 1, stream));
        if (!name.empty()) { nvtxRangePop(); }

        // present
        if (!name.empty()) { nvtxRangePushA(luisa::format("present", name).c_str()); }
        // Consume CUDA's signal independently of present: swapchain recreation
        // can return before the base submits a draw. The two queue submissions
        // still pair every ready signal and ownership handoff on that path.
        VkPipelineStageFlags ready_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo ready_submit{};
        ready_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        ready_submit.waitSemaphoreCount = 1u;
        ready_submit.pWaitSemaphores = &_semaphores[_current_frame];
        ready_submit.pWaitDstStageMask = &ready_stage;
        ready_submit.commandBufferCount = 1u;
        ready_submit.pCommandBuffers = &_acquire_image_command;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_base.queue(), 1u, &ready_submit, nullptr));
        _base.present(nullptr, nullptr, _image_view,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkSubmitInfo released_submit{};
        released_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        released_submit.commandBufferCount = 1u;
        released_submit.pCommandBuffers = &_release_image_command;
        released_submit.signalSemaphoreCount = 1u;
        released_submit.pSignalSemaphores = &_released_semaphore;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_base.queue(), 1u, &released_submit, nullptr));
        _has_presented_frame = true;
        if (!name.empty()) { nvtxRangePop(); }

        // These semaphores belong to the shared import, not to base swapchain
        // images, whose count may change during recreation.
        _current_frame = (_current_frame + 1u) % _semaphores.size();

        if (!name.empty()) { nvtxRangePop(); }
    }

    void set_name(luisa::string &&name) noexcept {
        std::scoped_lock lock{_name_mutex};
        _name = std::move(name);
    }
};

CUDASwapchain::CUDASwapchain(CUDADevice *device, SwapchainOption o) noexcept
    : _impl{luisa::make_unique<Impl>(device->handle().handle_uuid(),
                                     o.display, o.window, o.size.x, o.size.y,
                                     o.wants_hdr, o.wants_vsync, o.back_buffer_count, o.wants_transparent)} {}

CUDASwapchain::~CUDASwapchain() noexcept = default;

PixelStorage CUDASwapchain::pixel_storage() const noexcept {
    return _impl->pixel_storage();
}

VulkanSwapchain *CUDASwapchain::native_handle() noexcept {
    return _impl->native_handle();
}

void CUDASwapchain::present(CUDAStream *stream, CUDATexture *image) noexcept {
    LUISA_ASSERT(image->storage() == _impl->pixel_storage(),
                 "Image pixel format must match the swapchain.");
    LUISA_ASSERT(all(image->size() == make_uint3(_impl->size(), 1u)),
                 "Image size and pixel format must match the swapchain.");
    _impl->present(stream->handle(), image->level(0u));
}

void CUDASwapchain::set_name(luisa::string &&name) noexcept {
    _impl->set_name(std::move(name));
}

}// namespace luisa::compute::cuda

#endif
