#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <vulkan/vulkan.h>

#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/platform.h>

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <AclAPI.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#else
#error "Unsupported platform"
#endif

#include "../common/vulkan_instance.h"
#include "../common/vulkan_swapchain.h"
#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_texture.h"
#include "cuda_swapchain.h"

namespace luisa::compute::cuda {

#ifdef LUISA_PLATFORM_WINDOWS

class WindowsSecurityAttributes {

protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes{};
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor{};

public:
    WindowsSecurityAttributes() noexcept {
        m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
            1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);
        SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
        AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID,
                                 0, 0, 0, 0, 0, 0, 0, ppSID);
        EXPLICIT_ACCESS explicitAccess;
        ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
        explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
        explicitAccess.grfAccessMode = SET_ACCESS;
        explicitAccess.grfInheritance = INHERIT_ONLY;
        explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;
        SetEntriesInAcl(1, &explicitAccess, nullptr, ppACL);
        SetSecurityDescriptorDacl(m_winPSecurityDescriptor, true, *ppACL, false);
        m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
        m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
        m_winSecurityAttributes.bInheritHandle = true;
    }
    ~WindowsSecurityAttributes() noexcept {
        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        if (*ppSID) { FreeSid(*ppSID); }
        if (*ppACL) { LocalFree(*ppACL); }
        free(m_winPSecurityDescriptor);
    }
    [[nodiscard]] auto get() const noexcept {
        return &m_winSecurityAttributes;
    }
};

#endif

class CUDASwapchain::Impl {

private:
    static constexpr std::array required_extensions{
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
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
            barrier.srcAccessMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
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
    }

private:
    // cuda objects
    CUexternalMemory _cuda_ext_image_memory{};
    CUmipmappedArray _cuda_ext_image_mipmapped_array{};
    CUarray _cuda_ext_image_array{};
    luisa::vector<CUexternalSemaphore> _cuda_ext_semaphores;

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
#ifdef _WIN64
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
#ifdef _WIN64
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

        LUISA_CHECK_CUDA(cuImportExternalSemaphore(
            &ext_semaphore, &cuda_ext_semaphore_handle_desc));
    }

private:
    void _initialize() noexcept {
        // vulkan objects
        _create_image();
        _transition_image_layout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        _transition_image_layout(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        _create_image_view();
        _create_semaphores();
        // cuda objects
        _cuda_import_image();
        auto n = _base.back_buffer_count();
        _cuda_ext_semaphores.resize(n);
        for (auto i = 0u; i < n; i++) {
            _cuda_import_semaphore(_semaphores[i], _cuda_ext_semaphores[i]);
        }
    }

    void _cleanup() noexcept {
        auto device = _base.device();
        auto n = _base.back_buffer_count();
        // cuda objects
        LUISA_CHECK_CUDA(cuCtxSynchronize());
        LUISA_CHECK_CUDA(cuDestroyExternalMemory(_cuda_ext_image_memory));
        LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(_cuda_ext_image_mipmapped_array));
        for (auto i = 0u; i < n; i++) {
            LUISA_CHECK_CUDA(cuDestroyExternalSemaphore(_cuda_ext_semaphores[i]));
        }
        // vulkan objects
        LUISA_CHECK_VULKAN(vkDeviceWaitIdle(device));
        vkDestroyImageView(device, _image_view, nullptr);
        vkDestroyImage(device, _image, nullptr);
        vkFreeMemory(device, _image_memory, nullptr);
        for (auto i = 0u; i < n; i++) {
            vkDestroySemaphore(device, _semaphores[i], nullptr);
        }
    }

public:
    Impl(CUuuid device_uuid, uint64_t window_handle,
         uint width, uint height, bool allow_hdr,
         bool vsync, uint back_buffer_size) noexcept
        : _base{luisa::bit_cast<VulkanDeviceUUID>(device_uuid),
                window_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
                required_extensions},
          _size{make_uint2(width, height)} { _initialize(); }
    ~Impl() noexcept { _cleanup(); }
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

        if (!name.empty()) { nvtxRangePushA(luisa::format("{}::present", name).c_str()); }

        // wait for the frame to be ready
        _base.wait_for_fence();

        // copy image to swapchain image
        if (!name.empty()) { nvtxRangePushA("copy"); }
        CUDA_MEMCPY3D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.srcArray = image;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = _cuda_ext_image_array;
        copy.WidthInBytes = pixel_storage_size(pixel_storage(), make_uint3(_size.x, 1u, 1u));
        copy.Height = _size.y;
        copy.Depth = 1u;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, stream));
        if (!name.empty()) { nvtxRangePop(); }

        // signal that the frame is ready
        if (!name.empty()) { nvtxRangePushA(luisa::format("signal", name).c_str()); }
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal_params{};
        LUISA_CHECK_CUDA(cuSignalExternalSemaphoresAsync(
            &_cuda_ext_semaphores[_current_frame], &signal_params, 1, stream));
        if (!name.empty()) { nvtxRangePop(); }

        // present
        if (!name.empty()) { nvtxRangePushA(luisa::format("present", name).c_str()); }
        _base.present(_semaphores[_current_frame], nullptr, _image_view,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (!name.empty()) { nvtxRangePop(); }

        // update current frame index
        _current_frame = (_current_frame + 1u) % _base.back_buffer_count();

        if (!name.empty()) { nvtxRangePop(); }
    }

    void set_name(luisa::string &&name) noexcept {
        std::scoped_lock lock{_name_mutex};
        _name = std::move(name);
    }
};

CUDASwapchain::CUDASwapchain(CUDADevice *device, uint64_t window_handle,
                             uint width, uint height, bool allow_hdr,
                             bool vsync, uint back_buffer_size) noexcept
    : _impl{luisa::make_unique<Impl>(device->handle().uuid(),
                                     window_handle, width, height,
                                     allow_hdr, vsync, back_buffer_size)} {}

CUDASwapchain::~CUDASwapchain() noexcept = default;

PixelStorage CUDASwapchain::pixel_storage() const noexcept {
    return _impl->pixel_storage();
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
