#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <vulkan/vulkan.h>

#include <luisa/core/clock.h>
#include <luisa/core/platform.h>

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <vulkan/vulkan_win32.h>
#endif

#include "cuda_error.h"
#include "cuda_event.h"

namespace luisa::compute::cuda {

CUDAEvent::CUDAEvent(VkDevice device,
                     VkSemaphore vk_semaphore,
                     CUexternalSemaphore cuda_semaphore) noexcept
    : _device{device},
      _vk_semaphore{vk_semaphore},
      _cuda_semaphore{cuda_semaphore} {}

void CUDAEvent::signal(CUstream stream, uint64_t value) noexcept {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    params.params.fence.value = value;
    LUISA_CHECK_CUDA(cuSignalExternalSemaphoresAsync(&_cuda_semaphore, &params, 1, stream));
}

void CUDAEvent::wait(CUstream stream, uint64_t value) noexcept {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    params.params.fence.value = value;
    LUISA_CHECK_CUDA(cuWaitExternalSemaphoresAsync(&_cuda_semaphore, &params, 1, stream));
}

void CUDAEvent::notify(uint64_t value) noexcept {
    VkSemaphoreSignalInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.pNext = nullptr;
    info.semaphore = _vk_semaphore;
    info.value = value;
    LUISA_CHECK_VULKAN(vkSignalSemaphore(_device, &info));
}

void CUDAEvent::synchronize(uint64_t value) noexcept {
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.semaphoreCount = 1;
    info.pSemaphores = &_vk_semaphore;
    info.pValues = &value;
    constexpr auto uint64_max = std::numeric_limits<uint64_t>::max();
    LUISA_CHECK_VULKAN(vkWaitSemaphores(_device, &info, uint64_max));
}

uint64_t CUDAEvent::signaled_value() noexcept {
    auto signaled_value = static_cast<uint64_t>(0u);
    LUISA_CHECK_VULKAN(vkGetSemaphoreCounterValue(
        _device, _vk_semaphore, &signaled_value));
    return signaled_value;
}

bool CUDAEvent::is_completed(uint64_t value) noexcept {
    return signaled_value() >= value;
}

CUDAEventManager::CUDAEventManager(const CUuuid &uuid) noexcept
    : _instance{VulkanInstance::retain()} {

    auto check_uuid = [uuid = luisa::bit_cast<VulkanDeviceUUID>(uuid)](auto device) noexcept {
        VkPhysicalDeviceIDProperties id_properties{};
        id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &id_properties;
        vkGetPhysicalDeviceProperties2(device, &properties2);
        if (properties2.properties.apiVersion < LUISA_REQUIRED_VULKAN_VERSION) { return false; }
        return std::memcmp(id_properties.deviceUUID, uuid.bytes, sizeof(uuid.bytes)) == 0;
    };

    auto find_queue_family = [](auto device) noexcept -> uint32_t {
        auto queue_family_count = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        luisa::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
        for (auto i = 0u; i < queue_family_count; i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                return i;
            }
        }
        return 0u;
    };

    // find the physical device
    auto device_count = 0u;
    LUISA_CHECK_VULKAN(vkEnumeratePhysicalDevices(
        _instance->handle(), &device_count, nullptr));
    LUISA_ASSERT(device_count > 0u, "Failed to find GPUs with Vulkan support.");
    luisa::vector<VkPhysicalDevice> devices(device_count);
    LUISA_CHECK_VULKAN(vkEnumeratePhysicalDevices(
        _instance->handle(), &device_count, devices.data()));
    for (auto device : devices) {
        if (check_uuid(device)) {
            _physical_device = device;
            break;
        }
    }
    LUISA_ASSERT(_physical_device != nullptr,
                 "Failed to find a GPU with matching UUID.");

    // create the logical device
    VkDeviceQueueCreateInfo queue_create_info{};
    auto queue_priority = 1.f;
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = find_queue_family(_physical_device);
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceVulkan12Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features.timelineSemaphore = true;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &features;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    static constexpr std::array required_extensions{
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
    };
    device_create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
    device_create_info.ppEnabledExtensionNames = required_extensions.data();

    constexpr std::array validation_layers{"VK_LAYER_KHRONOS_validation"};
    if (_instance->has_debug_layer()) {
        device_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        device_create_info.ppEnabledLayerNames = validation_layers.data();
    }
    LUISA_CHECK_VULKAN(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));
    LUISA_ASSERT(_device != nullptr, "Failed to create Vulkan device.");

#ifdef LUISA_PLATFORM_WINDOWS
    _addr_vkGetSemaphoreHandle = reinterpret_cast<uint64_t>(
        vkGetDeviceProcAddr(_device, "vkGetSemaphoreWin32HandleKHR"));
    LUISA_ASSERT(_addr_vkGetSemaphoreHandle != 0u,
                 "Failed to load vkGetSemaphoreWin32HandleKHR function.");
#else
    _addr_vkGetSemaphoreHandle = reinterpret_cast<uint64_t>(
        vkGetDeviceProcAddr(_device, "vkGetSemaphoreFdKHR"));
    LUISA_ASSERT(_addr_vkGetSemaphoreHandle != 0u,
                 "Failed to load vkGetSemaphoreFdKHR function.");
#endif
}

CUDAEventManager::~CUDAEventManager() noexcept {
    if (auto count = _count.load()) {
        LUISA_WARNING_WITH_LOCATION(
            "CUDAEventManager destroyed with {} events remaining.",
            count);
    }
    vkDestroyDevice(_device, nullptr);
}

CUDAEvent *CUDAEventManager::create() noexcept {

    Clock clock;

    // create vulkan semaphore
    VkSemaphoreTypeCreateInfo timeline_info;
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.pNext = nullptr;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    VkExportSemaphoreCreateInfoKHR export_info = {};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    export_info.pNext = &timeline_info;
#ifdef LUISA_PLATFORM_WINDOWS
    export_info.handleTypes = IsWindows8OrGreater() ?
                                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &export_info;
    info.flags = 0;

    VkSemaphore vk_semaphore{};
    LUISA_CHECK_VULKAN(vkCreateSemaphore(_device, &info, nullptr, &vk_semaphore));

    // import vulkan semaphore into cuda
    auto vulkan_semaphore_handle = [this, vk_semaphore](auto type) noexcept {
#ifdef LUISA_PLATFORM_WINDOWS
        auto fp_vkGetSemaphoreWin32HandleKHR =
            reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(_addr_vkGetSemaphoreHandle);
        HANDLE handle{};
        VkSemaphoreGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.pNext = nullptr;
        handle_info.semaphore = vk_semaphore;
        handle_info.handleType = type;
        LUISA_CHECK_VULKAN(fp_vkGetSemaphoreWin32HandleKHR(_device, &handle_info, &handle));
        return handle;
#else
        auto fp_vkGetSemaphoreFdKHR =
            reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(_addr_vkGetSemaphoreHandle);
        auto fd = 0;
        VkSemaphoreGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        fd_info.pNext = nullptr;
        fd_info.semaphore = vk_semaphore;
        fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        LUISA_CHECK_VULKAN(fp_vkGetSemaphoreFdKHR(_device, &fd_info, &fd));
        return fd;
#endif
    };

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC cuda_ext_semaphore_handle_desc{};
#ifdef _WIN64
    cuda_ext_semaphore_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32;
    cuda_ext_semaphore_handle_desc.handle.win32.handle = vulkan_semaphore_handle(
        IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
    cuda_ext_semaphore_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    cuda_ext_semaphore_handle_desc.handle.fd = vulkan_semaphore_handle(
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
#endif

    CUexternalSemaphore cuda_semaphore{};
    LUISA_CHECK_CUDA(cuImportExternalSemaphore(&cuda_semaphore, &cuda_ext_semaphore_handle_desc));

    _count++;
    auto event = luisa::new_with_allocator<CUDAEvent>(_device, vk_semaphore, cuda_semaphore);
    LUISA_VERBOSE("Created CUDA event in {} ms.", clock.toc());
    return event;
}

void CUDAEventManager::destroy(CUDAEvent *event) noexcept {
    _count--;
    LUISA_CHECK_CUDA(cuDestroyExternalSemaphore(event->_cuda_semaphore));
    vkDestroySemaphore(_device, event->_vk_semaphore, nullptr);
    luisa::delete_with_allocator(event);
}

}// namespace luisa::compute::cuda

#else

#endif
