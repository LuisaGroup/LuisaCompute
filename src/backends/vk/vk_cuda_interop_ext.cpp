#ifdef LCVK_ENABLE_CUDA
#include <cuda.h>
#include "vk_cuda_interop_ext.h"
#include "device.h"
#include "texture.h"
#include "stream.h"
#include "default_buffer.h"
#include "../cuda/cuda_stream.h"

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

#define LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
#include "../cuda/cuda_event.h"
#ifndef LUISA_CHECK_CUDA
#define LUISA_CHECK_CUDA(...)                            \
    do {                                                 \
        if (auto ec = __VA_ARGS__; ec != CUDA_SUCCESS) { \
            const char *err_name = nullptr;              \
            const char *err_string = nullptr;            \
            cuGetErrorName(ec, &err_name);               \
            cuGetErrorString(ec, &err_string);           \
            if (!err_string) { err_string = "unknown"; } \
            LUISA_ERROR_WITH_LOCATION(                   \
                "{}: {}", err_name, err_string);         \
        }                                                \
    } while (false)
#endif
namespace lc::vk {

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

struct CudaCtxGuard {
    CUcontext ctx;
    explicit CudaCtxGuard(CUcontext ctx) noexcept : ctx{ctx} {
        LUISA_CHECK_CUDA(cuCtxPushCurrent(ctx));
    }
    ~CudaCtxGuard() noexcept {
        CUcontext ctx{nullptr};
        LUISA_CHECK_CUDA(cuCtxPopCurrent(&ctx));
        LUISA_ASSERT(ctx == this->ctx,
                     "Mismatched cuda context in CudaCtxGuard.");
    }
};

template<typename F>
decltype(auto) with_cuda(CUcontext ctx, F &&f) {
    CudaCtxGuard _{ctx};
    return std::invoke(std::forward<F>(f));
}

static void initialize_cuda() noexcept {
    static std::once_flag flag;
    std::call_once(flag, [] {
        LUISA_CHECK_CUDA(cuInit(0));
    });
}

[[nodiscard]] int getCudaDeviceForVulkanDevice(VkPhysicalDevice device) noexcept {
    initialize_cuda();
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(device, &properties2);
    int cudaDeviceCount = 0;
    LUISA_CHECK_CUDA(cuDeviceGetCount(&cudaDeviceCount));
    for (auto i = 0; i < cudaDeviceCount; i++) {
        char cudaLuid[sizeof(id_properties.deviceLUID)] = {};
        unsigned int cudaNodeMask = 0;
        LUISA_CHECK_CUDA(cuDeviceGetLuid(cudaLuid, &cudaNodeMask, i));
        if (!std::memcmp(&id_properties.deviceLUID, cudaLuid, sizeof(cudaLuid))) {
            LUISA_VERBOSE_WITH_LOCATION("Found cuda device at {} for vulkan device.", i);
            return i;
        }
    }
    LUISA_ERROR("Failed to get cuda device for d3d12 device.");
}

[[nodiscard]] auto _find_memory_type(uint32_t type_filter, VkPhysicalDevice physical_device, VkMemoryPropertyFlags properties) noexcept {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    LUISA_ERROR_WITH_LOCATION("Failed to find suitable memory type.");
    vstd::unreachable();
}
VkCudaInteropImpl::VkCudaInteropImpl(Device *device) : _device(device) {
    auto cuda_device = getCudaDeviceForVulkanDevice(device->physical_device());
    LUISA_CHECK_CUDA(cuDeviceGet(&_cu_device, cuda_device));
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_cu_context, _cu_device));
}
VkCudaInteropImpl::~VkCudaInteropImpl() {}
auto vulkan_device_memory_handle(VkDevice device, VkDeviceMemory memory, auto type) {
#ifdef LUISA_PLATFORM_WINDOWS
    auto fp_vkGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    LUISA_ASSERT(fp_vkGetMemoryWin32HandleKHR != nullptr,
                 "Failed to load vkGetMemoryWin32HandleKHR function.");
    HANDLE handle{};
    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.pNext = nullptr;
    handle_info.memory = memory;
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
    fd_info.memory = memory;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    LUISA_CHECK_VULKAN(fp_vkGetMemoryFdKHR(device, &fd_info, &fd));
    return fd;
#endif
}
BufferCreationInfo VkCudaInteropImpl::create_interop_buffer(const Type *element, size_t elem_count) noexcept {
    VkBuffer buffer;
    VkDeviceMemory buffer_memory;
    VkExternalMemoryBufferCreateInfo external_memory_info{};
    external_memory_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
#ifdef LUISA_PLATFORM_WINDOWS
    external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    VkImage image;
    VkDeviceMemory image_memory;
    size_t element_stride = (element == Type::of<void>() ? 1 : element->size());
    size_t size_bytes = element_stride * elem_count;
    VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_memory_info,
        .size = size_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr};
    LUISA_CHECK_VULKAN(vkCreateBuffer(
        _device->logic_device(),
        &buffer_info,
        Device::alloc_callbacks(),
        &buffer));
    // compute memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(_device->logic_device(), buffer, &mem_requirements);
    auto buffer_memory_size = mem_requirements.size;

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

    VkMemoryAllocateFlagsInfo alloc_flag_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = &export_allocate_info,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = _find_memory_type(mem_requirements.memoryTypeBits, _device->physical_device(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    alloc_info.pNext = &alloc_flag_info;
    LUISA_CHECK_VULKAN(vkAllocateMemory(_device->logic_device(), &alloc_info, Device::alloc_callbacks(), &buffer_memory));
    LUISA_CHECK_VULKAN(vkBindBufferMemory(_device->logic_device(), buffer, buffer_memory, 0));

    BufferCreationInfo info;
    auto lc_buffer = new DefaultBuffer(
        _device,
        buffer,
        buffer_memory,
        size_bytes);
    info.handle = reinterpret_cast<uint64_t>(lc_buffer);
    info.native_handle = lc_buffer->vk_buffer();
    info.element_stride = element_stride;
    info.total_size_bytes = size_bytes;
    return info;
}
ResourceCreationInfo VkCudaInteropImpl::create_interop_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept {
    VkExternalMemoryImageCreateInfo external_memory_info{};
    external_memory_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef LUISA_PLATFORM_WINDOWS
    external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    external_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    VkImage image;
    VkDeviceMemory image_memory;

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = Texture::to_vk_format(format);
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.pNext = &external_memory_info;
    LUISA_CHECK_VULKAN(vkCreateImage(_device->logic_device(), &image_info, Device::alloc_callbacks(), &image));

    // compute memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(_device->logic_device(), image, &mem_requirements);
    auto image_memory_size = mem_requirements.size;

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

    VkMemoryAllocateFlagsInfo alloc_flag_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = &export_allocate_info,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = _find_memory_type(mem_requirements.memoryTypeBits, _device->physical_device(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    alloc_info.pNext = &alloc_flag_info;
    LUISA_CHECK_VULKAN(vkAllocateMemory(_device->logic_device(), &alloc_info, Device::alloc_callbacks(), &image_memory));
    LUISA_CHECK_VULKAN(vkBindImageMemory(_device->logic_device(), image, image_memory, 0));
    auto tex = new Texture(
        _device,
        image,
        dimension,
        Texture::to_vk_format(format),
        uint3(width, height, 1),
        mipmap_levels,
        true,
        image_memory);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(tex),
        .native_handle = tex->vk_image()};
}
void VkCudaInteropImpl::_vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept {
    auto evt = reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle);
    auto stream = reinterpret_cast<lc::vk::Stream *>(vk_stream);
    auto semaphore = evt->vk_semaphore();
    if (_device->config_ext() && _device->config_ext()->signal_semaphore(stream->queue(), semaphore, fence_index))
        return;
    VkTimelineSemaphoreSubmitInfo timelineInfo1{};
    timelineInfo1.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo1.pNext = nullptr;
    timelineInfo1.waitSemaphoreValueCount = 0;
    timelineInfo1.pWaitSemaphoreValues = nullptr;
    timelineInfo1.signalSemaphoreValueCount = 1;
    timelineInfo1.pSignalSemaphoreValues = &fence_index;
    VkSubmitInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info1.pNext = &timelineInfo1;
    info1.waitSemaphoreCount = 0;
    info1.pWaitSemaphores = nullptr;
    info1.signalSemaphoreCount = 1;
    info1.pSignalSemaphores = &semaphore;
    // ... Enqueue initial device work here.
    info1.commandBufferCount = 0;
    info1.pCommandBuffers = nullptr;
    stream->queue_mtx().lock();
    LUISA_CHECK_VULKAN(vkQueueSubmit(stream->queue(), 1, &info1, VK_NULL_HANDLE));
    stream->queue_mtx().unlock();
}
void VkCudaInteropImpl::_vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept {
    auto evt = reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle);
    auto stream = reinterpret_cast<lc::vk::Stream *>(vk_stream);
    auto semaphore = evt->vk_semaphore();
    if (_device->config_ext() && _device->config_ext()->wait_semaphore(stream->queue(), semaphore, fence_index))
        return;
    VkTimelineSemaphoreSubmitInfo timelineInfo1{};
    timelineInfo1.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo1.pNext = nullptr;
    timelineInfo1.waitSemaphoreValueCount = 0;
    timelineInfo1.pWaitSemaphoreValues = nullptr;
    timelineInfo1.signalSemaphoreValueCount = 1;
    timelineInfo1.pSignalSemaphoreValues = &fence_index;
    VkSubmitInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info1.pNext = &timelineInfo1;
    info1.waitSemaphoreCount = 1;
    info1.pWaitSemaphores = &semaphore;
    info1.signalSemaphoreCount = 0;
    info1.pSignalSemaphores = nullptr;
    // ... Enqueue initial device work here.
    info1.commandBufferCount = 0;
    info1.pCommandBuffers = nullptr;
    stream->queue_mtx().lock();
    LUISA_CHECK_VULKAN(vkQueueSubmit(stream->queue(), 1, &info1, VK_NULL_HANDLE));
    stream->queue_mtx().unlock();
}

void VkCudaInteropImpl::cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept {
    with_cuda(_cu_context, [&] {
        auto vk_buffer = reinterpret_cast<DefaultBuffer const *>(vk_buffer_handle);
        LUISA_ASSERT(vk_buffer->is_external_allocation());
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC cuda_ext_memory_handle{};
#ifdef LUISA_PLATFORM_WINDOWS
        cuda_ext_memory_handle.type = IsWindows8OrGreater() ?
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 :
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT;
        cuda_ext_memory_handle.handle.win32.handle = vulkan_device_memory_handle(
            _device->logic_device(),
            vk_buffer->external_device_memory(),
            IsWindows8OrGreater() ?
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        cuda_ext_memory_handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        cuda_ext_memory_handle.handle.fd = vulkan_device_memory_handle(
            _device->logic_device(),
            vk_buffer->external_device_memory(),
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(_device->logic_device(), vk_buffer->vk_buffer(), &mem_requirements);
        cuda_ext_memory_handle.size = mem_requirements.size;
        cuda_ext_memory_handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CUexternalMemory external_memory{};
        LUISA_CHECK_CUDA(cuImportExternalMemory(&external_memory, &cuda_ext_memory_handle));
        *cuda_handle = reinterpret_cast<uint64_t>(external_memory);
        // TODO: need cuda buffer here
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc{};
        bufferDesc.offset = 0;
        bufferDesc.size = vk_buffer->byte_size();

        bufferDesc.flags = 0;
        LUISA_CHECK_CUDA(cuExternalMemoryGetMappedBuffer(cuda_ptr, external_memory, &bufferDesc));
    });
}
uint64_t VkCudaInteropImpl::cuda_texture(uint64_t vk_texture_handle) noexcept {
    return with_cuda(_cu_context, [&] {
        auto vk_texture = reinterpret_cast<Texture const *>(vk_texture_handle);
        LUISA_ASSERT(vk_texture->is_external_allocation());
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC cuda_ext_memory_handle{};
#ifdef LUISA_PLATFORM_WINDOWS
        cuda_ext_memory_handle.type = IsWindows8OrGreater() ?
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 :
                                          CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT;
        cuda_ext_memory_handle.handle.win32.handle = vulkan_device_memory_handle(
            _device->logic_device(),
            vk_texture->external_device_memory(),
            IsWindows8OrGreater() ?
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT :
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        cuda_ext_memory_handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        cuda_ext_memory_handle.handle.fd = vulkan_device_memory_handle(
            _device->logic_device(),
            vk_texture->external_device_memory(),
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(_device->logic_device(), vk_texture->vk_image(), &mem_requirements);
        cuda_ext_memory_handle.size = mem_requirements.size;
        cuda_ext_memory_handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CUexternalMemory external_memory{};
        LUISA_CHECK_CUDA(cuImportExternalMemory(&external_memory, &cuda_ext_memory_handle));
        return reinterpret_cast<uint64_t>(external_memory);
    });
}
void VkCudaInteropImpl::unmap(void *cuda_ptr, void *cuda_handle) {
    with_cuda(_cu_context, [&] {
        LUISA_CHECK_CUDA(cuMemFree(reinterpret_cast<CUdeviceptr>(cuda_ptr)));
        LUISA_CHECK_CUDA(cuDestroyExternalMemory(reinterpret_cast<CUexternalMemory>(cuda_handle)));
    });
}
DeviceInterface *VkCudaInteropImpl::device() {
    return _device;
}
CudaDeviceConfigExt::ExternalVkDevice VkCudaInteropImpl::get_external_vk_device() const noexcept {
    return {_device->physical_device(), _device->logic_device()};
}
}// namespace lc::vk
#endif