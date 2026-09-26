#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
#include <cuda.h>
#include "vk_cuda_interop_ext.h"
#include "device.h"
#include "texture.h"
#include "cuda_interop_texture_plan.h"
#include "stream.h"
#include "default_buffer.h"
// Cross-DLL access to cuda-backend shaders is restricted to pure-virtual and
// header-inline accessors on the CUDAShader base (no non-inline symbols, no
// dynamic_cast, no delete) — see cuda_shader.h.
#include "../cuda/cuda_shader.h"
#include "../cuda/cuda_stream.h"

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <winternl.h>// OBJECT_ATTRIBUTES for CUDA's win32HandleMetaData
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <AclAPI.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#include <unistd.h>// ::close on imported CUDA file descriptors
#else
#error "Unsupported platform"
#endif

#define LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
#include "../cuda/cuda_event.h"
#include <luisa/core/stl/functional.h>
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
    SECURITY_ATTRIBUTES _win_security_attributes{};
    PSECURITY_DESCRIPTOR _win_p_security_descriptor{};

public:
    WindowsSecurityAttributes() noexcept {
        _win_p_security_descriptor = (PSECURITY_DESCRIPTOR)calloc(
            1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
        PSID *ppSID = (PSID *)((PBYTE)_win_p_security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        InitializeSecurityDescriptor(_win_p_security_descriptor, SECURITY_DESCRIPTOR_REVISION);
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
        SetSecurityDescriptorDacl(_win_p_security_descriptor, true, *ppACL, false);
        _win_security_attributes.nLength = sizeof(_win_security_attributes);
        _win_security_attributes.lpSecurityDescriptor = _win_p_security_descriptor;
        _win_security_attributes.bInheritHandle = true;
    }
    ~WindowsSecurityAttributes() noexcept {
        PSID *ppSID = (PSID *)((PBYTE)_win_p_security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        if (*ppSID) { FreeSid(*ppSID); }
        if (*ppACL) { LocalFree(*ppACL); }
        free(_win_p_security_descriptor);
    }
    [[nodiscard]] auto get() const noexcept {
        return &_win_security_attributes;
    }
    // The descriptor itself, needed by CUDA's VMM API which takes an
    // OBJECT_ATTRIBUTES (win32HandleMetaData) rather than a
    // SECURITY_ATTRIBUTES.
    [[nodiscard]] auto descriptor() const noexcept {
        return _win_p_security_descriptor;
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
    return luisa::invoke(std::forward<F>(f));
}

static bool initialize_cuda() noexcept {
    static std::once_flag flag;
    static bool success{};
    std::call_once(flag, [] {
        success = cuInit(0) == CUDA_SUCCESS;
    });
    return success;
}

[[nodiscard]] int get_cuda_device_for_vulkan_device(VkPhysicalDevice device) noexcept {
    if (!initialize_cuda()) return -1;
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(device, &properties2);
    int cudaDeviceCount = 0;
    if (cuDeviceGetCount(&cudaDeviceCount) != CUDA_SUCCESS) {
        return -1;
    }
    for (auto i = 0; i < cudaDeviceCount; i++) {
        char cudaLuid[sizeof(id_properties.deviceLUID)] = {};
        unsigned int cudaNodeMask = 0;
        if (cuDeviceGetLuid(cudaLuid, &cudaNodeMask, i) != CUDA_SUCCESS) continue;
        if (!std::memcmp(&id_properties.deviceLUID, cudaLuid, sizeof(cudaLuid))) {
            LUISA_VERBOSE_WITH_LOCATION("Found cuda device at {} for vulkan device.", i);
            return i;
        }
    }
    return -1;
}

[[nodiscard]] auto find_memory_type(uint32_t type_filter, VkPhysicalDevice physical_device, VkMemoryPropertyFlags properties) noexcept {
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
VkCudaInteropImpl::VkCudaInteropImpl(Device *device) noexcept : _device(device) {
    // VK_NV_cuda_kernel_launch entry points are loaded manually because volk
    // is compiled without VK_ENABLE_BETA_EXTENSIONS. This path is pure Vulkan
    // and does not require a CUDA driver context.
    if (_device->enable_cuda_kernel_launch()) {
        auto logic_device = _device->logic_device();
        auto load = [logic_device](auto &pfn, const char *name) noexcept {
            using PFN = std::remove_reference_t<decltype(pfn)>;
            pfn = reinterpret_cast<PFN>(vkGetDeviceProcAddr(logic_device, name));
        };
          load(_cuda_launch_funcs.create_cuda_module, "vkCreateCudaModuleNV");
          load(_cuda_launch_funcs.create_cuda_function, "vkCreateCudaFunctionNV");
        load(_cuda_launch_funcs.destroy_cuda_module, "vkDestroyCudaModuleNV");
        load(_cuda_launch_funcs.destroy_cuda_function, "vkDestroyCudaFunctionNV");
        load(_cuda_launch_funcs.cmd_cuda_launch_kernel, "vkCmdCudaLaunchKernelNV");
        if (!_cuda_launch_funcs.valid()) {
            LUISA_WARNING(
                "VK_NV_cuda_kernel_launch entry points could not be loaded; "
                "CUDA kernel launch is disabled.");
        }
    }
    _cuda_device = get_cuda_device_for_vulkan_device(device->physical_device());
    if (_cuda_device == -1) return;
    LUISA_CHECK_CUDA(cuDeviceGet(&_cu_device, _cuda_device));
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_cu_context, _cu_device));
}
VkCudaInteropImpl::~VkCudaInteropImpl() noexcept {
    if (_cu_device)
        LUISA_CHECK_CUDA(cuDevicePrimaryCtxRelease(_cu_device));
}
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
    size_t element_stride = (element == Type::of<void>() ? 1 : element->size());
    size_t size_bytes = element_stride * elem_count;
    VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_memory_info,
        .size = size_bytes,
        .usage = (VkBufferUsageFlags)(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                      VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT |
                                      (_device->enable_device_address() ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0) |
                                      (_device->enable_raytracing() ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0)),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr};
    _device->allocator().apply_queue_sharing(buffer_info);
    LUISA_CHECK_VULKAN(vkCreateBuffer(
        _device->logic_device(),
        &buffer_info,
        Device::alloc_callbacks(),
        &buffer));
    // compute memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(_device->logic_device(), buffer, &mem_requirements);
    [[maybe_unused]] auto buffer_memory_size = mem_requirements.size;

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
        .flags = (VkMemoryAllocateFlags)(_device->enable_device_address() ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0)};

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, _device->physical_device(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    alloc_info.pNext = &alloc_flag_info;
    LUISA_CHECK_VULKAN(vkAllocateMemory(_device->logic_device(), &alloc_info, Device::alloc_callbacks(), &buffer_memory));
    LUISA_CHECK_VULKAN(vkBindBufferMemory(_device->logic_device(), buffer, buffer_memory, 0));

    BufferCreationInfo info;
    auto lc_buffer = new DefaultBuffer(
        _device,
        buffer,
        buffer_memory,
        size_bytes,
        _device->enable_device_address());
    info.handle = reinterpret_cast<uint64_t>(lc_buffer);
    info.native_handle = lc_buffer->vk_buffer();
    info.element_stride = element_stride;
    info.total_size_bytes = size_bytes;
    return info;
}
namespace {

// CUDA VMM shareable-handle type paired with the Vulkan external-memory handle
// type it is imported as. CUDA-owned memory can only reach Vulkan through an OS
// shareable handle, and which pairing (if any) the driver actually accepts is a
// platform/driver question, so the import tries them in order and keeps the
// first one that succeeds.
struct CudaToVkHandlePair {
    CUmemAllocationHandleType cuda_type;
    VkExternalMemoryHandleTypeFlagBits vk_type;
    luisa::string_view name;
};

[[nodiscard]] luisa::span<const CudaToVkHandlePair> cuda_to_vk_handle_pairs() noexcept {
#ifdef LUISA_PLATFORM_WINDOWS
    static constexpr CudaToVkHandlePair k_pairs[] = {
        {CU_MEM_HANDLE_TYPE_WIN32,
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
         "CU_MEM_HANDLE_TYPE_WIN32 -> VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT"},
        {CU_MEM_HANDLE_TYPE_WIN32_KMT,
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
         "CU_MEM_HANDLE_TYPE_WIN32_KMT -> VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT"},
    };
#else
    static constexpr CudaToVkHandlePair k_pairs[] = {
        {CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
         "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR -> VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT"},
        {CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
         "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR -> VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT"},
    };
#endif
    return {k_pairs, std::size(k_pairs)};
}

void log_cuda_error(luisa::string_view what, CUresult result) noexcept {
    const char *name = "unknown";
    const char *message = "unknown";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    LUISA_WARNING("{} failed: {}: {}.", what, name, message);
}

void log_vulkan_error(luisa::string_view what, VkResult result) noexcept {
    LUISA_WARNING("{} failed: VkResult={}.", what, static_cast<int32_t>(result));
}

// Buffer whose backing pages belong to a CUDA allocation imported into Vulkan.
// The Vulkan objects are the owned ones of DefaultBuffer (VkBuffer + imported
// VkDeviceMemory, both released by ~DefaultBuffer); this part releases the CUDA
// mapping and the CUDA allocation handle afterwards, so a CUDA-owned interop
// buffer is freed by destroying the buffer alone.
class CudaImportedBuffer final : public DefaultBuffer {
    CUcontext _cuda_context{};
    CUdeviceptr _cuda_ptr{};
    size_t _cuda_size{};
    CUmemGenericAllocationHandle _cuda_handle{};

public:
    CudaImportedBuffer(Device *device, VkBuffer vk_buffer, VkDeviceMemory memory,
                       size_t size_bytes, bool device_address_capable,
                       CUcontext cuda_context, CUdeviceptr cuda_ptr, size_t cuda_size,
                       CUmemGenericAllocationHandle cuda_handle) noexcept
        : DefaultBuffer{device, vk_buffer, memory, size_bytes, device_address_capable},
          _cuda_context{cuda_context}, _cuda_ptr{cuda_ptr},
          _cuda_size{cuda_size}, _cuda_handle{cuda_handle} {}
    CudaImportedBuffer(CudaImportedBuffer const &) = delete;
    CudaImportedBuffer(CudaImportedBuffer &&) = delete;
    ~CudaImportedBuffer() override {
        if (_cuda_ptr == 0u) { return; }
        // Resource destruction order is outside the backend's control: the
        // primary context may already be gone if the application keeps the
        // buffer alive past the device/extension, in which case CUDA cannot
        // unmap anything and the OS reclaims the allocation on exit.
        if (cuCtxPushCurrent(_cuda_context) != CUDA_SUCCESS) {
            LUISA_WARNING(
                "Cannot release the CUDA side of an imported interop buffer: "
                "the CUDA primary context is no longer available (destroy "
                "CUDA-owned interop buffers before the device).");
            return;
        }
        struct CudaCtxPopper {
            ~CudaCtxPopper() noexcept {
                CUcontext ctx{nullptr};
                cuCtxPopCurrent(&ctx);
            }
        } _popper;
          if (auto rc = cuMemUnmap(_cuda_ptr, _cuda_size); rc != CUDA_SUCCESS) {
              log_cuda_error("cuMemUnmap(CUDA-owned interop buffer)", rc);
          }
        if (auto rc = cuMemAddressFree(_cuda_ptr, _cuda_size); rc != CUDA_SUCCESS) {
            log_cuda_error("cuMemAddressFree(CUDA-owned interop buffer)", rc);
        }
        if (_cuda_handle != 0u) {
            if (auto rc = cuMemRelease(_cuda_handle); rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemRelease(CUDA-owned interop buffer)", rc);
            }
        }
    }

    [[nodiscard]] CUdeviceptr cuda_ptr() const noexcept { return _cuda_ptr; }
};

}// namespace

BufferCreationInfo VkCudaInteropImpl::create_interop_buffer_from_cuda(
    const Type *element, size_t elem_count, uint64_t *cuda_device_ptr) noexcept {

    auto element_stride = (element == Type::of<void>() ? size_t{1u} : element->size());
    auto size_bytes = element_stride * elem_count;
    if (cuda_device_ptr != nullptr) { *cuda_device_ptr = 0u; }
    // Fail closed: an invalid handle, but a nonzero element stride (Buffer
    // divides the total size by it when it is constructed from this info).
    auto failed = [element_stride]() noexcept {
        BufferCreationInfo info{};
        info.handle = invalid_resource_handle;
        info.native_handle = nullptr;
        info.element_stride = element_stride;
        info.total_size_bytes = 0u;
        return info;
    };
    if (_cuda_device < 0 || _cu_context == nullptr) {
        LUISA_WARNING(
            "No CUDA device is paired with this Vulkan device; "
            "CUDA-owned interop buffers are unavailable.");
        return failed();
    }
    if (size_bytes == 0u) {
        return failed();
    }

    return with_cuda(_cu_context, [&]() noexcept -> BufferCreationInfo {
        // Everything a single attempt may have created, released in reverse
        // order unless the attempt succeeds and hands ownership over.
        struct Attempt {
            VkBuffer vk_buffer{VK_NULL_HANDLE};
            VkDeviceMemory vk_memory{VK_NULL_HANDLE};
            CUmemGenericAllocationHandle cuda_handle{0u};
            CUdeviceptr cuda_ptr{0u};
            size_t cuda_size{0u};
            bool cuda_reserved{false};
            bool cuda_mapped{false};
#ifdef LUISA_PLATFORM_WINDOWS
                // HANDLE for CU_MEM_HANDLE_TYPE_WIN32, D3DKMT_HANDLE (a 32-bit
                // integer) for the KMT variant: keep the widest of the two.
                uintptr_t os_handle{0u};
#else
                int os_fd{-1};
#endif
            bool os_handle_open{false};
        } attempt;

        auto release = [&](Attempt &a) noexcept {
            if (a.vk_memory != VK_NULL_HANDLE) {
                vkFreeMemory(_device->logic_device(), a.vk_memory, Device::alloc_callbacks());
                a.vk_memory = VK_NULL_HANDLE;
            }
            if (a.vk_buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(_device->logic_device(), a.vk_buffer, Device::alloc_callbacks());
                a.vk_buffer = VK_NULL_HANDLE;
            }
#ifdef LUISA_PLATFORM_WINDOWS
            if (a.os_handle_open && a.os_handle != 0u) {
                CloseHandle(reinterpret_cast<HANDLE>(a.os_handle));
            }
#else
            if (a.os_handle_open && a.os_fd >= 0) {
                ::close(a.os_fd);
            }
#endif
            a.os_handle_open = false;
            if (a.cuda_mapped && a.cuda_ptr != 0u) {
                cuMemUnmap(a.cuda_ptr, a.cuda_size);
                a.cuda_mapped = false;
            }
            if (a.cuda_reserved && a.cuda_ptr != 0u) {
                cuMemAddressFree(a.cuda_ptr, a.cuda_size);
                a.cuda_reserved = false;
                a.cuda_ptr = 0u;
            }
            if (a.cuda_handle != 0u) {
                cuMemRelease(a.cuda_handle);
                a.cuda_handle = 0u;
            }
        };

        auto try_pair = [&](const CudaToVkHandlePair &pair) noexcept -> BufferCreationInfo {
            // 1. the Vulkan buffer first: its memory requirements (size and
            //    alignment) bound what CUDA has to allocate.
            VkExternalMemoryBufferCreateInfo external_info{};
            external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
            external_info.handleTypes = pair.vk_type;
            auto usage = static_cast<VkBufferUsageFlags>(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT |
                (_device->enable_device_address() ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0u));
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.pNext = &external_info;
            buffer_info.size = size_bytes;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            _device->allocator().apply_queue_sharing(buffer_info);
            if (auto result = vkCreateBuffer(_device->logic_device(), &buffer_info,
                                             Device::alloc_callbacks(), &attempt.vk_buffer);
                result != VK_SUCCESS) {
                log_vulkan_error("vkCreateBuffer(external)", result);
                release(attempt);
                return failed();
            }

            // Does the driver allow importing this handle type into a buffer?
            VkPhysicalDeviceExternalBufferInfo external_buffer_info{};
            external_buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
            external_buffer_info.flags = 0u;
            external_buffer_info.usage = usage;
            external_buffer_info.handleType = pair.vk_type;
            VkExternalBufferProperties external_buffer_properties{};
            external_buffer_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
            vkGetPhysicalDeviceExternalBufferProperties(
                _device->physical_device(), &external_buffer_info, &external_buffer_properties);
            if ((external_buffer_properties.externalMemoryProperties.externalMemoryFeatures &
                 VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0u) {
                LUISA_VERBOSE_WITH_LOCATION(
                    "Vulkan cannot import buffers from {}.", pair.name);
                release(attempt);
                return failed();
            }

            VkMemoryDedicatedRequirements dedicated_requirements{};
            dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
            VkMemoryRequirements2 requirements{};
            requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            requirements.pNext = &dedicated_requirements;
            VkBufferMemoryRequirementsInfo2 requirements_info{};
            requirements_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            requirements_info.buffer = attempt.vk_buffer;
            vkGetBufferMemoryRequirements2(
                _device->logic_device(), &requirements_info, &requirements);
            auto &mem_requirements = requirements.memoryRequirements;

            // 2. CUDA allocation with a shareable handle of this type.
            CUmemAllocationProp alloc_prop{};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = _cuda_device;
            alloc_prop.requestedHandleTypes = pair.cuda_type;
#ifdef LUISA_PLATFORM_WINDOWS
            // CU_MEM_HANDLE_TYPE_WIN32 hands out a named-object NT handle, so
            // CUDA requires the security descriptor to create it with; KMT
            // handles are process-local and take no metadata.
            WindowsSecurityAttributes security_attributes;
            OBJECT_ATTRIBUTES object_attributes{};
            if (pair.cuda_type == CU_MEM_HANDLE_TYPE_WIN32) {
                object_attributes.Length = sizeof(OBJECT_ATTRIBUTES);
                object_attributes.RootDirectory = nullptr;
                object_attributes.ObjectName = nullptr;
                object_attributes.Attributes = 0u;
                object_attributes.SecurityDescriptor = security_attributes.descriptor();
                object_attributes.SecurityQualityOfService = nullptr;
                alloc_prop.win32HandleMetaData = &object_attributes;
            }
#endif
            size_t granularity = 0u;
            if (auto rc = cuMemGetAllocationGranularity(
                    &granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemGetAllocationGranularity", rc);
                release(attempt);
                return failed();
            }
            if (granularity < mem_requirements.alignment) {
                LUISA_VERBOSE_WITH_LOCATION(
                    "CUDA allocation granularity {} is smaller than the Vulkan "
                    "buffer alignment {}: unsupported pairing.",
                    granularity, mem_requirements.alignment);
                release(attempt);
                return failed();
            }
            auto alloc_size = mem_requirements.size;
            if (alloc_size % granularity != 0u) {
                alloc_size = (alloc_size / granularity + 1u) * granularity;
            }
            if (alloc_size < size_bytes) {
                alloc_size = ((size_bytes + granularity - 1u) / granularity) * granularity;
            }
            if (auto rc = cuMemCreate(&attempt.cuda_handle, alloc_size, &alloc_prop, 0u);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemCreate", rc);
                release(attempt);
                return failed();
            }
            attempt.cuda_size = alloc_size;
            if (auto rc = cuMemAddressReserve(&attempt.cuda_ptr, alloc_size, 0u, 0u, 0u);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemAddressReserve", rc);
                release(attempt);
                return failed();
            }
            attempt.cuda_reserved = true;
            if (auto rc = cuMemMap(attempt.cuda_ptr, alloc_size, 0u, attempt.cuda_handle, 0u);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemMap", rc);
                release(attempt);
                return failed();
            }
            attempt.cuda_mapped = true;
            CUmemAccessDesc access_desc{};
            access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = _cuda_device;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if (auto rc = cuMemSetAccess(attempt.cuda_ptr, alloc_size, &access_desc, 1u);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemSetAccess", rc);
                release(attempt);
                return failed();
            }

            // 3. export the allocation to the OS handle Vulkan will import.
            if (auto rc = cuMemExportToShareableHandle(
#ifdef LUISA_PLATFORM_WINDOWS
                    &attempt.os_handle,
#else
                    &attempt.os_fd,
#endif
                    attempt.cuda_handle, pair.cuda_type, 0u);
                rc != CUDA_SUCCESS) {
                log_cuda_error("cuMemExportToShareableHandle", rc);
                release(attempt);
                return failed();
            }
            attempt.os_handle_open = true;

            // 4. import it into Vulkan. Chain: device-address flags -> import
            //    info -> dedicated allocation info (external memory often
            //    requires a dedicated allocation for the buffer it backs).
            VkMemoryAllocateFlagsInfo alloc_flags_info{};
            alloc_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            if (_device->enable_device_address()) {
                alloc_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            }
#ifdef LUISA_PLATFORM_WINDOWS
            VkImportMemoryWin32HandleInfoKHR import_info{};
            import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            import_info.handleType = pair.vk_type;
            import_info.handle = reinterpret_cast<HANDLE>(attempt.os_handle);
            import_info.name = nullptr;
#else
            VkImportMemoryFdInfoKHR import_info{};
            import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
            import_info.handleType = pair.vk_type;
            import_info.fd = attempt.os_fd;
#endif
            VkMemoryDedicatedAllocateInfo dedicated_info{};
            dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            dedicated_info.buffer = attempt.vk_buffer;
            dedicated_info.image = VK_NULL_HANDLE;
            alloc_flags_info.pNext = &import_info;
            if (dedicated_requirements.requiresDedicatedAllocation ||
                dedicated_requirements.prefersDedicatedAllocation) {
                import_info.pNext = &dedicated_info;
            }
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.pNext = &alloc_flags_info;
            alloc_info.allocationSize = alloc_size;
            alloc_info.memoryTypeIndex = find_memory_type(
                mem_requirements.memoryTypeBits, _device->physical_device(),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (auto result = vkAllocateMemory(_device->logic_device(), &alloc_info,
                                               Device::alloc_callbacks(), &attempt.vk_memory);
                result != VK_SUCCESS) {
                // This is the step where a driver that only accepts handles it
                // exported itself rejects foreign (CUDA-created) memory.
                log_vulkan_error("vkAllocateMemory(import CUDA allocation)", result);
                release(attempt);
                return failed();
            }
            if (auto result = vkBindBufferMemory(_device->logic_device(), attempt.vk_buffer,
                                                 attempt.vk_memory, 0u);
                result != VK_SUCCESS) {
                log_vulkan_error("vkBindBufferMemory(external)", result);
                release(attempt);
                return failed();
            }

            // 5. hand ownership over to the buffer object.
            auto *buffer = new CudaImportedBuffer{
                _device, attempt.vk_buffer, attempt.vk_memory, size_bytes,
                _device->enable_device_address(), _cu_context,
                attempt.cuda_ptr, attempt.cuda_size, attempt.cuda_handle};
            // Vulkan keeps its own reference to the imported handle; the
            // exported one and the CUDA mapping reference are ours to drop.
#ifdef LUISA_PLATFORM_WINDOWS
            CloseHandle(reinterpret_cast<HANDLE>(attempt.os_handle));
#else
            ::close(attempt.os_fd);
#endif
            attempt.os_handle_open = false;
            attempt.vk_buffer = VK_NULL_HANDLE;
            attempt.vk_memory = VK_NULL_HANDLE;
            attempt.cuda_handle = 0u;
            attempt.cuda_reserved = false;
            attempt.cuda_mapped = false;

            if (cuda_device_ptr != nullptr) {
                *cuda_device_ptr = static_cast<uint64_t>(buffer->cuda_ptr());
            }
            LUISA_VERBOSE_WITH_LOCATION(
                "Imported a CUDA allocation of {} bytes at CUDA address {:#x} "
                "into Vulkan using {}.",
                alloc_size, static_cast<unsigned long long>(buffer->cuda_ptr()),
                pair.name);
            BufferCreationInfo info;
            info.handle = reinterpret_cast<uint64_t>(static_cast<Buffer *>(buffer));
            info.native_handle = buffer->vk_buffer();
            info.element_stride = element_stride;
            info.total_size_bytes = size_bytes;
            return info;
        };

        for (auto &&pair : cuda_to_vk_handle_pairs()) {
            if (auto info = try_pair(pair); info.valid()) {
                return info;
            }
        }
        LUISA_WARNING(
            "CUDA-owned memory cannot be imported into Vulkan on this system: "
            "no CUDA shareable handle type is accepted by the Vulkan driver "
            "(the CUDA backend can still use Vulkan-owned interop memory "
            "through create_interop_buffer).");
        return failed();
    });
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

    auto plan = detail::plan_cuda_interop_texture(
        format, dimension, width, height, depth,
        mipmap_levels, simultaneous_access,
        allow_raster_target);
    LUISA_ASSERT(
        plan.valid(),
        "Invalid Vulkan-CUDA interop texture plan for dimension {} and "
        "extent {}x{}x{}: {}.",
        dimension, width, height, depth,
        detail::cuda_interop_texture_plan_status_name(plan.status));

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = plan.image_type;
    image_info.extent = plan.extent;
    image_info.mipLevels = plan.mip_levels;
    image_info.arrayLayers = 1;
    image_info.format = Texture::to_vk_format(format);
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = plan.usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.pNext = &external_memory_info;
    _device->allocator().apply_queue_sharing(image_info);
    LUISA_CHECK_VULKAN(vkCreateImage(_device->logic_device(), &image_info, Device::alloc_callbacks(), &image));

    // compute memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(_device->logic_device(), image, &mem_requirements);
    [[maybe_unused]] auto image_memory_size = mem_requirements.size;

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
        .flags = (VkMemoryAllocateFlags)(_device->enable_device_address() ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0)};
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, _device->physical_device(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    alloc_info.pNext = &alloc_flag_info;
    LUISA_CHECK_VULKAN(vkAllocateMemory(_device->logic_device(), &alloc_info, Device::alloc_callbacks(), &image_memory));
    LUISA_CHECK_VULKAN(vkBindImageMemory(_device->logic_device(), image, image_memory, 0));
    auto tex = new Texture(
        _device,
        image,
        plan.dimension,
        format,
        uint3(
            plan.extent.width,
            plan.extent.height,
            plan.extent.depth),
        plan.mip_levels,
        plan.simultaneous_access,
        image_memory);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(tex),
        .native_handle = tex->vk_image()};
}
void VkCudaInteropImpl::vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept {
    auto evt = reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle);
    evt->_mark_signal_fence(fence_index);
    auto stream = reinterpret_cast<lc::vk::Stream *>(vk_stream);
    auto semaphore = evt->vk_semaphore();
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
    std::lock_guard queue_lock{stream->queue_mtx()};
    auto config_ext = _device->config_ext();
    if (!(config_ext && config_ext->signal_semaphore(
                            stream->queue(), semaphore, fence_index))) {
        LUISA_CHECK_VULKAN(vkQueueSubmit(
            stream->queue(), 1, &info1, VK_NULL_HANDLE));
    }
}
void VkCudaInteropImpl::vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept {
    auto evt = reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle);
    auto stream = reinterpret_cast<lc::vk::Stream *>(vk_stream);
    auto semaphore = evt->vk_semaphore();
    VkTimelineSemaphoreSubmitInfo timelineInfo1{};
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    timelineInfo1.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo1.pNext = nullptr;
    timelineInfo1.waitSemaphoreValueCount = 1;
    timelineInfo1.pWaitSemaphoreValues = &fence_index;
    timelineInfo1.signalSemaphoreValueCount = 0;
    timelineInfo1.pSignalSemaphoreValues = nullptr;
    VkSubmitInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info1.pNext = &timelineInfo1;
    info1.waitSemaphoreCount = 1;
    info1.pWaitSemaphores = &semaphore;
    info1.signalSemaphoreCount = 0;
    info1.pSignalSemaphores = nullptr;
    info1.pWaitDstStageMask = &stage;
    // ... Enqueue initial device work here.
    info1.commandBufferCount = 0;
    info1.pCommandBuffers = nullptr;
    std::lock_guard queue_lock{stream->queue_mtx()};
    auto config_ext = _device->config_ext();
    if (!(config_ext && config_ext->wait_semaphore(
                            stream->queue(), semaphore, fence_index))) {
        LUISA_CHECK_VULKAN(vkQueueSubmit(
            stream->queue(), 1, &info1, VK_NULL_HANDLE));
    }
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
        // NOTE: CUDA external memory buffer mapping via cuExternalMemoryGetMappedBuffer
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc{};
        bufferDesc.offset = 0;
        bufferDesc.size = vk_buffer->byte_size();

        bufferDesc.flags = 0;
        static_assert(sizeof(cuda_ptr) == sizeof(CUdeviceptr *));
        LUISA_CHECK_CUDA(cuExternalMemoryGetMappedBuffer((CUdeviceptr *)cuda_ptr, external_memory, &bufferDesc));
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
void VkCudaInteropImpl::unmap(void *cuda_ptr, void *cuda_handle) noexcept {
    with_cuda(_cu_context, [&] {
        LUISA_CHECK_CUDA(cuMemFree(reinterpret_cast<CUdeviceptr>(cuda_ptr)));
        LUISA_CHECK_CUDA(cuDestroyExternalMemory(reinterpret_cast<CUexternalMemory>(cuda_handle)));
    });
}
DeviceInterface *VkCudaInteropImpl::device() noexcept {
    return _device;
}
CUDADeviceConfigExt::ExternalVkDevice VkCudaInteropImpl::get_external_vk_device() const noexcept {
    return {_device->physical_device(), _device->logic_device()};
}

bool VkCudaInteropImpl::cuda_kernel_launch_supported() const noexcept {
    return _device->enable_cuda_kernel_launch() && _cuda_launch_funcs.valid();
}

uint64_t VkCudaInteropImpl::create_cuda_kernel_shader(uint64_t cuda_shader_handle) noexcept {
    if (!cuda_kernel_launch_supported()) {
        LUISA_WARNING(
            "VK_NV_cuda_kernel_launch is not enabled on this device; "
            "cannot create a CUDA kernel shader.");
        return 0u;
    }
    if (cuda_shader_handle == 0u) {
        LUISA_WARNING("Cannot import a null CUDA shader handle.");
        return 0u;
    }
    // The handle identifies a shader compiled by the cuda backend from a DSL
    // kernel. Only virtual/inline accessors may be used cross-DLL (see
    // cuda_shader.h); the caller guarantees the handle's provenance.
    auto cuda_shader = reinterpret_cast<const luisa::compute::cuda::CUDAShader *>(cuda_shader_handle);
    auto image = cuda_shader->module_image();
    if (image.empty()) {
        LUISA_WARNING(
            "CUDA shader has no importable module image; OptiX ray-tracing "
            "shaders cannot be imported for VK_NV_cuda_kernel_launch.");
        return 0u;
    }
    if (cuda_shader->bound_argument_count() != 0u) {
        LUISA_WARNING(
            "CUDA kernels with bound (captured) arguments cannot be imported "
            "for VK_NV_cuda_kernel_launch.");
        return 0u;
    }
    if (cuda_shader->printer() != nullptr) {
        LUISA_WARNING(
            "CUDA kernels using device printing cannot be imported for "
            "VK_NV_cuda_kernel_launch.");
        return 0u;
    }
    auto entry = cuda_shader->entry();
    if (entry.empty()) {
        LUISA_WARNING("CUDA shader has no entry point; cannot import.");
        return 0u;
    }
    // The image is the PTX text or linked cubin the cuda backend loaded the
    // shader from. The CUDA module loader consumes PTX as a null-terminated
    // string (a trailing byte is harmless for cubins), so guarantee one.
    luisa::vector<std::byte> image_storage{image.begin(), image.end()};
    if (image_storage.empty() || image_storage.back() != std::byte{0}) {
        image_storage.emplace_back(std::byte{0});
    }
    auto shader = new CudaKernelShader{};
    VkCudaModuleCreateInfoNV module_info{
        .sType = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV,
        .pNext = nullptr,
        .dataSize = image_storage.size(),
        .pData = image_storage.data()};
    LUISA_CHECK_VULKAN(_cuda_launch_funcs.create_cuda_module(
        _device->logic_device(), &module_info,
        Device::alloc_callbacks(), &shader->module));
    luisa::string kernel_name{entry};
    VkCudaFunctionCreateInfoNV function_info{
        .sType = VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV,
        .pNext = nullptr,
        .module = shader->module,
        .pName = kernel_name.c_str()};
    if (auto result = _cuda_launch_funcs.create_cuda_function(
            _device->logic_device(), &function_info,
            Device::alloc_callbacks(), &shader->function);
        result != VK_SUCCESS) {
        _cuda_launch_funcs.destroy_cuda_module(
            _device->logic_device(), shader->module, Device::alloc_callbacks());
        delete shader;
        LUISA_CHECK_VULKAN(result);
    }
    // Mirror the source shader's metadata for launch-time validation.
    shader->argument_count = cuda_shader->argument_count();
    shader->block_size = cuda_shader->block_size();
    auto usages = cuda_shader->argument_usages();
    shader->usages.assign(usages.begin(), usages.end());
    return reinterpret_cast<uint64_t>(shader);
}

void VkCudaInteropImpl::destroy_cuda_kernel_shader(uint64_t handle) noexcept {
    auto shader = reinterpret_cast<CudaKernelShader *>(handle);
    if (shader == nullptr) return;
    if (shader->function != VK_NULL_HANDLE) {
        _cuda_launch_funcs.destroy_cuda_function(
            _device->logic_device(), shader->function, Device::alloc_callbacks());
    }
    if (shader->module != VK_NULL_HANDLE) {
        _cuda_launch_funcs.destroy_cuda_module(
            _device->logic_device(), shader->module, Device::alloc_callbacks());
    }
    delete shader;
}

void cuda_launch_kernel(Device *device, VkCommandBuffer cmdbuffer,
                        const vk_cuda_interop::CudaKernelLaunchCommand *cmd) noexcept {
    LUISA_ASSERT(cmd != nullptr, "CUDA kernel launch command is null.");
    auto ext = static_cast<VkCudaInteropImpl *>(device->extension(VkCudaInterop::name));
    if (ext == nullptr || !ext->cuda_kernel_launch_supported()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "VK_NV_cuda_kernel_launch is not enabled on this device.");
    }
    auto shader = reinterpret_cast<CudaKernelShader const *>(cmd->cuda_function());
    if (shader == nullptr || shader->function == VK_NULL_HANDLE) [[unlikely]] {
        // create_cuda_kernel_shader documents returning 0 when unsupported;
        // dispatching such a command is a hard error, not a release-mode crash.
        LUISA_ERROR_WITH_LOCATION(
            "CUDA kernel launch command references an invalid CUDA kernel shader.");
    }
    // Imported shaders are DSL kernels compiled by the cuda backend; validate
    // the command against the metadata recorded at import time.
    if (shader->argument_count != 0u) {
        LUISA_ASSERT(cmd->arguments().size() == shader->argument_count,
                     "CUDA kernel launch argument count mismatch: the command "
                     "has {} argument(s), the imported DSL kernel expects {}.",
                     cmd->arguments().size(), shader->argument_count);
    }
    LUISA_ASSERT(cmd->shared_mem_bytes() == 0u,
                 "Imported DSL kernels use static __shared__ memory; dynamic "
                 "shared memory is not supported for CUDA kernel launch.");
    // Pack the DSL kernel ABI: a single by-value Params blob in which every
    // argument occupies a 16-byte-aligned slot (buffers as LCBuffer
    // {ptr, size_bytes} bindings, uniforms as raw values), followed by the
    // ls_kid trailer {dispatch_size.xyz, kernel_id = 0}. This mirrors
    // CUDAShaderNative::_launch's allocate_argument() layout exactly.
    auto args = cmd->arguments();
    vstd::vector<std::byte> params_blob;
    auto params_blob_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&params_blob, &params_blob_offset](size_t bytes) noexcept {
        static constexpr auto cuda_shader_param_alignment = 16u;
        auto offset = (params_blob_offset + cuda_shader_param_alignment - 1u) /
                      cuda_shader_param_alignment * cuda_shader_param_alignment;
        params_blob.resize(offset + bytes);
        params_blob_offset = offset + bytes;
        return params_blob.data() + offset;
    };
    for (auto &&arg : args) {
        switch (arg.tag) {
            case Argument::Tag::UNIFORM: {
                auto data = cmd->uniform(arg.uniform);
                std::memcpy(allocate_argument(data.size()), data.data(), data.size());
            } break;
            case Argument::Tag::BUFFER: {
                auto buffer = reinterpret_cast<Buffer const *>(arg.buffer.handle);
                LUISA_ASSERT(buffer != nullptr && buffer->device_address_capable(),
                             "CUDA kernel launch buffer arguments must be device-address capable.");
                // LCBuffer<T> { T *ptr; size_t size_bytes; }
                uint64_t binding[2] = {buffer->get_device_address() + arg.buffer.offset,
                                       arg.buffer.size};
                std::memcpy(allocate_argument(sizeof(binding)), binding, sizeof(binding));
            } break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Argument type {} is not supported for imported DSL CUDA "
                    "kernels (buffers and uniforms only).",
                    luisa::to_underlying(arg.tag));
        }
    }
    // ls_kid trailer: exact dispatch size and kernel id (0).
    auto dispatch_size = cmd->dispatch_size();
    auto ls_kid = uint4{dispatch_size.x, dispatch_size.y, dispatch_size.z, 0u};
    std::memcpy(allocate_argument(sizeof(uint4)), &ls_kid, sizeof(uint4));
    // The kernel takes the Params struct by value: a single parameter.
    const void *kernel_params[] = {params_blob.data()};
    VkCudaLaunchInfoNV launch_info{
        .sType = VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV,
        .pNext = nullptr,
        .function = shader->function,
        .gridDimX = cmd->grid_dim().x,
        .gridDimY = cmd->grid_dim().y,
        .gridDimZ = cmd->grid_dim().z,
        .blockDimX = cmd->block_dim().x,
        .blockDimY = cmd->block_dim().y,
        .blockDimZ = cmd->block_dim().z,
        .sharedMemBytes = cmd->shared_mem_bytes(),
        .paramCount = 1u,
        .pParams = kernel_params,
        .extraCount = 0u,
        .pExtras = nullptr};
    ext->cuda_kernel_launch_funcs().cmd_cuda_launch_kernel(cmdbuffer, &launch_info);
}
}// namespace lc::vk
#endif
