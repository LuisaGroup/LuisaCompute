#ifdef LCDX_ENABLE_CUDA
#include "ext.h"
#include <cuda.h>
#include <Resource/Buffer.h>
#include <Resource/TextureBase.h>
#include <DXApi/LCEvent.h>
#include <DXApi/LCDevice.h>
#include <aclapi.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/RenderTexture.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/core/stl/functional.h>
#include <Shader/ComputeShader.h>
#include "TypeCheck.h"
#include "../../cuda/cuda_stream.h"

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

namespace lc::dx {
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES _win_security_attributes;
    PSECURITY_DESCRIPTOR _win_p_security_descriptor;

public:
    WindowsSecurityAttributes();
    ~WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *get_attributes();
};

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

WindowsSecurityAttributes::WindowsSecurityAttributes()
    : _win_security_attributes{} {
    _win_p_security_descriptor = static_cast<PSECURITY_DESCRIPTOR>(calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**)));
    LUISA_ASSUME(_win_p_security_descriptor != nullptr);

    PSID *ppSID = reinterpret_cast<PSID *>(reinterpret_cast<PBYTE>(_win_p_security_descriptor) + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = reinterpret_cast<PACL *>(reinterpret_cast<PBYTE>(ppSID) + sizeof(PSID*));

    InitializeSecurityDescriptor(_win_p_security_descriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, nullptr, ppACL);

    SetSecurityDescriptorDacl(_win_p_security_descriptor, TRUE, *ppACL, FALSE);

    _win_security_attributes.nLength = sizeof(_win_security_attributes);
    _win_security_attributes.lpSecurityDescriptor = _win_p_security_descriptor;
    _win_security_attributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID *ppSID = reinterpret_cast<PSID *>(reinterpret_cast<PBYTE>(_win_p_security_descriptor) + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = reinterpret_cast<PACL *>(reinterpret_cast<PBYTE>(ppSID) + sizeof(PSID*));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(_win_p_security_descriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::get_attributes() {
    return &_win_security_attributes;
}
void DxCudaInteropImpl::unmap(void *cuda_ptr, void *cuda_handle) noexcept {
    with_cuda(_cu_context, [&] {
        LUISA_CHECK_CUDA(cuMemFree(reinterpret_cast<CUdeviceptr>(cuda_ptr)));
        LUISA_CHECK_CUDA(cuDestroyExternalMemory(reinterpret_cast<CUexternalMemory>(cuda_handle)));
    });
}
void DxCudaInteropImpl::cuda_buffer(uint64_t dx_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle) noexcept {
    with_cuda(_cu_context, [&] {
        auto dxBuffer = reinterpret_cast<Buffer const *>(dx_buffer_handle);
        SECURITY_ATTRIBUTES windowsSecurityAttributes = {};
        windowsSecurityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
        windowsSecurityAttributes.bInheritHandle = TRUE;
        windowsSecurityAttributes.lpSecurityDescriptor = nullptr;
        HANDLE sharedHandle;

        //In order to make this work, the buffers now uses committed resource instead of placed
        if (!SUCCEEDED(_device.native_device.device->CreateSharedHandle(dxBuffer->GetResource(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle))) [[unlikely]] {
            LUISA_ERROR("Failed to create shared handle.");
        }

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc{};
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = dxBuffer->GetByteSize();
        externalMemoryHandleDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CUexternalMemory externalMemory{};
        LUISA_CHECK_CUDA(cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
        *cuda_handle = reinterpret_cast<uint64_t>(externalMemory);
        // TODO: need cuda buffer here
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc{};
        bufferDesc.offset = 0;
        bufferDesc.size = dxBuffer->GetByteSize();
        bufferDesc.flags = 0;
        static_assert(sizeof(*cuda_ptr) == sizeof(CUdeviceptr));
        LUISA_CHECK_CUDA(cuExternalMemoryGetMappedBuffer((CUdeviceptr *)cuda_ptr, externalMemory, &bufferDesc));
    });
}
uint64_t DxCudaInteropImpl::cuda_texture(uint64_t dx_texture_handle) noexcept {
    return with_cuda(_cu_context, [&] {
        auto dxTex = reinterpret_cast<TextureBase const *>(dx_texture_handle);
        auto allocateInfo = _device.native_device.device->GetResourceAllocationInfo(0, 1, vstd::get_rval_ptr(dxTex->GetResource()->GetDesc()));
        WindowsSecurityAttributes windowsSecurityAttributes;
        HANDLE sharedHandle;
        if (!SUCCEEDED(_device.native_device.device->CreateSharedHandle(dxTex->GetResource(), windowsSecurityAttributes.get_attributes(), GENERIC_ALL, nullptr, &sharedHandle))) [[unlikely]] {
            LUISA_ERROR("Failed to create shared handle.");
        }
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc{};
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = allocateInfo.SizeInBytes;
        externalMemoryHandleDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CUexternalMemory externalMemory{};
        LUISA_CHECK_CUDA(cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
        return reinterpret_cast<uint64_t>(externalMemory);
    });
}
ResourceCreationInfo DxCudaInteropImpl::create_interop_event() noexcept {
    ResourceCreationInfo info{};
    auto res = new LCEvent(&_device.native_device, true);
    info.handle = resource_to_handle(res);
    info.native_handle = res->fence();
    return info;
}
void DxCudaInteropImpl::destroy_cuda_event(void *cuda_event_handle) noexcept {
    with_cuda(_cu_context, [&] {
        auto evt = static_cast<CUexternalSemaphore>(cuda_event_handle);
        LUISA_CHECK_CUDA(cuDestroyExternalSemaphore(evt));
    });
}
void *DxCudaInteropImpl::cuda_event(uint64_t dx_event_handle) noexcept {
    return with_cuda(_cu_context, [&] {
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC externalSemaphoreHandleDesc{};
        auto dxEvent = reinterpret_cast<LCEvent *>(dx_event_handle);
        WindowsSecurityAttributes windowsSecurityAttributes;
        HANDLE sharedHandle;
        externalSemaphoreHandleDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
        if (!SUCCEEDED(_device.native_device.device->CreateSharedHandle(dxEvent->fence(), windowsSecurityAttributes.get_attributes(), GENERIC_ALL, nullptr, &sharedHandle))) [[unlikely]] {
            LUISA_ERROR("Failed to create shared handle.");
        }
        externalSemaphoreHandleDesc.handle.win32.handle = static_cast<void *>(sharedHandle);
        externalSemaphoreHandleDesc.handle.win32.name = nullptr;
        externalSemaphoreHandleDesc.flags = 0;
        CUexternalSemaphore externalSemaphore{};
        LUISA_CHECK_CUDA(cuImportExternalSemaphore(&externalSemaphore, &externalSemaphoreHandleDesc));
        return externalSemaphore;
    });
}
void DxCudaInteropImpl::cuda_signal(uint64_t stream_handle, void *event_handle, uint64_t fence) noexcept {
    with_cuda(_cu_context, [&] {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
        params.params.fence.value = fence;
        auto cuda_evt = static_cast<CUexternalSemaphore>(event_handle);
        auto stream = reinterpret_cast<luisa::compute::cuda::CUDAStream *>(stream_handle);
        auto handle = stream->handle();
        LUISA_CHECK_CUDA(cuSignalExternalSemaphoresAsync(
            &cuda_evt, &params, 1,
            handle));
    });
}
void DxCudaInteropImpl::cuda_signal(/*CUStream*/ void *cu_stream_ptr, void *event_handle, uint64_t fence) noexcept {
    with_cuda(_cu_context, [&] {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
        params.params.fence.value = fence;
        auto cuda_evt = static_cast<CUexternalSemaphore>(event_handle);
        LUISA_CHECK_CUDA(cuSignalExternalSemaphoresAsync(
            &cuda_evt, &params, 1,
            static_cast<CUstream>(cu_stream_ptr)));
    });
}

void DxCudaInteropImpl::cuda_wait(uint64_t stream_handle, void *event_handle, uint64_t fence) noexcept {
    with_cuda(_cu_context, [&] {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
        params.params.fence.value = fence;
        auto cuda_evt = static_cast<CUexternalSemaphore>(event_handle);
        auto stream = reinterpret_cast<luisa::compute::cuda::CUDAStream *>(stream_handle);
        auto handle = stream->handle();
        LUISA_CHECK_CUDA(cuWaitExternalSemaphoresAsync(
            &cuda_evt, &params, 1,
            handle));
    });
}
void DxCudaInteropImpl::cuda_wait(/*CUStream*/ void *cu_stream_ptr, void *event_handle, uint64_t fence) noexcept {
    with_cuda(_cu_context, [&] {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
        params.params.fence.value = fence;
        auto cuda_evt = static_cast<CUexternalSemaphore>(event_handle);
        LUISA_CHECK_CUDA(cuWaitExternalSemaphoresAsync(
            &cuda_evt, &params, 1,
            static_cast<CUstream>(cu_stream_ptr)));
    });
}

BufferCreationInfo DxCudaInteropImpl::create_interop_buffer(const Type *element, size_t elem_count) noexcept {
    BufferCreationInfo info{};
    Buffer *res{};
    if (element == Type::of<void>()) {
        info.total_size_bytes = elem_count;
        info.element_stride = 1u;
        res = new DefaultBuffer(
            &_device.native_device,
            info.total_size_bytes,
            nullptr,
            D3D12_RESOURCE_STATE_COMMON, true);
        info.handle = reinterpret_cast<uint64_t>(res);
        info.native_handle = res->GetResource();
        return info;
    }
    if (element->is_custom()) {
        if (element == Type::of<IndirectKernelDispatch>()) {
            info.element_stride = ComputeShader::kDispatchIndirectStride;
            info.total_size_bytes = 4 + info.element_stride * elem_count;
            res = static_cast<Buffer *>(new DefaultBuffer(&_device.native_device, info.total_size_bytes,
                                                          static_cast<GpuAllocator *>(nullptr)));
        } else {
            LUISA_ERROR("Un-known custom type in dx-backend.");
        }
    } else {
        info.total_size_bytes = element->size() * elem_count;
        res = static_cast<Buffer *>(
            new DefaultBuffer(
                &_device.native_device,
                info.total_size_bytes,
                nullptr,
                D3D12_RESOURCE_STATE_COMMON, true));
        info.element_stride = element->size();
    }
    info.handle = reinterpret_cast<uint64_t>(res);
    info.native_handle = res->GetResource();
    return info;
}
ResourceCreationInfo DxCudaInteropImpl::create_interop_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept {
    bool allowUAV = !is_block_compressed(format);
    ResourceCreationInfo info{};
    auto res = new RenderTexture(
        &_device.native_device,
        width,
        height,
        TextureBase::ToGFXFormat(format),
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        allowUAV,
        simultaneous_access,
        allow_raster_target,
        nullptr,
        true);
    info.handle = reinterpret_cast<uint64_t>(res);
    info.native_handle = res->GetResource();
    return info;
}
DeviceInterface *DxCudaInteropImpl::device() noexcept {
    return &_device;
}

static bool initialize_cuda() noexcept {
    static std::once_flag flag;
    static bool success{};
    std::call_once(flag, [] {
        success = cuInit(0) == CUDA_SUCCESS;
    });
    return success;
}

[[nodiscard]] int get_cuda_device_for_d3d12_device(ID3D12Device *d3d12Device) noexcept {
    if (!initialize_cuda()) return -1;
    LUID d3d12Luid = d3d12Device->GetAdapterLuid();
    int cudaDeviceCount = 0;
    if (cuDeviceGetCount(&cudaDeviceCount) != CUDA_SUCCESS) {
        return -1;
    }
    for (auto i = 0; i < cudaDeviceCount; i++) {
        char cudaLuid[sizeof(d3d12Luid.LowPart) + sizeof(d3d12Luid.HighPart)] = {};
        unsigned int cudaNodeMask = 0;
        if (cuDeviceGetLuid(cudaLuid, &cudaNodeMask, i) != CUDA_SUCCESS) continue;
        if (!std::memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
            !std::memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
            LUISA_VERBOSE_WITH_LOCATION("Found cuda device at {} for d3d12 device.", i);
            return i;
        }
    }
    return -1;
}

DxCudaInteropImpl::DxCudaInteropImpl(LCDevice &device) noexcept : _device{device} {
    auto d3d12_device = device.native_device.device.Get();
    _cuda_device = get_cuda_device_for_d3d12_device(d3d12_device);
    if (_cuda_device == -1) return;
    LUISA_CHECK_CUDA(cuDeviceGet(&_cu_device, _cuda_device));
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_cu_context, _cu_device));
}

DxCudaInteropImpl::~DxCudaInteropImpl() noexcept {
    if (_cu_device)
        LUISA_CHECK_CUDA(cuDevicePrimaryCtxRelease(_cu_device));
}

}// namespace lc::dx
#endif
