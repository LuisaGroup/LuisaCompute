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
#include <Shader/ComputeShader.h>
#include "TypeCheck.h"

#define LUISA_CHECK_CUDA(...)                            \
    do {                                                 \
        if (auto ec = __VA_ARGS__; ec != CUDA_SUCCESS) { \
            const char *err_name = nullptr;              \
            const char *err_string = nullptr;            \
            cuGetErrorName(ec, &err_name);               \
            cuGetErrorString(ec, &err_string);           \
            LUISA_ERROR_WITH_LOCATION(                   \
                "{}: {}", err_name, err_string);         \
        }                                                \
    } while (false)

namespace lc::dx {
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    ~WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *operator&();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

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

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

    SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

    m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
    m_winSecurityAttributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
    return &m_winSecurityAttributes;
}
void DxCudaInteropImpl::unmap(void *cuda_ptr, void *cuda_handle) noexcept {
    LUISA_CHECK_CUDA(cuMemFree(reinterpret_cast<CUdeviceptr>(cuda_ptr)));
    LUISA_CHECK_CUDA(cuDestroyExternalMemory(reinterpret_cast<CUexternalMemory>(cuda_handle)));
}
void DxCudaInteropImpl::cuda_buffer(uint64_t dx_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle) noexcept {
    auto dxBuffer = reinterpret_cast<Buffer const *>(dx_buffer_handle);
    SECURITY_ATTRIBUTES windowsSecurityAttributes = {};
    windowsSecurityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    windowsSecurityAttributes.bInheritHandle = TRUE;
    windowsSecurityAttributes.lpSecurityDescriptor = nullptr;
    HANDLE sharedHandle;

    //In order to make this work, the buffers now uses committed resource instead of placed
    if (!SUCCEEDED(_device.nativeDevice.device->CreateSharedHandle(dxBuffer->GetResource(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle))) {
        LUISA_ERROR("Failed to create shared handle");
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
    LUISA_CHECK_CUDA(cuExternalMemoryGetMappedBuffer(cuda_ptr, externalMemory, &bufferDesc));
}
uint64_t DxCudaInteropImpl::cuda_texture(uint64_t dx_texture_handle) noexcept {
    auto dxTex = reinterpret_cast<TextureBase const *>(dx_texture_handle);
    auto allocateInfo = _device.nativeDevice.device->GetResourceAllocationInfo(0, 1, vstd::get_rval_ptr(dxTex->GetResource()->GetDesc()));
    WindowsSecurityAttributes windowsSecurityAttributes;
    HANDLE sharedHandle;
    _device.nativeDevice.device->CreateSharedHandle(dxTex->GetResource(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc{};
    externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = allocateInfo.SizeInBytes;
    externalMemoryHandleDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CUexternalMemory externalMemory{};
    LUISA_CHECK_CUDA(cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
    return reinterpret_cast<uint64_t>(externalMemory);
}
uint64_t DxCudaInteropImpl::cuda_event(uint64_t dx_event_handle) noexcept {
    auto dxEvent = reinterpret_cast<LCEvent const *>(dx_event_handle);
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC externalSemaphoreHandleDesc{};
    WindowsSecurityAttributes windowsSecurityAttributes;
    HANDLE sharedHandle;
    externalSemaphoreHandleDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    _device.nativeDevice.device->CreateSharedHandle(dxEvent->Fence(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle);
    externalSemaphoreHandleDesc.handle.win32.handle = (void *)sharedHandle;
    externalSemaphoreHandleDesc.flags = 0;
    CUexternalSemaphore externalSemaphre{};
    LUISA_CHECK_CUDA(cuImportExternalSemaphore(&externalSemaphre, &externalSemaphoreHandleDesc));
    return reinterpret_cast<uint64_t>(externalSemaphre);
}
BufferCreationInfo DxCudaInteropImpl::create_interop_buffer(const Type *element, size_t elem_count) noexcept {
    BufferCreationInfo info{};
    Buffer *res{};
    if (element == Type::of<void>()) {
        info.total_size_bytes = elem_count;
        info.element_stride = 1u;
        res = new DefaultBuffer(
            &_device.nativeDevice,
            info.total_size_bytes,
            nullptr,
            D3D12_RESOURCE_STATE_COMMON, true);
        info.handle = reinterpret_cast<uint64_t>(res);
        info.native_handle = res->GetResource();
        return info;
    }
    if (element->is_custom()) {
        if (element == Type::of<IndirectKernelDispatch>()) {
            info.element_stride = ComputeShader::DispatchIndirectStride;
            info.total_size_bytes = 4 + info.element_stride * elem_count;
            res = static_cast<Buffer *>(new DefaultBuffer(&_device.nativeDevice, info.total_size_bytes,
                                                          static_cast<GpuAllocator *>(nullptr)));
        } else {
            LUISA_ERROR("Un-known custom type in dx-backend.");
        }
    } else {
        info.total_size_bytes = element->size() * elem_count;
        res = static_cast<Buffer *>(
            new DefaultBuffer(
                &_device.nativeDevice,
                info.total_size_bytes,
                nullptr,
                D3D12_RESOURCE_STATE_COMMON, true));
        info.element_stride = element->size();
    }
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->GetResource();
    return info;
}
ResourceCreationInfo DxCudaInteropImpl::create_interop_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    bool allowUAV = !is_block_compressed(format);
    ResourceCreationInfo info{};
    auto res = new RenderTexture(
        &_device.nativeDevice,
        width,
        height,
        TextureBase::ToGFXFormat(format),
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        allowUAV,
        simultaneous_access,
        nullptr,
        true);
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->GetResource();
    return info;
}
DeviceInterface *DxCudaInteropImpl::device() {
    return &_device;
}
}// namespace lc::dx
#endif
