#include "ext.h"
#ifdef LCDX_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <Resource/Buffer.h>
#include <Resource/TextureBase.h>
#include <DXApi/LCEvent.h>
#include <aclapi.h>
#else
#include <luisa/core/logging.h>
#endif
namespace lc::dx {
#ifdef LCDX_ENABLE_CUDA
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
uint64_t DxCudaInteropImpl::cuda_buffer(uint64_t dx_buffer_handle) noexcept {
    auto dxBuffer = reinterpret_cast<Buffer const *>(dx_buffer_handle);
    WindowsSecurityAttributes windowsSecurityAttributes;
    HANDLE sharedHandle;
    _device.device->CreateSharedHandle(dxBuffer->GetResource(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle);

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = dxBuffer->GetByteSize();
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    cudaExternalMemory_t externalMemory{};
    assert(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
    // TODO: need cuda buffer here
    return reinterpret_cast<uint64_t>(externalMemory);
}
uint64_t DxCudaInteropImpl::cuda_texture(uint64_t dx_texture_handle) noexcept {
    auto dxTex = reinterpret_cast<TextureBase const *>(dx_texture_handle);
    auto allocateInfo = _device.device->GetResourceAllocationInfo(0, 1, vstd::get_rval_ptr(dxTex->GetResource()->GetDesc()));
    WindowsSecurityAttributes windowsSecurityAttributes;
    HANDLE sharedHandle;
    _device.device->CreateSharedHandle(dxTex->GetResource(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle);

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = allocateInfo.SizeInBytes;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    cudaExternalMemory_t externalMemory{};
    assert(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
    // TODO: need cuda buffer here
    return reinterpret_cast<uint64_t>(externalMemory);
}
uint64_t DxCudaInteropImpl::cuda_event(uint64_t dx_event_handle) noexcept {
    auto dxEvent = reinterpret_cast<LCEvent const *>(dx_event_handle);
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;

    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    WindowsSecurityAttributes windowsSecurityAttributes;
    HANDLE sharedHandle;
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    _device.device->CreateSharedHandle(dxEvent->Fence(), &windowsSecurityAttributes, GENERIC_ALL, nullptr, &sharedHandle);
    externalSemaphoreHandleDesc.handle.win32.handle = (void *)sharedHandle;
    externalSemaphoreHandleDesc.flags = 0;
    cudaExternalSemaphore_t externalSemaphre{};
    assert(cudaImportExternalSemaphore(&externalSemaphre, &externalSemaphoreHandleDesc));
    // TODO: need cuda event here
    return reinterpret_cast<uint64_t>(externalSemaphre);
}
#else
#define LUISA_UNIMPL_ERROR LUISA_ERROR("Method unimplemented.")
uint64_t DxCudaInteropImpl::cuda_buffer(uint64_t dx_buffer) noexcept {
    LUISA_UNIMPL_ERROR;
    return invalid_resource_handle;
}
uint64_t DxCudaInteropImpl::cuda_texture(uint64_t dx_texture) noexcept {
    LUISA_UNIMPL_ERROR;
    return invalid_resource_handle;
}
uint64_t DxCudaInteropImpl::cuda_event(uint64_t dx_event) noexcept {
    LUISA_UNIMPL_ERROR;
    return invalid_resource_handle;
}
#undef LUISA_UNIMPL_ERROR
#endif
};// namespace lc::dx
