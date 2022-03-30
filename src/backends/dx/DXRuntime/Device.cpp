#include <DXRuntime/Device.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/IGpuAllocator.h>
#include <Shader/BuiltinKernel.h>
#include <dxgi1_3.h>
#include <Shader/ShaderCompiler.h>
namespace toolhub::directx {
static std::mutex gDxcMutex;
static vstd::optional<DXShaderCompiler> gDxcCompiler;
static int32 gDxcRefCount = 0;
Device::~Device() {
    if (defaultAllocator) delete defaultAllocator;
    CloseHandle(eventHandle);
    {
        std::lock_guard lck(gDxcMutex);
        if (--gDxcRefCount == 0) {
            gDxcCompiler.Delete();
        }
    }
}
DXShaderCompiler *Device::Compiler() {
    return gDxcCompiler;
}

Device::Device() {
    using Microsoft::WRL::ComPtr;
    uint32_t dxgiFactoryFlags = 0;

#if defined(_DEBUG)
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();

            // Enable additional debug layers.
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));
    uint adapterIndex = 0;    // we'll start looking for directx 12  compatible graphics devices starting at index 0
    bool adapterFound = false;// set this to true when a good one was found
    while (dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1,
                                           IID_PPV_ARGS(&device));
            if (SUCCEEDED(hr)) {
                adapterFound = true;
                break;
            }
        }
        adapter = nullptr;
        adapterIndex++;
    }
    defaultAllocator = IGpuAllocator::CreateAllocator(
        this,
        IGpuAllocator::Tag::DefaultAllocator);
    globalHeap = vstd::create_unique(
        new DescriptorHeap(
            this,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            500000,
            true));
    samplerHeap = vstd::create_unique(
        new DescriptorHeap(
            this,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            16,
            true));
    auto samplers = GlobalSamplers::GetSamplers();
    for (auto i : vstd::range(samplers.size())) {
        samplerHeap->CreateSampler(
            samplers[i], i);
    }
    LPCWSTR falseValue = nullptr;
    eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
    setAccelKernel = BuiltinKernel::LoadAccelSetKernel(this);
    {
        std::lock_guard lck(gDxcMutex);
        gDxcRefCount++;
        gDxcCompiler.New();
    }
}
}// namespace toolhub::directx