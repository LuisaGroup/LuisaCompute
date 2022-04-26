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
    {
        std::lock_guard lck(gDxcMutex);
        if (--gDxcRefCount == 0) {
            gDxcCompiler.Delete();
        }
    }
}

void Device::WaitFence(ID3D12Fence *fence, uint64 fenceIndex) {
    if (fenceIndex <= 0) return;
    HANDLE eventHandle = CreateEventEx(nullptr, (LPCWSTR) nullptr, false, EVENT_ALL_ACCESS);
    auto d = vstd::create_disposer([&] {
        CloseHandle(eventHandle);
    });
    if (fence->GetCompletedValue() < fenceIndex) {
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
}
void Device::WaitFence_Async(ID3D12Fence *fence, uint64 fenceIndex) {
    if (fenceIndex <= 0) return;
    HANDLE eventHandle = CreateEventEx(nullptr, (LPCWSTR)nullptr, false, EVENT_ALL_ACCESS);
    auto d = vstd::create_disposer([&] {
        CloseHandle(eventHandle);
    });
    while (fence->GetCompletedValue() < fenceIndex) {
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, 1);
    }
}

DXShaderCompiler *Device::Compiler() {
    return gDxcCompiler;
}

Device::Device(uint index) {
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
    auto capableAdapterIndex = 0u;
    for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1,
                                           IID_PPV_ARGS(&device));
            if (SUCCEEDED(hr) && capableAdapterIndex++ == index) {
                std::wstring s{desc.Description};
                luisa::string ss(s.size(), ' ');
                std::transform(s.cbegin(), s.cend(), ss.begin(), [](auto c) noexcept { return static_cast<char>(c); });
                LUISA_INFO("Found capable DirectX device at index {}: {}.", index, ss);
                break;
            }
            device = nullptr;
            adapter = nullptr;
        }
    }
    if (adapter == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to create DirectX device at index {}.", index); }
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
    {
        std::lock_guard lck(gDxcMutex);
        gDxcRefCount++;
        gDxcCompiler.New();
    }
    setAccelKernel = BuiltinKernel::LoadAccelSetKernel(this);
}
}// namespace toolhub::directx