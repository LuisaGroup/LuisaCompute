#pragma vengine_package vengine_directx
#include <DXRuntime/Device.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/IGpuAllocator.h>
namespace toolhub::directx {
Device::~Device() {
    if (defaultAllocator) delete defaultAllocator;
    CloseHandle(eventHandle);
}

Device::Device() {
    using Microsoft::WRL::ComPtr;
#if defined(_DEBUG)
    // Enable the D3D12 debug layer.
    {
        ComPtr<ID3D12Debug> debugController;
        ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
        debugController->EnableDebugLayer();
    }
#endif
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
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
}
}// namespace toolhub::directx