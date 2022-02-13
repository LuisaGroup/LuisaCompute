#pragma once
#include <d3dx12.h>
#include <vstl/PoolAllocator.h>
#include <Resource/BufferView.h>
#include <vstl/VGuid.h>
class ElementAllocator;
using Microsoft::WRL::ComPtr;
namespace toolhub::directx {
class IGpuAllocator;
class DescriptorHeap;
class ComputeShader;
class Device {
    std::mutex computeAllocMtx;
    vstd::HashMap<ComputeShader const *, vstd::Guid> collectPipeline;

public:
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    IGpuAllocator *defaultAllocator = nullptr;
    vstd::unique_ptr<DescriptorHeap> globalHeap;
    Device();
    ~Device();
    void CreateShader(ComputeShader const *cs, vstd::Guid guid);
    void DestroyShader(ComputeShader const *cs);
    void IteratePipeline(vstd::function<bool(ComputeShader const *, vstd::Guid)> const &func);
};
}// namespace toolhub::directx