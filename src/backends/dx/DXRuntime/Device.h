#pragma once
#include <d3dx12.h>
#include <vstl/PoolAllocator.h>
#include <Resource/BufferView.h>
#include <vstl/VGuid.h>
#include <dxgi1_4.h>
class ElementAllocator;
using Microsoft::WRL::ComPtr;
namespace toolhub::directx {
class IGpuAllocator;
class DescriptorHeap;
class ComputeShader;
class PipelineLibrary;
class DXShaderCompiler;
class Device {
public:
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<IDXGIFactory4> dxgiFactory;
    IGpuAllocator *defaultAllocator = nullptr;
    vstd::unique_ptr<DescriptorHeap> globalHeap;
    vstd::unique_ptr<DescriptorHeap> samplerHeap;
    ComputeShader const *setAccelKernel;
    explicit Device(uint index);
    Device(Device const &) = delete;
    Device(Device &&) = delete;
    ~Device();
    void WaitFence(ID3D12Fence *fence, uint64 fenceIndex);
    void WaitFence_Async(ID3D12Fence *fence, uint64 fenceIndex);
    static DXShaderCompiler *Compiler();
};
}// namespace toolhub::directx