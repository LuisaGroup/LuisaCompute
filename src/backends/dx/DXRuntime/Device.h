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
class PipelineLibrary;
class Device {
    HANDLE eventHandle;

public:
    HANDLE EventHandle() const { return eventHandle; }
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    IGpuAllocator *defaultAllocator = nullptr;
    vstd::unique_ptr<DescriptorHeap> globalHeap;
    vstd::unique_ptr<DescriptorHeap> samplerHeap;
    Device();
    Device(Device const &) = delete;
    Device(Device &&) = delete;
    ~Device();
};
}// namespace toolhub::directx