#pragma once
#include <d3dx12.h>
#include <vstl/PoolAllocator.h>
#include <Resource/BufferView.h>
class ElementAllocator;
namespace toolhub::directx {
class IGpuAllocator;
class DescriptorHeap;
class Device {
	std::mutex meshAllocMtx;
	vstd::PoolAllocator meshAlloc;

public:
	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	Microsoft::WRL::ComPtr<ID3D12Device5> device;
	Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
	IGpuAllocator* defaultAllocator = nullptr;
	vstd::unique_ptr<DescriptorHeap> globalHeap;
	BufferView AllocateMeshBuffer();
	void DeAllocateMeshBuffer(BufferView const& b);
	Device();
	~Device();
};
}// namespace toolhub::directx