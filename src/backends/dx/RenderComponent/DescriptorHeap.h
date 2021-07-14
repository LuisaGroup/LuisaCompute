#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/vector.h>
class GPUResourceBase;
class ThreadCommand;
enum class BindType : uint {
	None,
	SRV,
	UAV,
	RTV,
	DSV,
	Sampler
};
class VENGINE_DLL_RENDERER DescriptorHeap : public VObject {
	friend class ThreadCommand;

public:
	DescriptorHeap(
		GFXDevice* pDevice,
		D3D12_DESCRIPTOR_HEAP_TYPE Type,
		uint64 NumDescriptors,
		bool bShaderVisible = false);
	DescriptorHeap() : pDH(nullptr), recorder(nullptr) {
		memset(&Desc, 0, sizeof(D3D12_DESCRIPTOR_HEAP_DESC));
	}
	void Create(GFXDevice* pDevice,
				D3D12_DESCRIPTOR_HEAP_TYPE Type,
				uint64 NumDescriptors,
				bool bShaderVisible = false);
	ID3D12DescriptorHeap* Get() const { return pDH.Get(); }
	constexpr D3D12_GPU_DESCRIPTOR_HANDLE hGPU(uint64 index) const {
		if (index >= Desc.NumDescriptors) index = Desc.NumDescriptors - 1;
		D3D12_GPU_DESCRIPTOR_HANDLE h = {hGPUHeapStart.ptr + index * HandleIncrementSize};
		return h;
	}
	void SetDescriptorHeap(ThreadCommand* tCmd) const;
	constexpr D3D12_DESCRIPTOR_HEAP_DESC GetDesc() const { return Desc; };
	constexpr uint64 Size() const { return Desc.NumDescriptors; }
	void CreateUAV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, uint64 index);
	void CreateSRV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, uint64 index);
	void CreateRTV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, uint64 index);
	void CreateDSV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, uint64 index);
	void CreateSampler(
		GFXDevice* device,
		VObject* resource,
		D3D12_SAMPLER_DESC const* sampDesc,
		uint64 index);
	BindType GetBindType(uint64 index) const;
	void ClearView(uint64 index);
	~DescriptorHeap();
	constexpr D3D12_CPU_DESCRIPTOR_HANDLE hCPU(uint64 index) const {
		if (index >= Desc.NumDescriptors) index = Desc.NumDescriptors - 1;
		D3D12_CPU_DESCRIPTOR_HANDLE h = {hCPUHeapStart.ptr + index * HandleIncrementSize};
		return h;
	}

private:
	struct alignas(uint64) BindData {
		BindType type;
		uint64 instanceID;
		bool operator==(const BindData& data) const noexcept {
			return type == data.type && instanceID == data.instanceID;
		}
		bool operator!=(const BindData& data) const noexcept {
			return !operator==(data);
		}
	};
	BindData* recorder = nullptr;
	mutable spin_mutex mtx;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> pDH;
	D3D12_DESCRIPTOR_HEAP_DESC Desc;
	D3D12_CPU_DESCRIPTOR_HANDLE hCPUHeapStart;
	D3D12_GPU_DESCRIPTOR_HANDLE hGPUHeapStart;
	uint HandleIncrementSize;
};