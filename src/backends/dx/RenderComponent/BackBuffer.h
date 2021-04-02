#pragma once
#include "GPUResourceBase.h"
#include "IBackBuffer.h"
class VEngine;
class VENGINE_DLL_RENDERER BackBuffer final : public GPUResourceBase, public IBackBuffer {
private:
	friend class VEngine;
	D3D12_CPU_DESCRIPTOR_HANDLE handle;
	Microsoft::WRL::ComPtr<GFXResource>& GetResourcePtr() {
		return Resource;
	}

public:
	BackBuffer() : GPUResourceBase(GPUResourceType::Texture) {}
	GPUResourceBase const* GetBackBufferGPUResource() const override {
		return this;
	};
	D3D12_CPU_DESCRIPTOR_HANDLE GetRTVHandle() const override { return handle; }
	GPUResourceState GetInitState() const override {
		return GPUResourceState_Present;
	}
	~BackBuffer();
	GFXFormat GetBackBufferFormat() const;
};
