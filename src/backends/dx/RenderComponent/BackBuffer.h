#pragma once
#include "GPUResourceBase.h"
#include "IBackBuffer.h"
class VEngine;
class BackBuffer final : public GPUResourceBase, public IBackBuffer {
private:
	friend class VEngine;
	D3D12_CPU_DESCRIPTOR_HANDLE handle;
	Microsoft::WRL::ComPtr<GFXResource>& GetResourcePtr() {
		return Resource;
	}

public:
	GPUResourceBase const* GetBackBufferGPUResource() const override {
		return this;
	};
	D3D12_CPU_DESCRIPTOR_HANDLE GetRTVHandle() const override { return handle; }
	GFXResourceState GetInitState() const override {
		return GFXResourceState_Present;
	}
	~BackBuffer();
};
