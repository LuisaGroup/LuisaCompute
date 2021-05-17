#include <RenderComponent/TextureBase.h>
#include <Singleton/Graphics.h>
TextureBase::TextureBase(GFXDevice* device, IGPUAllocator* alloc)
	: GPUResourceBase(device, GPUResourceType::Texture, alloc) {
	srvDescID = Graphics::GetDescHeapIndexFromPool();
}
D3D12_RESOURCE_STATES TextureBase::GetGFXResourceState(GPUResourceState gfxState) const {
	if (gfxState == GPUResourceState_GenericRead) {
		return D3D12_RESOURCE_STATE_GENERIC_READ;
	} else {
		return (D3D12_RESOURCE_STATES)gfxState;
	}
}

TextureBase::~TextureBase() {
	Graphics::ReturnDescHeapIndexToPool(srvDescID);
}
