#pragma once
#include <RenderComponent/IGPUResourceState.h>
#include <RenderComponent/Utility/IGPUAllocator.h>
#include <Common/VObject.h>
namespace luisa::compute {
class FrameResource;
}
class ThreadCommand;
enum class GPUResourceType : uint8_t {
	Buffer,
	Texture
};
class VENGINE_DLL_RENDERER GPUResourceBase : public VObject, public IGPUResourceState {
	friend class ThreadCommand;

protected:
	Microsoft::WRL::ComPtr<GFXResource> Resource;
	GFXDevice* device;
	IGPUAllocator* allocator;

private:
	GPUResourceType resourceType;
	vengine::string name;

public:
	IGPUAllocator* GetAllocator() const { return allocator; }
	GPUResourceType GetResourceType() const { return resourceType; }
	GPUResourceBase(GFXDevice* device, GPUResourceType resourceType, IGPUAllocator* allocator);
	vengine::string const& GetName() const { return name; }
	void ReleaseAfterFrame(luisa::compute::FrameResource* resource);
	virtual ~GPUResourceBase();
	virtual GPUResourceState GetInitState() const = 0;
	GFXResource* GetResource() const {
		return Resource.Get();
	}
	void SetName(vengine::string const& str) {
		name = str;
		Resource->SetName(vengine::wstring(str).c_str());
	}
};