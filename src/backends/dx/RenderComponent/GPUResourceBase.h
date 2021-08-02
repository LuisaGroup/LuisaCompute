#pragma once
#include <RenderComponent/IGPUResourceState.h>
#include <RenderComponent/Utility/IGPUAllocator.h>
#include <core/vstl/VObject.h>
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
	vstd::string name;

public:
	IGPUAllocator* GetAllocator() const { return allocator; }
	GPUResourceType GetResourceType() const { return resourceType; }
	GPUResourceBase(GFXDevice* device, GPUResourceType resourceType, IGPUAllocator* allocator);
	vstd::string const& GetName() const { return name; }
	void ReleaseAfterFrame(luisa::compute::FrameResource* resource);
	virtual ~GPUResourceBase();
	virtual GPUResourceState GetInitState() const = 0;
	GFXResource* GetResource() const {
		return Resource.Get();
	}
	void SetName(vstd::string const& str) {
		name = str;
		Resource->SetName(vstd::wstring(str).c_str());
	}
};