#pragma once
#include <RenderComponent/IGPUResourceState.h>
#include <Common/VObject.h>
class ThreadCommand;
enum class GPUResourceType : uint8_t {
	Buffer,
	Texture
};
class VENGINE_DLL_RENDERER GPUResourceBase : public VObject, public IGPUResourceState {
	friend class ThreadCommand;

protected:
	Microsoft::WRL::ComPtr<GFXResource> Resource;

private:
	GPUResourceType resourceType;
	vengine::string name;

public:
	GPUResourceType GetResourceType() const { return resourceType; }
	GPUResourceBase(GPUResourceType resourceType);
	vengine::string const& GetName() const { return name; }

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