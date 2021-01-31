#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/VObject.h"
class GPUResourceBase : public VObject {
protected:
	Microsoft::WRL::ComPtr<GFXResource> Resource;

private:
	vengine::string name;

public:
	GPUResourceBase();
	vengine::string const& GetName() const { return name; }
	virtual ~GPUResourceBase();
	virtual GFXResourceState GetInitState() const = 0;
	GFXResource* GetResource() const {
		return Resource.Get();
	}
	void SetName(vengine::string const& str) {
		name = str;
		Resource->SetName(vengine::wstring(str).c_str());
	}
};