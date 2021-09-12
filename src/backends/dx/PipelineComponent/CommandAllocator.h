#pragma once
#include <Common/GFXUtil.h>
class CommandAllocator
{
	Microsoft::WRL::ComPtr<GFXCommandAllocator> allocator;
	uint64 updatedFrame = 0;
public:
	CommandAllocator(GFXDevice* device, GFXCommandListType type);
	void Reset(uint64 frameIndex);
	Microsoft::WRL::ComPtr<GFXCommandAllocator> const& GetAllocator()
	{
		return allocator;
	}
	VSTL_OVERRIDE_OPERATOR_NEW
};