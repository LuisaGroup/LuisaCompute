#pragma once
#include <Common/GFXUtil.h>
class SeparableRendererManager;
class SeparableRenderer {
	friend class SeparableRendererManager;

public:
	virtual ~SeparableRenderer() {}
	uint GetListIndex() const { return listIndex; }
	bool ISRendererRemoved() const { return rendererRemoved; }

private:
	uint64 lastUpdatedFrame = 0;
	uint listIndex = -1;
	bool rendererRemoved = false;
};