#pragma once
#include <stdint.h>
class RenderTexture;

struct RenderTarget
{
	RenderTexture const* rt;
	uint32_t depthSlice;
	uint32_t mipCount;
	RenderTarget(RenderTexture const* rt = nullptr, uint32_t depthSlice = 0, uint32_t mipCount = 0) :
		rt(rt),
		depthSlice(depthSlice),
		mipCount(mipCount)
	{}
};