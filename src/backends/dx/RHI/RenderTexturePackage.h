#pragma once
#include <RenderComponent/RenderTexture.h>
#include <runtime/pixel.h>
#include <runtime/sampler.h>
class UploadBuffer;
namespace luisa::compute {
class RenderTexturePackage {
public:
	StackObject<RenderTexture, true> rt;
	PixelFormat format;
	Sampler sampler;
	UploadBuffer* descHeap;
	uint descIndex;

	VSTL_OVERRIDE_OPERATOR_NEW
};
}// namespace luisa::compute
