#pragma once
#include <RenderComponent/RenderTexture.h>
#include <runtime/pixel.h>
#include <runtime/texture_sampler.h>
class UploadBuffer;
namespace luisa::compute {
class RenderTexturePackage {
public:
	StackObject<RenderTexture, true> rt;
	PixelFormat format;
	TextureSampler sampler;
	UploadBuffer* descHeap;
	uint descIndex;

	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
}// namespace luisa::compute