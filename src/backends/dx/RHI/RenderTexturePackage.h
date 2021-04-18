#pragma once
#include <RenderComponent/RenderTexture.h>
#include <runtime/pixel.h>
namespace luisa::compute {
class RenderTexturePackage {
public:
	StackObject<RenderTexture, true> rt;
	PixelFormat format;
	bool bindless;
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
}// namespace luisa::compute