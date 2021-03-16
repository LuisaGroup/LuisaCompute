#pragma once
#include "IShader.h"
namespace lc_rhi {
class IComputeShader : public IShader {
public:
	virtual ~IComputeShader() = 0;
	virtual void Dispatch(
		ICommandBuffer* commandBuffer,
		uint groupWidth,
		uint groupHeight,
		uint groupDepth) const = 0;
};
}// namespace lc_rhi