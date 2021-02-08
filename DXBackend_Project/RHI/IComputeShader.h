#pragma once
#include "IShader.h"
namespace lc_rhi {
class IComputeShader : public IShader {
public:
	virtual ~IComputeShader() = 0;
	virtual void Dispatch(
		ICommandBuffer* commandBuffer,
		uint32_t groupWidth,
		uint32_t groupHeight,
		uint32_t groupDepth) const = 0;
};
}// namespace lc_rhi