#pragma once
#include "IShader.h"
namespace lc_rhi {
class IRayTracingShader : public IShader {
public:
	virtual void Dispatch(
		ICommandBuffer* commandBuffer,
		uint32_t width,
		uint32_t height,
		uint32_t depth) const = 0;
	virtual ~IRayTracingShader() {}
};
}// namespace lc_rhi