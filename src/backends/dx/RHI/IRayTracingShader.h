#pragma once
#include <RHI/IShader.h>
namespace lc_rhi {
class IRayTracingShader : public IShader {
public:
	virtual void Dispatch(
		ICommandBuffer* commandBuffer,
		uint width,
		uint height,
		uint depth) const = 0;
	virtual ~IRayTracingShader() {}
};
}// namespace lc_rhi