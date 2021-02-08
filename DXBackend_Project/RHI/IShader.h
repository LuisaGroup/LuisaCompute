#pragma once
#include <string_view>
#include <stdint.h>
namespace lc_rhi {
class ITexture;
class IBuffer;
class ICommandBuffer;
class IShader {
public:
	virtual void SetGlobalData(ICommandBuffer* commandBuffer, void const* bufferData) const = 0;
	virtual void SetTexture(ICommandBuffer* commandBuffer, std::string_view name, ITexture const* texture) const = 0;
	virtual void SetBuffer(ICommandBuffer* commandBuffer, std::string_view name, IBuffer const* buffer) const = 0;
	virtual ~IShader() {}
};
}// namespace lc_rhi