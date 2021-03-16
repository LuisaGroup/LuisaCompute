#pragma once
namespace lc_rhi {
class IGFXDevice;
class IRenderTexture;
class ICommandBuffer {
public:
	virtual void Clear() = 0;
	virtual void BlitToScreen(IRenderTexture const* rt) = 0;
	virtual ~ICommandBuffer() {}
};
}// namespace lc_rhi