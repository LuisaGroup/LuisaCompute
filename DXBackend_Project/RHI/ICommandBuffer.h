#pragma once
namespace lc_rhi {
class IGFXDevice;
class ICommandBuffer {
public:
	virtual void Clear() = 0;
	virtual void Submit(IGFXDevice* device) = 0;
	virtual ~ICommandBuffer() {}
};
}// namespace lc_rhi