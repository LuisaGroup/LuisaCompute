#pragma once
#include <Common/GFXUtil.h>
#include <util/Runnable.h>
namespace luisa::compute {
class DXEvent {
public:
	DXEvent();
	~DXEvent();
	void AddSignal(
		size_t handle,
		uint64 signalCount);
	void GPUWaitEvent(
		GFXCommandQueue* queue,
		ID3D12Fence* fence);

	void Sync(
		Runnable<void(uint64)>&& syncFunc);
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	vstd::vector<std::pair<size_t, uint64>> signals;
};
}// namespace luisa::compute
