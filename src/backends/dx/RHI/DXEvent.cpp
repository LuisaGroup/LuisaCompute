#include "DXEvent.h"
namespace luisa::compute {
DXEvent::DXEvent() {
}
DXEvent::~DXEvent() {
}
void DXEvent::AddSignal(size_t handle, uint64 signalCount) {
	if (signalCount == 0) return;
	for (auto& i : signals) {
		if (i.first == handle) {
			i.second = signalCount;
			return;
		}
	}
	signals.emplace_back(handle, signalCount);
}
void DXEvent::GPUWaitEvent(GFXCommandQueue* queue, ID3D12Fence* fence) {
	for (auto& i : signals) {
		queue->Wait(fence, i.second);
	}
}
void DXEvent::Sync(Runnable<void(uint64)>&& syncFunc) {
	for (auto& i : signals) {
		syncFunc(i.second);
	}
}
}// namespace luisa::compute