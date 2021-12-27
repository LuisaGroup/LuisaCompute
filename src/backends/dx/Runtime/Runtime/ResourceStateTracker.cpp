#pragma vengine_package vengine_directx
#include <Runtime/ResourceStateTracker.h>
#include <Runtime/CommandBuffer.h>
#include <Resource/Resource.h>
namespace toolhub::directx {
ResourceStateTracker::ResourceStateTracker() {
}
ResourceStateTracker::~ResourceStateTracker() = default;

void ResourceStateTracker::RecordState(Resource const* resource, D3D12_RESOURCE_STATES state) {
	bool newAdd = false;
	auto ite = stateMap.Emplace(
		resource,
		vstd::MakeLazyEval([&] {
			newAdd = true;
			auto initState = resource->GetInitState();
			return State{
				.lastState = initState,
				.curState = state,
				.uavBarrier = false};
		}));
	if (!newAdd) {
		auto&& st = ite.Value();
		st.uavBarrier = (state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS
						 && ite.Value().lastState == state);
		st.curState = state;
	}
}
void ResourceStateTracker::RecordState(Resource const* resource) {
	RecordState(resource, resource->GetInitState());
}

void ResourceStateTracker::UpdateState(CommandBufferBuilder const& cmdBuffer) {
	for (auto&& i : stateMap) {
		if (i.second.uavBarrier) {
			D3D12_RESOURCE_BARRIER& uavBarrier = states.emplace_back();
			uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			uavBarrier.UAV.pResource = i.first->GetResource();
			i.second.uavBarrier = false;
		} else if (i.second.curState != i.second.lastState) {
			D3D12_RESOURCE_BARRIER& transBarrier = states.emplace_back();
			transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			transBarrier.Transition.pResource = i.first->GetResource();
			transBarrier.Transition.StateBefore = i.second.lastState;
			transBarrier.Transition.StateAfter = i.second.curState;
		}
		i.second.lastState = i.second.curState;
	}
	if (!states.empty()) {
		cmdBuffer.CmdList()->ResourceBarrier(
			states.size(),
			states.data());
		states.clear();
	}
}
void ResourceStateTracker::RestoreState(CommandBufferBuilder const& cmdBuffer) {
	for (auto&& i : stateMap) {
		auto curState = i.first->GetInitState();
		if (curState != i.second.lastState) {
			D3D12_RESOURCE_BARRIER& transBarrier = states.emplace_back();
			transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			transBarrier.Transition.pResource = i.first->GetResource();
			transBarrier.Transition.StateBefore = i.second.lastState;
			transBarrier.Transition.StateAfter = curState;
			i.second.lastState = curState;
		}
	}
	stateMap.Clear();
	if (!states.empty()) {
		cmdBuffer.CmdList()->ResourceBarrier(
			states.size(),
			states.data());
		states.clear();
	}
}
}// namespace toolhub::directx
