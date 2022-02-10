#pragma once
#include <d3dx12.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class Resource;
class ResourceStateTracker : public vstd::IOperatorNewBase {
private:
	struct State {
		D3D12_RESOURCE_STATES lastState;
		D3D12_RESOURCE_STATES curState;
		bool uavBarrier;
	};
	vstd::HashMap<Resource const*, State> stateMap;
	vstd::vector<D3D12_RESOURCE_BARRIER> states;
    void ExecuteStateMap();

public:
	ResourceStateTracker();
	~ResourceStateTracker();
	void RecordState(
		Resource const* resource,
		D3D12_RESOURCE_STATES state);
	void RecordState(Resource const* resource);
	void UpdateState(CommandBufferBuilder const& cmdBuffer);
	void RestoreState(CommandBufferBuilder const& cmdBuffer);
};
}// namespace toolhub::directx