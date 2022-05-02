#pragma once
#include <d3dx12.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class Resource;
class ResourceStateTracker : public vstd::IOperatorNewBase {
private:
    struct State {
        uint64 fence;
        D3D12_RESOURCE_STATES lastState;
        D3D12_RESOURCE_STATES curState;
        bool uavBarrier;
        bool isWrite;
    };
    vstd::HashMap<Resource const *, State> stateMap;
    vstd::HashMap<Resource const *> writeStateMap;
    vstd::vector<D3D12_RESOURCE_BARRIER> states;
    void ExecuteStateMap();
    void RestoreStateMap();
    void MarkWritable(Resource const *res, bool writable);
    uint64 fenceCount = 1;

public:
    void ClearFence() { fenceCount++; }
    vstd::HashMap<Resource const *> const &WriteStateMap() const { return writeStateMap; }
    ResourceStateTracker();
    ~ResourceStateTracker();
    void RecordState(
        Resource const *resource,
        D3D12_RESOURCE_STATES state,
        bool lock = false);
    void RecordState(Resource const *resource, bool lock = false);
    void UpdateState(CommandBufferBuilder const &cmdBuffer);
    void RestoreState(CommandBufferBuilder const &cmdBuffer);
};
}// namespace toolhub::directx