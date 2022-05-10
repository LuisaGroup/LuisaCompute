
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/Resource.h>
namespace toolhub::directx {
namespace detail {
static bool IsWriteState(D3D12_RESOURCE_STATES state) {
    switch (state) {
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        case D3D12_RESOURCE_STATE_COPY_DEST:
            return true;
        default:
            return false;
    }
}
}// namespace detail
ResourceStateTracker::ResourceStateTracker() {
}
ResourceStateTracker::~ResourceStateTracker() = default;
void ResourceStateTracker::RecordState(
    Resource const *resource,
    D3D12_RESOURCE_STATES state,
    bool lock) {
    auto initState = resource->GetInitState();
    bool newAdd = false;
    bool isWrite = detail::IsWriteState(state);
    auto ite = stateMap.Emplace(
        resource,
        vstd::MakeLazyEval([&] {
            newAdd = true;
            if (isWrite) {
                writeStateMap.Emplace(resource);
            }
            return State{
                .fence = lock ? fenceCount : 0,
                .lastState = initState,
                .curState = state,
                .uavBarrier = (state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && initState == state),
                .isWrite = isWrite};
        }));
    if (!newAdd) {
        auto &&st = ite.Value();
        if (lock) {
            st.fence = fenceCount;
        } else if (st.fence >= fenceCount)
            return;

        st.uavBarrier = (state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && ite.Value().lastState == state);
        st.curState = state;
        if (isWrite != st.isWrite) {
            st.isWrite = isWrite;
            MarkWritable(resource, isWrite);
        }
    }
}
void ResourceStateTracker::RecordState(
    Resource const *resource,
    bool lock) {
    RecordState(resource, resource->GetInitState(), lock);
}
void ResourceStateTracker::ExecuteStateMap() {
    for (auto &&i : stateMap) {
        if (i.second.uavBarrier) {
            D3D12_RESOURCE_BARRIER &uavBarrier = states.emplace_back();
            uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            uavBarrier.UAV.pResource = i.first->GetResource();
            i.second.uavBarrier = false;
        } else if (i.second.curState != i.second.lastState) {
            D3D12_RESOURCE_BARRIER &transBarrier = states.emplace_back();
            transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            transBarrier.Transition.pResource = i.first->GetResource();
            transBarrier.Transition.StateBefore = i.second.lastState;
            transBarrier.Transition.StateAfter = i.second.curState;
        }
        i.second.lastState = i.second.curState;
    }
}
void ResourceStateTracker::RestoreStateMap() {
    for (auto &&i : stateMap) {
        i.second.curState = i.first->GetInitState();
        bool isWrite = detail::IsWriteState(i.second.curState);
        if (isWrite != i.second.isWrite) {
            MarkWritable(i.first, isWrite);
        }
        bool useUavBarrier =
            (i.second.lastState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
             i.second.curState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (useUavBarrier) {
            D3D12_RESOURCE_BARRIER &uavBarrier = states.emplace_back();
            uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            uavBarrier.UAV.pResource = i.first->GetResource();
        } else if (i.second.curState != i.second.lastState) {
            D3D12_RESOURCE_BARRIER &transBarrier = states.emplace_back();
            transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            transBarrier.Transition.pResource = i.first->GetResource();
            transBarrier.Transition.StateBefore = i.second.lastState;
            transBarrier.Transition.StateAfter = i.second.curState;
        }
    }
    stateMap.Clear();
}

void ResourceStateTracker::UpdateState(CommandBufferBuilder const &cmdBuffer) {
    ExecuteStateMap();
    if (!states.empty()) {
        cmdBuffer.CmdList()->ResourceBarrier(
            states.size(),
            states.data());
        states.clear();
    }
}
void ResourceStateTracker::RestoreState(CommandBufferBuilder const &cmdBuffer) {
    RestoreStateMap();
    if (!states.empty()) {
        cmdBuffer.CmdList()->ResourceBarrier(
            states.size(),
            states.data());
        states.clear();
    }
}

void ResourceStateTracker::MarkWritable(Resource const *res, bool writable) {
    if (writable) {
        writeStateMap.Emplace(res);
    } else {
        writeStateMap.Remove(res);
    }
}
}// namespace toolhub::directx
