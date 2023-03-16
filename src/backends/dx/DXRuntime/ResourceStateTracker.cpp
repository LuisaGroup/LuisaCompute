#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/TextureBase.h>
namespace toolhub::directx {
namespace detail {
static bool IsReadState(D3D12_RESOURCE_STATES state) {
    switch (state) {
        case D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:
        case D3D12_RESOURCE_STATE_INDEX_BUFFER:
        case D3D12_RESOURCE_STATE_DEPTH_READ:
        case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT:
        case D3D12_RESOURCE_STATE_COPY_SOURCE:
        case D3D12_RESOURCE_STATE_RESOLVE_SOURCE:
            return true;
        default:
            return false;
    }
}
static bool IsUAV(D3D12_RESOURCE_STATES state) {
    switch (state) {
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        case D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
            return true;
        default:
            return false;
    }
}
static bool IsWriteState(D3D12_RESOURCE_STATES state) {
    switch (state) {
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        case D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
        case D3D12_RESOURCE_STATE_DEPTH_WRITE:
        case D3D12_RESOURCE_STATE_COPY_DEST:
        case D3D12_RESOURCE_STATE_RENDER_TARGET:
            return true;
        default:
            return false;
    }
}
static bool ShouldIgnoreState(D3D12_RESOURCE_STATES lastState, D3D12_RESOURCE_STATES curState) {
    return IsReadState(curState) && ((lastState & curState) == curState);
}
}// namespace detail
D3D12_RESOURCE_STATES ResourceStateTracker::GetState(Resource const *res) const {
    auto iter = stateMap.find(res);
    if (iter == stateMap.end()) {
        auto nonSimulState = res->GetNonSimulCurrentState();
        if (luisa::to_underlying(nonSimulState) == std::numeric_limits<uint32_t>::max()) {
            return res->GetInitState();
        } else {
            return nonSimulState;
        }
    }
    return iter->second.curState;
}
ResourceStateTracker::ResourceStateTracker() {}
ResourceStateTracker::~ResourceStateTracker() = default;
void ResourceStateTracker::RecordState(
    Resource const *resource,
    D3D12_RESOURCE_STATES state,
    bool lock) {
    bool isWrite = detail::IsWriteState(state);
    auto ite = stateMap.try_emplace(
        resource,
        vstd::lazy_eval([&] {
            if (isWrite) {
                writeStateMap.emplace(resource);
            }
            auto initState = resource->GetNonSimulCurrentState();
            if (luisa::to_underlying(initState) == std::numeric_limits<uint32_t>::max()) {
                initState = resource->GetInitState();
            }
            return State{
                .fence = lock ? fenceCount : 0,
                .lastState = initState,
                .curState = state,
                .uavBarrier = (detail::IsUAV(state) && initState == state),
                .isWrite = isWrite};
        }));
    if (!ite.second) {
        auto &&st = ite.first->second;
        if (lock) {
            st.fence = fenceCount;
        } else if (st.fence >= fenceCount)
            return;

        st.uavBarrier = (detail::IsUAV(state) && st.lastState == state);
        if (!st.uavBarrier && detail::IsReadState(st.curState) && detail::IsReadState(state)) {
            st.curState |= state;
        } else {
            st.curState = state;
        }
        if (detail::ShouldIgnoreState(st.lastState, st.curState)) {
            st.curState = st.lastState;
        }
        if (isWrite != st.isWrite) {
            st.isWrite = isWrite;
            MarkWritable(resource, isWrite);
        }
    }
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
        auto res = i.first;
        if (res->IsNonSimulResource()) {
            res->SetNonSimulCurrentState(i.second.curState);
        } else {
            i.second.curState = res->GetInitState();
        }
        bool useUavBarrier =
            (i.second.lastState == i.second.curState) &&
            (detail::IsUAV(i.second.lastState) &&
             detail::IsUAV(i.second.curState));

        if (useUavBarrier) {
            D3D12_RESOURCE_BARRIER &uavBarrier = states.emplace_back();
            uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            uavBarrier.UAV.pResource = res->GetResource();
        } else if (i.second.curState != i.second.lastState) {
            D3D12_RESOURCE_BARRIER &transBarrier = states.emplace_back();
            transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            transBarrier.Transition.pResource = res->GetResource();
            transBarrier.Transition.StateBefore = i.second.lastState;
            transBarrier.Transition.StateAfter = i.second.curState;
        }
    }
    stateMap.clear();
}

void ResourceStateTracker::UpdateState(CommandBufferBuilder const &cmdBuffer) {
    ExecuteStateMap();
    if (!states.empty()) {
        cmdBuffer.GetCB()->CmdList()->ResourceBarrier(
            states.size(),
            states.data());
        states.clear();
    }
}
void ResourceStateTracker::RestoreState(CommandBufferBuilder const &cmdBuffer) {
    RestoreStateMap();
    if (!states.empty()) {
        cmdBuffer.GetCB()->CmdList()->ResourceBarrier(
            states.size(),
            states.data());
        states.clear();
    }
    writeStateMap.clear();
    fenceCount = 1;
}

void ResourceStateTracker::MarkWritable(Resource const *res, bool writable) {
    if (writable) {
        writeStateMap.emplace(res);
    } else {
        writeStateMap.erase(res);
    }
}
D3D12_RESOURCE_STATES ResourceStateTracker::ReadState(ResourceReadUsage usage, Resource const *res) const {
    if (res && res->GetTag() == Resource::Tag::DepthBuffer) {
        switch (usage) {
            case ResourceReadUsage::Srv:
                return D3D12_RESOURCE_STATE_DEPTH_READ;
            case ResourceReadUsage::CopySource:
                return D3D12_RESOURCE_STATE_COPY_SOURCE;
            default:
                assert(false);
        }
    } else {
        static constexpr D3D12_RESOURCE_STATES computeStates[] = {
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE};
        static constexpr D3D12_RESOURCE_STATES copyStates[] = {
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE};
        static constexpr D3D12_RESOURCE_STATES graphicsStates[] = {
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE};
        switch (listType) {
            case D3D12_COMMAND_LIST_TYPE_COMPUTE:
                return computeStates[eastl::to_underlying(usage)];
            case D3D12_COMMAND_LIST_TYPE_COPY:
                return copyStates[eastl::to_underlying(usage)];
            default:
                return graphicsStates[eastl::to_underlying(usage)];
        }
    }
}
}// namespace toolhub::directx
