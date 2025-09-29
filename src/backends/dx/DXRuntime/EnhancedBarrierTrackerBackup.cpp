#include "EnhancedBarrierTrackerBackup.h"
#include "CommandBuffer.h"
namespace lc::dx {
D3D12_RESOURCE_STATES EnhancedBarrierTrackerBackup::ToStates(
    D3D12_BARRIER_SYNC sync,
    D3D12_BARRIER_ACCESS access,
    D3D12_BARRIER_LAYOUT layout) {
    auto state = D3D12_RESOURCE_STATE_COMMON;
    if ((access & (D3D12_BARRIER_ACCESS_CONSTANT_BUFFER | D3D12_BARRIER_ACCESS_VERTEX_BUFFER)) != 0) {
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    }
    if ((access & D3D12_BARRIER_ACCESS_INDEX_BUFFER) != 0)
        state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;

    if ((access & D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ) != 0)
        state |= D3D12_RESOURCE_STATE_DEPTH_READ;
    if ((access & D3D12_BARRIER_ACCESS_SHADER_RESOURCE) != 0) {
        D3D12_RESOURCE_STATES st{};
        if (sync & (D3D12_BARRIER_SYNC_ALL |
                    D3D12_BARRIER_SYNC_DRAW |
                    D3D12_BARRIER_SYNC_ALL_SHADING)) {
            st |= D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        } else {
            if ((sync & D3D12_BARRIER_SYNC_PIXEL_SHADING) == 0) {
                st |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            } else {
                st |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            }
        }
        state |= st;
    }

    if ((access & D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT) != 0)
        state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;

    if ((access & D3D12_BARRIER_ACCESS_COPY_SOURCE) != 0)
        state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    if ((access & D3D12_BARRIER_ACCESS_RESOLVE_DEST) != 0)
        state |= D3D12_RESOURCE_STATE_RESOLVE_DEST;
    if ((access & D3D12_BARRIER_ACCESS_RESOLVE_SOURCE) != 0)
        state |= D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    if ((access & (D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ)) != 0)
        state |= D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    if ((access & D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE) != 0)
        state |= D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_DECODE_READ;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ;
    if ((access & D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE) != 0)
        state |= D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE;

    switch (layout) {
        case D3D12_BARRIER_LAYOUT_PRESENT: {
            state |= D3D12_RESOURCE_STATE_PRESENT;
        } break;
        case D3D12_BARRIER_LAYOUT_GENERIC_READ: {
            state |= D3D12_RESOURCE_STATE_GENERIC_READ;
        } break;
        case D3D12_BARRIER_LAYOUT_RENDER_TARGET: {
            state |= D3D12_RESOURCE_STATE_RENDER_TARGET;
        } break;
        case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS: {
            state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        } break;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE: {
            state |= D3D12_RESOURCE_STATE_DEPTH_WRITE;
        } break;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ: {
            state |= D3D12_RESOURCE_STATE_DEPTH_READ;
        } break;
        case D3D12_BARRIER_LAYOUT_COPY_SOURCE: {
            state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
        } break;
        case D3D12_BARRIER_LAYOUT_COPY_DEST: {
            state |= D3D12_RESOURCE_STATE_COPY_DEST;
        } break;
        case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE: {
            state |= D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        } break;
        case D3D12_BARRIER_LAYOUT_RESOLVE_DEST: {
            state |= D3D12_RESOURCE_STATE_RESOLVE_DEST;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ: {
            state |= D3D12_RESOURCE_STATE_VIDEO_DECODE_READ;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE: {
            state |= D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ: {
            state |= D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE: {
            state |= D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ: {
            state |= D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ;
        } break;
        case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE: {
            state |= D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE;
        } break;
    }
    if ((access & D3D12_BARRIER_ACCESS_COPY_DEST) != 0)
        state = D3D12_RESOURCE_STATE_COPY_DEST;
    if ((access & D3D12_BARRIER_ACCESS_STREAM_OUTPUT) != 0)
        state = D3D12_RESOURCE_STATE_STREAM_OUT;
    if ((access & D3D12_BARRIER_ACCESS_RENDER_TARGET) != 0)
        state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    if ((access & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS) != 0)
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if ((access & D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE) != 0)
        state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    switch (listType) {
        case D3D12_COMMAND_LIST_TYPE_COMPUTE: {
            state &= ~D3D12_RESOURCE_STATE_INDEX_BUFFER;
            state &= ~D3D12_RESOURCE_STATE_RENDER_TARGET;
            state &= ~D3D12_RESOURCE_STATE_DEPTH_WRITE;
            state &= ~D3D12_RESOURCE_STATE_DEPTH_READ;
            state &= ~D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            state &= ~D3D12_RESOURCE_STATE_STREAM_OUT;
            state &= ~D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
        } break;
        case D3D12_COMMAND_LIST_TYPE_COPY: {
            state &= D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_COPY_SOURCE;
        } break;
    }
    return state;
}
void EnhancedBarrierTrackerBackup::UpdateResourceState(Resource const *resPtr, ResourceStates &state) {
    state.require_update = false;
    bool is_write = false;
    auto disp = vstd::scope_exit([&]() {
        if (is_write) {
            writeStateMap.emplace(resPtr, state.size);
        } else {
            writeStateMap.remove(resPtr);
        }
    });
    if (state.layer_states.index() == 0) {
        D3D12_RESOURCE_BARRIER barrier{};
        auto &bf = state.layer_states.get<0>();
        auto before_state = bf.first_time ? resPtr->GetInitState() : ToStates(bf.before_sync, bf.before_access, D3D12_BARRIER_LAYOUT_UNDEFINED);
        auto after_state = ToStates(bf.after_sync, bf.after_access, D3D12_BARRIER_LAYOUT_UNDEFINED);
        bool emplace = true;
        auto uav_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE | D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (before_state != after_state) {
            if ((before_state & uav_state) != 0 && (before_state & uav_state) != 0) {
                D3D12_RESOURCE_BARRIER extra_barrier{};
                extra_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                extra_barrier.UAV.pResource = resPtr->GetResource();
                barriers.emplace_back(extra_barrier);
                after_state = D3D12_RESOURCE_STATE_COMMON;
            }
            if ((before_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) == 0 && (after_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) == 0) {
                barrier.Transition.Subresource = UINT32_MAX;
                barrier.Transition.pResource = resPtr->GetResource();
                barrier.Transition.StateBefore = before_state;
                barrier.Transition.StateAfter = after_state;
            } else
                emplace = false;
        } else if ((before_state & uav_state) != 0) {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = resPtr->GetResource();
        } else {
            emplace = false;
        }

        if (emplace)
            barriers.emplace_back(barrier);
        is_write |= (bf.after_access & detail::write_access) != 0;
        bf.before_sync = bf.after_sync;
        bf.after_sync = D3D12_BARRIER_SYNC_NONE;
        bf.before_access = bf.after_access;
        bf.after_access = D3D12_BARRIER_ACCESS_COMMON;
        bf.first_time = false;
    } else {
        auto &vec = state.layer_states.get<1>();
        bool setted = false;
        for (auto idx : vstd::range(vec.size())) {
            auto &i = vec[idx];
            if (!i.level_require_update) continue;
            i.level_require_update = false;
            D3D12_RESOURCE_BARRIER barrier{};
            auto before_state = i.first_time ? resPtr->GetInitState() : ToStates(i.before_sync, i.before_access, i.before_layout);
            auto after_state = ToStates(i.after_sync, i.after_access, i.after_layout);
            bool emplace = true;
            if ((before_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) != 0 || (after_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) != 0 || (before_state == after_state && (before_state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) != 0)) {
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = resPtr->GetResource();
            } else if (before_state != after_state) {
                barrier.Transition.Subresource = idx;
                barrier.Transition.pResource = resPtr->GetResource();
                barrier.Transition.StateBefore = before_state;
                barrier.Transition.StateAfter = after_state;
            } else {
                emplace = false;
            }
            if (emplace)
                barriers.emplace_back(barrier);
            is_write |= (i.after_access & detail::write_access) != 0;
            i.before_sync = i.after_sync;
            i.after_sync = D3D12_BARRIER_SYNC_NONE;
            i.before_access = i.after_access;
            i.after_access = D3D12_BARRIER_ACCESS_COMMON;
            i.before_layout = i.after_layout;
            i.first_time = false;
        }
    }
}

void EnhancedBarrierTrackerBackup::UpdateState(BarrierCallback *cmdBuffer) {
    barriers.clear();
    for (auto &i : current_update_states) {
        UpdateResourceState(i.first, *i.second);
    }
    current_update_states.clear();
    if (!barriers.empty())
        cmdBuffer->ResourceBarrier(barriers.size(), barriers.data());
}
void EnhancedBarrierTrackerBackup::RestoreState(BarrierCallback *cmdBuffer) {
    barriers.clear();
    writeStateMap.clear();
    current_update_states.clear();
    for (auto &i : frameStates) {
        Resource const *resPtr = i.first;
        ResourceStates &state = i.second;
        if (state.layer_states.index() == 0) {
            auto &bf = state.layer_states.get<0>();
            if (bf.before_access == D3D12_BARRIER_ACCESS_COMMON || bf.first_time) continue;
            auto before_state = ToStates(bf.before_sync, bf.before_access, D3D12_BARRIER_LAYOUT_UNDEFINED);
            auto after_state = resPtr->GetInitState();
            if (before_state == after_state) continue;
            if (before_state != after_state) {
                if (0 != (after_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) || 0 != (before_state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
                    continue;
                }
            }
            D3D12_RESOURCE_BARRIER &barrier = barriers.emplace_back();
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resPtr->GetResource();
            barrier.Transition.Subresource = UINT32_MAX;
            barrier.Transition.StateBefore = before_state;
            barrier.Transition.StateAfter = after_state;
        } else {
            auto &vec = state.layer_states.get<1>();
            auto init_state = resPtr->GetInitState();
            for (auto idx : vstd::range(vec.size())) {
                auto &i = vec[idx];
                if (!i.level_inited || i.first_time) continue;
                auto before_state = ToStates(i.before_sync, i.before_access, i.before_layout);
                auto after_state = resPtr->GetInitState();
                if (before_state == after_state) continue;
                D3D12_RESOURCE_BARRIER &barrier = barriers.emplace_back();
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = resPtr->GetResource();
                barrier.Transition.Subresource = idx;
                barrier.Transition.StateBefore = before_state;
                barrier.Transition.StateAfter = after_state;
            }
        }
    }
    frameStates.clear();
    if (!barriers.empty())
        cmdBuffer->ResourceBarrier(barriers.size(), barriers.data());
}
EnhancedBarrierTrackerBackup::EnhancedBarrierTrackerBackup() {}
EnhancedBarrierTrackerBackup::~EnhancedBarrierTrackerBackup() {}
}// namespace lc::dx