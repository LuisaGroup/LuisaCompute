#pragma once
#include <runtime/rhi/command.h>
#include <core/stl/functional.h>
#include <d3d12.h>
#include <dxgi1_4.h>
namespace lc::dx {
class LCCmdBuffer;
class LCPreProcessVisitor;
class LCCmdVisitor;
}// namespace lc::dx
namespace luisa::compute {
class DXCustomCmd : public CustomDispatchCommand {
public:
    struct ResourceState {
        UsedResource::ResourceHandle resource;
        D3D12_RESOURCE_STATES required_state;
    };

    using Callback = luisa::move_only_function<void(
        IDXGIAdapter1 *adapter,
        IDXGIFactory4 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list)>;

private:
    friend class lc::dx::LCCmdBuffer;
    friend class lc::dx::LCPreProcessVisitor;
    friend class lc::dx::LCCmdVisitor;
    struct InternalResourceState {
        uint64_t resource_handle;
        D3D12_RESOURCE_STATES required_state;
    };
    luisa::vector<InternalResourceState> _resource_states;
    Callback _callback;

public:
    DXCustomCmd(StreamTag stream_tag, luisa::span<const ResourceState> resource_states, Callback &&callback) noexcept
        : CustomDispatchCommand{stream_tag}, _callback{std::move(callback)} {
        _resource_states.push_back_uninitialized(resource_states.size());
        _used_resources.push_back_uninitialized(resource_states.size());
        for (size_t i = 0; i < resource_states.size(); ++i) {
            auto &v = resource_states[i];
            auto &dst = _resource_states[i];
            _used_resources[i].resource = v.resource;
            luisa::visit(
                [&](auto &&t) {
                    dst.resource_handle = t.handle;
                    dst.required_state = v.required_state;
                },
                v.resource);
            auto &res_base = _used_resources[i];
            res_base.resource = v.resource;
            switch (v.required_state) {
                case D3D12_RESOURCE_STATE_COMMON:
                case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
                    res_base.resource_usage = Usage::READ_WRITE;
                    break;
                case D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:
                case D3D12_RESOURCE_STATE_INDEX_BUFFER:
                case D3D12_RESOURCE_STATE_DEPTH_READ:
                case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
                case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
                case D3D12_RESOURCE_STATE_STREAM_OUT:
                case D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT:
                case D3D12_RESOURCE_STATE_COPY_SOURCE:
                case D3D12_RESOURCE_STATE_RESOLVE_SOURCE:
                case D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE:
                case D3D12_RESOURCE_STATE_GENERIC_READ:
                case D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE:
                case D3D12_RESOURCE_STATE_VIDEO_DECODE_READ:
                case D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ:
                case D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ:
                    res_base.resource_usage = Usage::READ;
                    break;
                default:
                    res_base.resource_usage = Usage::WRITE;
                    break;
            }
        }
    }
};
}// namespace luisa::compute