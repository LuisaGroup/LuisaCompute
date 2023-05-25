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
    struct ResourceUsage {
        UsedResource::ResourceHandle resource;
        D3D12_RESOURCE_STATES required_state;
        template<typename Arg>
            requires(std::is_constructible_v<UsedResource::ResourceHandle, Arg &&>)
        ResourceUsage(
            Arg &&resource,
            D3D12_RESOURCE_STATES required_state)
            : resource{std::forward<Arg>(resource)},
              required_state{required_state} {}
    };

private:
    friend class lc::dx::LCCmdBuffer;
    friend class lc::dx::LCPreProcessVisitor;
    friend class lc::dx::LCCmdVisitor;
    virtual void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory4 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept = 0;

protected:
    luisa::vector<ResourceUsage> resource_usages;

public:
    virtual ~DXCustomCmd() noexcept = default;
    size_t used_resources_size() const noexcept override {
        return resource_usages.size();
    }
    UsedResource used_resource(size_t index) const noexcept override {
        auto &&v = resource_usages[index];
        Usage resource_usage;
        switch (v.required_state) {
            case D3D12_RESOURCE_STATE_COMMON:
            case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
                resource_usage = Usage::READ_WRITE;
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
                resource_usage = Usage::READ;
                break;
            default:
                resource_usage = Usage::WRITE;
                break;
        }
        return UsedResource{
            v.resource,
            resource_usage};
    }
    DXCustomCmd() noexcept = default;
};
}// namespace luisa::compute