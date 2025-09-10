#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/core/stl/functional.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/logging.h>

namespace lc::dx {
class LCCmdBuffer;
class LCPreProcessVisitor;
class LCCmdVisitor;
}// namespace lc::dx

namespace luisa::compute {

namespace dx_detail {

static constexpr D3D12_BARRIER_SYNC BarrierSyncMap[] = {
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeRead,
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeAccelRead,
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeUAV,
    D3D12_BARRIER_SYNC_COPY,                                                                              // CopySource,
    D3D12_BARRIER_SYNC_COPY,                                                                              // CopyDest,
    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,                                           // BuildAccel,
    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,                                            // CopyAccelSrc
    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,                                            // CopyAccelDst
    D3D12_BARRIER_SYNC_DEPTH_STENCIL,                                                                     //DepthRead
    D3D12_BARRIER_SYNC_DEPTH_STENCIL,                                                                     //DepthWrite
    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,                                                                  //IndirectArgs
    D3D12_BARRIER_SYNC_VERTEX_SHADING,                                                                    //VertexRead,
    static_cast<D3D12_BARRIER_SYNC>(0x4),                                                                 //  IndexRead,
    D3D12_BARRIER_SYNC_RENDER_TARGET,                                                                     //  RenderTarget
    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,                                           // AccelInstanceBuffer
    static_cast<D3D12_BARRIER_SYNC>(D3D12_BARRIER_SYNC_PIXEL_SHADING | D3D12_BARRIER_SYNC_VERTEX_SHADING),// RasterRead
    static_cast<D3D12_BARRIER_SYNC>(D3D12_BARRIER_SYNC_PIXEL_SHADING | D3D12_BARRIER_SYNC_VERTEX_SHADING),//RasterAccelRead
    static_cast<D3D12_BARRIER_SYNC>(D3D12_BARRIER_SYNC_PIXEL_SHADING | D3D12_BARRIER_SYNC_VERTEX_SHADING),//RasterUAV
    D3D12_BARRIER_SYNC_VIDEO_ENCODE,                                                                      //VideoEncodeRead,
    D3D12_BARRIER_SYNC_VIDEO_ENCODE,                                                                      //VideoEncodeWrite,
    D3D12_BARRIER_SYNC_VIDEO_PROCESS,                                                                     //VideoProcessRead,
    D3D12_BARRIER_SYNC_VIDEO_PROCESS,                                                                     //VideoProcessWrite,
    D3D12_BARRIER_SYNC_VIDEO_DECODE,                                                                      //VideoDecodeRead,
    D3D12_BARRIER_SYNC_VIDEO_DECODE,                                                                      //VideoDecodeWrite,
};

static constexpr D3D12_BARRIER_ACCESS BarrierAccessMap[] = {
    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,                        // ComputeRead,
    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ, // ComputeAccelRead,
    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,                       // ComputeUAV,
    D3D12_BARRIER_ACCESS_COPY_SOURCE,                            // CopySource,
    D3D12_BARRIER_ACCESS_COPY_DEST,                              // CopyDest,
    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,// BuildAccel,
    D3D12_BARRIER_ACCESS_COPY_SOURCE,                            // CopyAccelSrc
    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,// CopyAccelDst
    D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ,                     //DepthRead
    D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,                    //DepthWrite
    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,                      // IndirectArgs
    D3D12_BARRIER_ACCESS_VERTEX_BUFFER,                          //VertexRead,
    D3D12_BARRIER_ACCESS_INDEX_BUFFER,                           //  IndexRead,
    D3D12_BARRIER_ACCESS_RENDER_TARGET,                          //RenderTarget
    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,                        //AccelInstanceBuffer
    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,                        // RasterRead
    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ, // RasterAccelRead,
    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,                       // RasterUAV,
    D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ,                      //VideoEncodeRead,
    D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE,                     //VideoEncodeWrite,
    D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ,                     //VideoProcessRead,
    D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE,                    //VideoProcessWrite,
    D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ,                      //VideoDecodeRead,
    D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE,                     //VideoDecodeWrite,
};

static constexpr D3D12_BARRIER_LAYOUT BarrierLayoutMap[] = {
    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,    // ComputeRead,
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // ComputeAccelRead,
    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,   // ComputeUAV,
    D3D12_BARRIER_LAYOUT_COPY_SOURCE,        // CopySource,
    D3D12_BARRIER_LAYOUT_COPY_DEST,          // CopyDest,
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // BuildAccel,
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // CopyAccelSrc
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // CopyAccelDst
    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, //DepthRead
    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,//DepthWrite
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // DepthWrite
    D3D12_BARRIER_LAYOUT_UNDEFINED,          //VertexRead,
    D3D12_BARRIER_LAYOUT_UNDEFINED,          //  IndexRead,
    D3D12_BARRIER_LAYOUT_RENDER_TARGET,      //RenderTarget
    D3D12_BARRIER_LAYOUT_UNDEFINED,          //AccelInstanceBuffer
    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,    // RasterRead
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // RasterAccelRead,
    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,   // RasterUAV,
    D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ,  //VideoEncodeRead,
    D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE, //VideoEncodeWrite,
    D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ, //VideoProcessRead,
    D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE,//VideoProcessWrite,
    D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ,  //VideoDecodeRead,
    D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE, //VideoDecodeWrite,
};
}// namespace dx_detail
class DXCustomCmd : public CustomDispatchCommand {
public:
    using ResourceHandle = luisa::variant<
        Argument::Buffer,
        Argument::Texture,
        Argument::BindlessArray>;
    struct ResourceUsage {
        ResourceHandle resource;
        D3D12_RESOURCE_STATES required_state;
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        ResourceUsage(
            Arg &&resource,
            D3D12_RESOURCE_STATES required_state)
            : resource{std::forward<Arg>(resource)},
              required_state{required_state} {}
    };
    enum class EnhancedResourceUsageType : uint {
        ComputeRead,
        ComputeAccelRead,
        ComputeUAV,
        CopySource,
        CopyDest,
        BuildAccel,
        CopyAccelSrc,
        CopyAccelDst,
        DepthRead,
        DepthWrite,
        IndirectArgs,
        VertexRead,
        IndexRead,
        RenderTarget,
        AccelInstanceBuffer,
        RasterRead,
        RasterAccelRead,
        RasterUAV,
        VideoEncodeRead,
        VideoEncodeWrite,
        VideoProcessRead,
        VideoProcessWrite,
        VideoDecodeRead,
        VideoDecodeWrite,
    };
    struct EnhancedResourceUsage {
        ResourceHandle resource;
        D3D12_BARRIER_SYNC sync;
        D3D12_BARRIER_ACCESS access;
        D3D12_BARRIER_LAYOUT texture_layout;// only used for texture_layout
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        EnhancedResourceUsage(
            Arg &&resource,
            D3D12_BARRIER_SYNC sync,
            D3D12_BARRIER_ACCESS access,
            D3D12_BARRIER_LAYOUT texture_layout = D3D12_BARRIER_LAYOUT_UNDEFINED)
            : resource{std::forward<Arg>(resource)},
              sync{sync},
              access{access},
              texture_layout{texture_layout} {
            LUISA_ASSERT((this->resource.index() == 1) == (texture_layout != D3D12_BARRIER_LAYOUT_UNDEFINED), "Buffer must not have valid layout and texture must have a defined layout");
        }
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        EnhancedResourceUsage(
            Arg &&resource,
            EnhancedResourceUsageType type)
            : resource{std::forward<Arg>(resource)},
              sync(dx_detail::BarrierSyncMap[luisa::to_underlying(type)]),
              access(dx_detail::BarrierAccessMap[luisa::to_underlying(type)]),
              texture_layout(dx_detail::BarrierLayoutMap[luisa::to_underlying(type)]) {}
    };

private:
    friend class lc::dx::LCCmdBuffer;
    friend class lc::dx::LCPreProcessVisitor;
    friend class lc::dx::LCCmdVisitor;
    virtual void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept = 0;
    [[nodiscard]] virtual luisa::span<ResourceUsage> get_resource_usages() noexcept {
        return {};
    }
    [[nodiscard]] virtual luisa::span<EnhancedResourceUsage> get_enhanced_resource_usages() noexcept {
        return {};
    }

    [[nodiscard]] static auto resource_state_to_usage(D3D12_RESOURCE_STATES state) noexcept {
        constexpr auto read_state =
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER |
            D3D12_RESOURCE_STATE_INDEX_BUFFER |
            D3D12_RESOURCE_STATE_DEPTH_READ |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_STREAM_OUT |
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT |
            D3D12_RESOURCE_STATE_COPY_SOURCE |
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE |
            D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE |
            D3D12_RESOURCE_STATE_VIDEO_DECODE_READ |
            D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ |
            D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        constexpr auto write_state =
            D3D12_RESOURCE_STATE_RENDER_TARGET |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_DEPTH_WRITE |
            D3D12_RESOURCE_STATE_STREAM_OUT |
            D3D12_RESOURCE_STATE_COPY_DEST |
            D3D12_RESOURCE_STATE_RESOLVE_DEST |
            D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE |
            D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE |
            D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        if (state == 0) return Usage::READ_WRITE;
        Usage usage = Usage::NONE;
        if ((state & read_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::READ));
        }
        if ((state & write_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::WRITE));
        }
        return usage;
    }
    [[nodiscard]] static auto resource_state_to_usage(
        D3D12_BARRIER_ACCESS state) noexcept {
        constexpr auto read_state =
            D3D12_BARRIER_ACCESS_VERTEX_BUFFER |
            D3D12_BARRIER_ACCESS_CONSTANT_BUFFER |
            D3D12_BARRIER_ACCESS_INDEX_BUFFER |
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ |
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE |
            D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT |
            D3D12_BARRIER_ACCESS_COPY_SOURCE |
            D3D12_BARRIER_ACCESS_RESOLVE_SOURCE |
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ |
            D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE |
            D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ |
            D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ |
            D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ |
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

        constexpr auto write_state =
            D3D12_BARRIER_ACCESS_RENDER_TARGET |
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS |
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE |
            D3D12_BARRIER_ACCESS_STREAM_OUTPUT |
            D3D12_BARRIER_ACCESS_COPY_DEST |
            D3D12_BARRIER_ACCESS_RESOLVE_DEST |
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;
        if (state == 0) {
            return Usage::READ_WRITE;
        }
        Usage usage = Usage::NONE;
        if ((state & read_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::READ));
        }
        if ((state & write_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::WRITE));
        }
        return usage;
    }

public:
    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        auto usages = const_cast<DXCustomCmd *>(this)->get_resource_usages();
        for (auto &&[handle, state] : usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(state));
            },
                         handle);
        }
        auto enhanced_usages = const_cast<DXCustomCmd *>(this)->get_enhanced_resource_usages();
        for (auto &&[handle, sync, access, layout] : enhanced_usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(access));
            },
                         handle);
        }
    }
    void traverse_arguments(MutableArgumentVisitor &visitor) noexcept override {
        auto usages = get_resource_usages();
        for (auto &&[handle, state] : usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(state));
            },
                         handle);
        }
        auto enhanced_usages = get_enhanced_resource_usages();
        for (auto &&[handle, sync, access, layout] : enhanced_usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(access));
            },
                         handle);
        }
    }
    DXCustomCmd() noexcept = default;
    virtual ~DXCustomCmd() noexcept override = default;
    [[nodiscard]] uint64_t uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::CUSTOM_DISPATCH);
    }
};

}// namespace luisa::compute
