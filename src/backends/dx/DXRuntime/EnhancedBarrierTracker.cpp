#include "EnhancedBarrierTracker.h"
#include <Resource/DefaultBuffer.h>
#include <Resource/TextureBase.h>
#include <DXRuntime/CommandBuffer.h>
#include "EnhancedBarrierTrackerImpl.h"
namespace lc::dx {
namespace detail {
constexpr D3D12_BARRIER_SYNC LUISA_DX12_BARRIER_SYNC_INPUT_INDEX = static_cast<D3D12_BARRIER_SYNC>(0x4);// 0x04

static constexpr D3D12_BARRIER_SYNC BarrierSyncMap[] = {
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeRead,
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeAccelRead,
    D3D12_BARRIER_SYNC_COMPUTE_SHADING,                                                                   // ComputeUAV,
    D3D12_BARRIER_SYNC_COPY,                                                                              // CopySource,
    D3D12_BARRIER_SYNC_COPY,                                                                              // CopyDest,
    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,                                           // BuildAccel,
    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,                                           // BuildAccelScratch,
    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,                                            // CopyAccelSrc
    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,                                            // CopyAccelDst
    D3D12_BARRIER_SYNC_DEPTH_STENCIL,                                                                     //DepthRead
    D3D12_BARRIER_SYNC_DEPTH_STENCIL,                                                                     //DepthWrite
    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,                                                                  //IndirectArgs
    D3D12_BARRIER_SYNC_VERTEX_SHADING,                                                                    //VertexRead,
    LUISA_DX12_BARRIER_SYNC_INPUT_INDEX,                                                                  //  IndexRead,
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
    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,                       // BuildAccelScratch,
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
    D3D12_BARRIER_LAYOUT_UNDEFINED,          // BuildAccelScratch,
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
static std::pair<D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT> combine(
    std::pair<D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT> first,
    std::pair<D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT> second,
    bool &dual_write_combined) {
    D3D12_BARRIER_ACCESS access = D3D12_BARRIER_ACCESS_COMMON;
    D3D12_BARRIER_LAYOUT layout = D3D12_BARRIER_LAYOUT_COMMON;
    bool first_is_write = (first.first & detail::write_access) != 0;
    bool second_is_write = (second.first & detail::write_access) != 0;
    dual_write_combined = false;
    if (first_is_write && second_is_write && (first.first != second.first)) {
        access = first.first | second.first;
        dual_write_combined = true;
        // LUISA_ERROR("Shader error, can not be writen in different way in same pass.");
    }
    if (first_is_write) {
        access = first.first;
        layout = first.second;
    } else if (second_is_write) {
        access = second.first;
        layout = second.second;
    } else {
        access = first.first | second.first;
    }
    return {access, layout};
}

static D3D12_BARRIER_LAYOUT filter_layout(D3D12_BARRIER_LAYOUT last_layout, D3D12_BARRIER_ACCESS access) {
    switch (last_layout) {
        case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
            if (access != D3D12_BARRIER_ACCESS_SHADER_RESOURCE) {
                return D3D12_BARRIER_LAYOUT_COMMON;
            }
            break;
        case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
            if (access != D3D12_BARRIER_ACCESS_COPY_SOURCE) {
                return D3D12_BARRIER_LAYOUT_COMMON;
            }
            break;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
            if (access != D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ) {
                return D3D12_BARRIER_LAYOUT_COMMON;
            }
            break;
    }
    return last_layout;
}
}// namespace detail

void EnhancedBarrierTracker::Record(
    Resource const *res,
    Range range,
    Usage resUsage) {
    auto barrier_state_idx = luisa::to_underlying(resUsage);
    Record(
        res,
        range,
        detail::BarrierSyncMap[barrier_state_idx],
        detail::BarrierAccessMap[barrier_state_idx],
        detail::BarrierLayoutMap[barrier_state_idx]);
}
EnhancedBarrierTrackerImpl::ResourceStates::ResourceStates(Type type, size_t size) : size(size) {
    if (type == Type::Texture) {
        layer_states.reset_as<vstd::vector<TextureRange>>(size);
    } else {
        layer_states.reset_as<BufferRange>();
        // switch (res->GetInitState()) {
        //     case D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
        //         // layer_states.get<0>().init_access = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ;
        //         break;
        // }
    }
}
void EnhancedBarrierTracker::Record(
    Resource const *res,
    Range range,
    D3D12_BARRIER_SYNC sync,
    D3D12_BARRIER_ACCESS access,
    D3D12_BARRIER_LAYOUT layout) {
    ResourceStates::Type type;
    switch (res->GetTag()) {
        case Resource::Tag::DefaultBuffer:
        case Resource::Tag::SparseBuffer:
        case Resource::Tag::ExternalBuffer:
            type = ResourceStates::Type::Buffer;
            if (range == Range()) {
                auto b = static_cast<Buffer const *>(res);
                range = Range(0, b->GetByteSize());
            }
            Record(
                BufferView{
                    static_cast<Buffer const *>(res),
                    range.min,
                    range.max - range.min},
                sync, access, layout);
            break;
        case Resource::Tag::RenderTexture:
        case Resource::Tag::SparseTexture:
        case Resource::Tag::DepthBuffer:
        case Resource::Tag::ExternalTexture:
        case Resource::Tag::ExternalDepth:
            type = ResourceStates::Type::Texture;
            if (range == Range()) {
                auto b = static_cast<TextureBase const *>(res);
                range = Range(0, b->Mip());
            }
            for (uint i = range.min; i < range.max; ++i) {
                Record(TexView{
                           static_cast<TextureBase const *>(res),
                           i},
                       sync, access, layout);
            }
            break;
        case Resource::Tag::SwapChain:
            type = ResourceStates::Type::Texture;
            Record(ResourceView(static_cast<SwapChain const *>(res)), sync, access, layout);
            break;
        default:
            LUISA_ERROR("Bad resource for barrier.");
            break;
    }
}
namespace detail {
static void LegacyBarrierToEnhanced(
    D3D12_RESOURCE_STATES state,
    D3D12_BARRIER_SYNC &sync,
    D3D12_BARRIER_ACCESS &access,
    D3D12_BARRIER_LAYOUT &layout) {
    if (state == D3D12_RESOURCE_STATE_COMMON) {
        sync = D3D12_BARRIER_SYNC_ALL;
    }
    if ((state & D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER) != 0) {
        sync |= D3D12_BARRIER_SYNC_VERTEX_SHADING;
        access |= D3D12_BARRIER_ACCESS_VERTEX_BUFFER;
    }
    if ((state & D3D12_RESOURCE_STATE_INDEX_BUFFER) != 0) {
        sync |= D3D12_BARRIER_SYNC_VERTEX_SHADING;
        access |= D3D12_BARRIER_ACCESS_INDEX_BUFFER;
    }
    if ((state & D3D12_RESOURCE_STATE_RENDER_TARGET) != 0) {
        sync |= D3D12_BARRIER_SYNC_PIXEL_SHADING;
        access |= D3D12_BARRIER_ACCESS_RENDER_TARGET;
        layout = D3D12_BARRIER_LAYOUT_RENDER_TARGET;
    }
    if ((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0) {
        sync |= D3D12_BARRIER_SYNC_ALL_SHADING;
        access |= D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
        layout = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
    }
    if ((state & D3D12_RESOURCE_STATE_DEPTH_WRITE) != 0) {
        sync |= D3D12_BARRIER_SYNC_DEPTH_STENCIL;
        access |= D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
        layout = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
    }
    if ((state & D3D12_RESOURCE_STATE_DEPTH_READ) != 0) {
        sync |= D3D12_BARRIER_SYNC_DEPTH_STENCIL;
        access |= D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
        layout = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
    }
    if ((state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) != 0) {
        sync |= D3D12_BARRIER_SYNC_NON_PIXEL_SHADING;
        access |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        layout = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
    }
    if ((state & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) != 0) {
        sync |= D3D12_BARRIER_SYNC_PIXEL_SHADING;
        access |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        layout = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
    }
    if ((state & D3D12_RESOURCE_STATE_STREAM_OUT) != 0) {
        sync = D3D12_BARRIER_SYNC_ALL;
        access |= D3D12_BARRIER_ACCESS_STREAM_OUTPUT;
        layout = D3D12_BARRIER_LAYOUT_COMMON;
    }
    if ((state & D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT) != 0) {
        sync |= D3D12_BARRIER_SYNC_EXECUTE_INDIRECT;
        access |= D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT;
    }
    if ((state & D3D12_RESOURCE_STATE_COPY_DEST) != 0) {
        sync |= D3D12_BARRIER_SYNC_COPY;
        access |= D3D12_BARRIER_ACCESS_COPY_DEST;
        layout = D3D12_BARRIER_LAYOUT_COPY_DEST;
    }
    if ((state & D3D12_RESOURCE_STATE_COPY_SOURCE) != 0) {
        sync |= D3D12_BARRIER_SYNC_COPY;
        access |= D3D12_BARRIER_ACCESS_COPY_SOURCE;
        layout = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
    }
    if ((state & D3D12_RESOURCE_STATE_RESOLVE_DEST) != 0) {
        sync |= D3D12_BARRIER_SYNC_RESOLVE;
        access |= D3D12_BARRIER_ACCESS_RESOLVE_DEST;
        layout = D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
    }
    if ((state & D3D12_RESOURCE_STATE_RESOLVE_SOURCE) != 0) {
        sync |= D3D12_BARRIER_SYNC_RESOLVE;
        access |= D3D12_BARRIER_ACCESS_RESOLVE_SOURCE;
        layout = D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
    }
    if ((state & D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) != 0) {
        sync |= D3D12_BARRIER_SYNC_RAYTRACING | D3D12_BARRIER_SYNC_ALL_SHADING;
        access |= D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ;
    }
    if ((state & D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE) != 0) {
        sync |= D3D12_BARRIER_SYNC_VERTEX_SHADING | D3D12_BARRIER_SYNC_PIXEL_SHADING;
        access |= D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE;
        layout = D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_DECODE_READ) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_DECODE;
        access |= D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_DECODE;
        access |= D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_PROCESS;
        access |= D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_PROCESS;
        access |= D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_ENCODE;
        access |= D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ;
    }
    if ((state & D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE) != 0) {
        sync |= D3D12_BARRIER_SYNC_VIDEO_ENCODE;
        access |= D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;
        layout = D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE;
    }
}
}// namespace detail
void EnhancedBarrierTracker::Record(
    ResourceView const &res,
    D3D12_RESOURCE_STATES state) {
    D3D12_BARRIER_SYNC sync{D3D12_BARRIER_SYNC_NONE};
    D3D12_BARRIER_ACCESS access{D3D12_BARRIER_ACCESS_COMMON};
    D3D12_BARRIER_LAYOUT layout{D3D12_BARRIER_LAYOUT_COMMON};
    detail::LegacyBarrierToEnhanced(
        state,
        sync,
        access,
        layout);
    Record(
        res,
        sync,
        access,
        layout);
}
void EnhancedBarrierTracker::Record(
    Resource const *res,
    Range range,
    D3D12_RESOURCE_STATES state) {

    D3D12_BARRIER_SYNC sync{D3D12_BARRIER_SYNC_NONE};
    D3D12_BARRIER_ACCESS access{D3D12_BARRIER_ACCESS_COMMON};
    D3D12_BARRIER_LAYOUT layout{D3D12_BARRIER_LAYOUT_COMMON};
    detail::LegacyBarrierToEnhanced(
        state,
        sync,
        access,
        layout);
    Record(
        res,
        range,
        sync,
        access,
        layout);
}
void EnhancedBarrierTracker::Record(
    ResourceView const &res,
    Usage resUsage) {
    //TODO
    auto barrier_state_idx = luisa::to_underlying(resUsage);
    Record(
        res,
        detail::BarrierSyncMap[barrier_state_idx],
        detail::BarrierAccessMap[barrier_state_idx],
        detail::BarrierLayoutMap[barrier_state_idx]);
}

void EnhancedBarrierTracker::Record(
    ResourceView const &res,
    D3D12_BARRIER_SYNC sync,
    D3D12_BARRIER_ACCESS access,
    D3D12_BARRIER_LAYOUT layout) {
    using SubResource = vstd::variant<
        BufferAfterRange,
        uint /*tex level*/>;
    ResourceStates::Type type;
    Resource const *d3d12Res;
    size_t size;
    bool allow_simul_access = true;
    D3D12_BARRIER_LAYOUT init_layout = D3D12_BARRIER_LAYOUT_COMMON;

    auto resRange = res.multi_visit_or(
        vstd::UndefEval<SubResource>{},
        [&](BufferView const &bufferView) -> SubResource {
            type = ResourceStates::Type::Buffer;
            d3d12Res = bufferView.buffer;
            size = bufferView.buffer->GetByteSize();
            return BufferAfterRange{
                // Range{bufferView.offset, bufferView.byteSize},
                sync,
                access};
        },
        [&](TexView const &texView) -> SubResource {
            // TODO: set init layout
            type = ResourceStates::Type::Texture;
            size = texView.tex->Mip();
            d3d12Res = texView.tex;
            allow_simul_access = texView.tex->AllowSimulAccess();
            return texView.level;
        },
        [&](SwapChain const *swapchain) -> SubResource {
            init_layout = D3D12_BARRIER_LAYOUT_PRESENT;
            type = ResourceStates::Type::Texture;
            d3d12Res = swapchain;
            size = 1;
            return 0u;
        });
    auto ite = frameStates.emplace(d3d12Res, type, size);
    auto &vec = ite.value().layer_states;
    if (!ite.value().require_update) {
        current_update_states.emplace_back(ite.key(), &ite.value());
    }
    ite.value().require_update = true;

    vec.visit(
        [&]<typename T>(T &vec) {
            if constexpr (std::is_same_v<T, BufferRange>) {
                LUISA_DEBUG_ASSERT(resRange.index() == 0);
                auto &&current_range = resRange.template get<0>();
                BufferRange new_range{
                    D3D12_BARRIER_SYNC_NONE,
                    current_range.sync,
                    D3D12_BARRIER_ACCESS_COMMON,
                    // vec.init_access,
                    current_range.access};
                // if (vec.size() > 64)// Combine all if too many ranges
                // {
                //     BufferRange jumbo_range{
                //         Range(
                //             std::numeric_limits<uint64>::max(),
                //             1ull),
                //         D3D12_BARRIER_SYNC_NONE,
                //         D3D12_BARRIER_SYNC_NONE,
                //         D3D12_BARRIER_ACCESS_COMMON,
                //         D3D12_BARRIER_ACCESS_COMMON};
                //     for (auto &i : vec) {
                //         jumbo_range.range.Combine(i.range);
                //         jumbo_range.before_sync |= i.before_sync;
                //         jumbo_range.after_sync |= i.after_sync;
                //         jumbo_range.before_access |= i.before_access;
                //         jumbo_range.after_access |= i.after_access;
                //     }
                //     vec.clear();
                //     vec.emplace_back(jumbo_range);
                // }

                // for (int i = 0; i < vec.size(); ++i) {
                //     auto &last_range = vec[i];
                //     if (last_range.range.Collide(new_range.range)) {
                //         new_range.range.Combine(last_range.range);
                //         new_range.before_access |= last_range.before_access;
                //         new_range.after_access |= last_range.after_access;
                //         new_range.before_sync |= last_range.before_sync;
                //         new_range.after_sync |= last_range.after_sync;
                //         if (i != vec.size() - 1) {
                //             last_range = vec.back();
                //         }
                //         vec.pop_back();
                //         --i;
                //     }
                // }
                // vec.emplace_back(new_range);
                // vec.before_access |= new_range.before_access;
                // vec.before_sync |= new_range.before_sync;
                bool dual_write_combined;
                auto result = detail::combine(
                    {vec.after_access, D3D12_BARRIER_LAYOUT_COMMON},
                    {new_range.after_access, D3D12_BARRIER_LAYOUT_COMMON},
                    dual_write_combined);
                vec.after_access = result.first;
                if (dual_write_combined) {
                    vec.after_sync = D3D12_BARRIER_SYNC_ALL;
                } else {
                    vec.after_sync |= new_range.after_sync;
                }
            } else {
                LUISA_DEBUG_ASSERT(resRange.index() == 1);
                auto current_level = resRange.template get<1>();
                auto &tex_range = vec[current_level];
                if (!tex_range.level_inited) {
                    tex_range.level_inited = true;
                    tex_range.before_layout = init_layout;
                }
                tex_range.level_require_update = true;
                tex_range.after_sync |= sync;
                // tex_range.after_access |= access;
                if ((access & (D3D12_BARRIER_ACCESS_RENDER_TARGET |
                               D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE)) != 0) {
                    allow_simul_access = false;
                }
                bool dual_write_combined;
                auto result = detail::combine(
                    {tex_range.after_access, tex_range.after_layout},
                    {access, allow_simul_access ? D3D12_BARRIER_LAYOUT_COMMON : layout},
                    dual_write_combined);
                if (dual_write_combined) {
                    tex_range.after_sync = D3D12_BARRIER_SYNC_ALL;
                }
                tex_range.after_access = result.first;
                tex_range.after_layout = result.second;
            }
        });
}
void EnhancedBarrierTrackerImpl::UpdateResourceState(Resource const *resPtr, ResourceStates &state) {
    state.require_update = false;
    bool is_write = false;
    if (auto bf = state.layer_states.try_get<BufferRange>()) {
        D3D12_BUFFER_BARRIER &barrier = bufferBarriers.emplace_back();
        barrier.SyncBefore = bf->before_sync;
        barrier.SyncAfter = bf->after_sync;
        barrier.AccessBefore = bf->before_access;
        barrier.AccessAfter = bf->after_access;
        barrier.pResource = resPtr->GetResource();
        barrier.Offset = 0;
        barrier.Size = UINT64_MAX;
        is_write |= (barrier.AccessAfter & detail::write_access) != 0;

        bf->before_sync = bf->after_sync;
        bf->after_sync = D3D12_BARRIER_SYNC_NONE;
        bf->before_access = bf->after_access;
        bf->after_access = D3D12_BARRIER_ACCESS_COMMON;
        // bf.after_access = bf.init_access;
    } else {// Texture
        auto &vec = state.layer_states.get<1>();
        for (auto idx : vstd::range(vec.size())) {
            auto &i = vec[idx];
            if (!i.level_require_update) continue;
            i.level_require_update = false;
            D3D12_TEXTURE_BARRIER &barrier = texBarriers.emplace_back();
            i.after_layout = detail::filter_layout(i.after_layout, i.after_access);
            barrier.SyncBefore = i.before_sync;
            barrier.SyncAfter = i.after_sync;
            barrier.AccessBefore = i.before_access;
            barrier.AccessAfter = i.after_access;
            barrier.LayoutBefore = i.before_layout;
            barrier.LayoutAfter = i.after_layout;
            barrier.pResource = resPtr->GetResource();
            barrier.Subresources = D3D12_BARRIER_SUBRESOURCE_RANGE{
                .IndexOrFirstMipLevel = (uint)idx,
                .NumMipLevels = 1,
                .FirstArraySlice = 0,
                .NumArraySlices = 1,
                .FirstPlane = 0,
                .NumPlanes = 1};
            is_write |= (barrier.AccessAfter & detail::write_access) != 0;
            i.before_sync = i.after_sync;
            i.after_sync = D3D12_BARRIER_SYNC_NONE;
            i.before_access = i.after_access;
            i.after_access = D3D12_BARRIER_ACCESS_COMMON;
            i.before_layout = i.after_layout;
        }
    }
    if (is_write) {
        writeStateMap.emplace(resPtr, state.size);
    } else {
        writeStateMap.remove(resPtr);
    }
}

void EnhancedBarrierTrackerImpl::UpdateState(BarrierCallback *cmdBuffer) {
    bufferBarriers.clear();
    texBarriers.clear();
    for (auto &&i : current_update_states) {
        UpdateResourceState(i.first, *i.second);
    }
    current_update_states.clear();
    vstd::fixed_vector<D3D12_BARRIER_GROUP, 2> barriers;
    if (!texBarriers.empty()) {
        for (auto &i : texBarriers) {
            BarrierFilter(i);
        }
        auto &v = barriers.emplace_back();
        v.NumBarriers = texBarriers.size();
        v.Type = D3D12_BARRIER_TYPE_TEXTURE;
        v.pTextureBarriers = texBarriers.data();
    }
    if (!bufferBarriers.empty()) {
        for (auto &i : bufferBarriers) {
            BarrierFilter(i);
        }
        auto &v = barriers.emplace_back();
        v.NumBarriers = bufferBarriers.size();
        v.Type = D3D12_BARRIER_TYPE_BUFFER;
        v.pBufferBarriers = bufferBarriers.data();
    }
    if (!barriers.empty()) {
        cmdBuffer->Barrier(barriers.size(), barriers.data());
    }
}
void EnhancedBarrierTrackerImpl::RestoreState(BarrierCallback *cmdBuffer) {
    current_update_states.clear();
    writeStateMap.clear();
    bufferBarriers.clear();
    texBarriers.clear();
    for (auto &i : frameStates) {
        Resource const *resPtr = i.first;
        ResourceStates &state = i.second;
        if (auto bf = state.layer_states.try_get<BufferRange>()) {
            D3D12_BUFFER_BARRIER &barrier = bufferBarriers.emplace_back();
            barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL;
            barrier.SyncAfter = D3D12_BARRIER_SYNC_NONE;
            barrier.AccessBefore = bf->before_access;
            barrier.AccessAfter = D3D12_BARRIER_ACCESS_COMMON;
            barrier.pResource = resPtr->GetResource();
            barrier.Offset = 0;
            barrier.Size = UINT64_MAX;
        } else {// Texture
            auto &vec = state.layer_states.get<1>();
            auto init_layout = D3D12_BARRIER_LAYOUT_COMMON;
            for (auto idx : vstd::range(vec.size())) {
                auto &i = vec[idx];
                if (!i.level_inited) continue;
                D3D12_TEXTURE_BARRIER &barrier = texBarriers.emplace_back();
                barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL;
                barrier.SyncAfter = D3D12_BARRIER_SYNC_NONE;
                barrier.AccessBefore = i.before_access;
                barrier.AccessAfter = D3D12_BARRIER_ACCESS_COMMON;
                barrier.LayoutBefore = i.before_layout;
                barrier.LayoutAfter = init_layout;
                barrier.pResource = resPtr->GetResource();
                barrier.Subresources = D3D12_BARRIER_SUBRESOURCE_RANGE{
                    .IndexOrFirstMipLevel = (uint)idx,
                    .NumMipLevels = 1,
                    .FirstArraySlice = 0,
                    .NumArraySlices = 1,
                    .FirstPlane = 0,
                    .NumPlanes = 1};
            }
        }
    }
    vstd::fixed_vector<D3D12_BARRIER_GROUP, 2> barriers;
    if (!texBarriers.empty()) {
        for (auto &i : texBarriers) {
            BarrierFilter(i);
        }
        auto &v = barriers.emplace_back();
        v.NumBarriers = texBarriers.size();
        v.Type = D3D12_BARRIER_TYPE_TEXTURE;
        v.pTextureBarriers = texBarriers.data();
    }
    if (!bufferBarriers.empty()) {
        for (auto &i : bufferBarriers) {
            BarrierFilter(i);
        }
        auto &v = barriers.emplace_back();
        v.NumBarriers = bufferBarriers.size();
        v.Type = D3D12_BARRIER_TYPE_BUFFER;
        v.pBufferBarriers = bufferBarriers.data();
    }
    if (!barriers.empty()) {
        cmdBuffer->Barrier(barriers.size(), barriers.data());
    }
    frameStates.clear();
}
namespace detail {
void FilterAccess(
    D3D12_COMMAND_LIST_TYPE type,
    D3D12_BARRIER_SYNC &sync,
    D3D12_BARRIER_ACCESS &access,
    D3D12_BARRIER_LAYOUT &layout) {
    switch (type) {
        case D3D12_COMMAND_LIST_TYPE_COMPUTE: {
            sync &= ~D3D12_BARRIER_SYNC_DRAW;
            sync &= ~D3D12_BARRIER_SYNC_PIXEL_SHADING;
            sync &= ~D3D12_BARRIER_SYNC_DEPTH_STENCIL;
            sync &= ~D3D12_BARRIER_SYNC_RENDER_TARGET;
            access &= ~D3D12_BARRIER_ACCESS_VERTEX_BUFFER;
            access &= ~D3D12_BARRIER_ACCESS_CONSTANT_BUFFER;
            access &= ~D3D12_BARRIER_ACCESS_INDEX_BUFFER;
            access &= ~D3D12_BARRIER_ACCESS_RENDER_TARGET;
            access &= ~D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
            access &= ~D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
            access &= ~D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE;
        } break;
        case D3D12_COMMAND_LIST_TYPE_COPY: {
            sync &= (D3D12_BARRIER_SYNC_ALL | D3D12_BARRIER_SYNC_COPY | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE);
            access &= (D3D12_BARRIER_ACCESS_COPY_DEST | D3D12_BARRIER_ACCESS_COPY_SOURCE | D3D12_BARRIER_ACCESS_RESOLVE_DEST | D3D12_BARRIER_ACCESS_RESOLVE_SOURCE);
            layout = D3D12_BARRIER_LAYOUT_COMMON;
        } break;
    }
    // Type is render-target
    const auto tex_read_sync = D3D12_BARRIER_SYNC_INDEX_INPUT | D3D12_BARRIER_SYNC_VERTEX_SHADING | D3D12_BARRIER_SYNC_PIXEL_SHADING | D3D12_BARRIER_SYNC_COMPUTE_SHADING | D3D12_BARRIER_SYNC_NON_PIXEL_SHADING | D3D12_BARRIER_SYNC_ALL_SHADING;
    if ((access & (D3D12_BARRIER_ACCESS_RENDER_TARGET | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE)) != 0) {
        sync &= ~tex_read_sync;
    }
}
}// namespace detail
void EnhancedBarrierTrackerImpl::BarrierFilter(D3D12_BUFFER_BARRIER &barrier) {
    if (barrier.AccessBefore == D3D12_BARRIER_ACCESS_COMMON && barrier.SyncBefore == D3D12_BARRIER_SYNC_NONE) {
        barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL;
    }
    if (barrier.AccessAfter == D3D12_BARRIER_ACCESS_COMMON && barrier.SyncAfter == D3D12_BARRIER_SYNC_NONE) {
        barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
    }
    D3D12_BARRIER_LAYOUT layout;
    detail::FilterAccess(listType, barrier.SyncBefore, barrier.AccessBefore, layout);
    detail::FilterAccess(listType, barrier.SyncAfter, barrier.AccessAfter, layout);
}
void EnhancedBarrierTrackerImpl::BarrierFilter(D3D12_TEXTURE_BARRIER &barrier) {
    if (barrier.AccessBefore == D3D12_BARRIER_ACCESS_COMMON && barrier.SyncBefore == D3D12_BARRIER_SYNC_NONE) {
        barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL;
    }
    if (barrier.AccessAfter == D3D12_BARRIER_ACCESS_COMMON && barrier.SyncAfter == D3D12_BARRIER_SYNC_NONE) {
        barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
    }
    detail::FilterAccess(listType, barrier.SyncBefore, barrier.AccessBefore, barrier.LayoutBefore);
    detail::FilterAccess(listType, barrier.SyncAfter, barrier.AccessAfter, barrier.LayoutAfter);
}
EnhancedBarrierTracker::EnhancedBarrierTracker() = default;
EnhancedBarrierTracker::~EnhancedBarrierTracker() = default;
EnhancedBarrierTrackerImpl::EnhancedBarrierTrackerImpl() = default;
EnhancedBarrierTrackerImpl::~EnhancedBarrierTrackerImpl() = default;
}// namespace lc::dx