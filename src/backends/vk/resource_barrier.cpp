#include <luisa/core/logging.h>

#include "resource_barrier.h"
#include "bindless_array.h"

namespace lc::vk {
namespace detail {

static constexpr auto kRasterStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
static constexpr VkPipelineStageFlagBits2 kBarrierSyncMap[] = {
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // kComputeRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // kComputeAccelRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // kComputeUAV,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // kCopySource,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // kCopyDest,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // kBuildAccel,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // kCopyAccelSrc
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // kCopyAccelDst
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//kDepthRead
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//kDepthWrite
    VK_PIPELINE_STAGE_2_CLEAR_BIT,                                                             //kDepthClear
    VK_PIPELINE_STAGE_2_CLEAR_BIT,                                                             //kRenderTargetClear
    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,                                                     //IndirectArgs
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,                                            //kVertexRead,
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,                                                       //  kIndexRead,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,                                           //  RenderTarget
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // AccelInstanceBuffer
    kRasterStage,                                                                              // kRasterRead
    kRasterStage,                                                                              //RasterAccelRead
    kRasterStage                                                                               //kRasterUAV
};
static constexpr VkAccessFlagBits2 kBarrierAccessMap[] = {
    VK_ACCESS_2_SHADER_READ_BIT,                                               // kComputeRead,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,                           // kComputeAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,// kComputeUAV,
    VK_ACCESS_2_TRANSFER_READ_BIT,                                             // kCopySource,
    VK_ACCESS_2_TRANSFER_WRITE_BIT,                                            // kCopyDest,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,                          // kBuildAccel,
    VK_ACCESS_2_TRANSFER_READ_BIT,                                             // kCopyAccelSrc
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,                          // kCopyAccelDst
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,                             //kDepthRead
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,                            //kDepthWrite
    VK_ACCESS_2_TRANSFER_WRITE_BIT,                                            //kDepthClear
    VK_ACCESS_2_TRANSFER_WRITE_BIT,                                            //kRenderTargetClear
    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,                                     // kIndirectArgs
    VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,                                     //kVertexRead,
    VK_ACCESS_2_INDEX_READ_BIT,                                                //  kIndexRead,
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,                                    //kRenderTarget
    VK_ACCESS_2_SHADER_READ_BIT,                                               //kAccelInstanceBuffer
    VK_ACCESS_2_SHADER_READ_BIT,                                               // kRasterRead
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,                           // kRasterAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,// kRasterUAV,
};
static constexpr VkAccessFlagBits2 kWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
static constexpr VkImageLayout kBarrierLayoutMap[] = {
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // kComputeRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // kComputeAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // kComputeUAV,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,            // kCopySource,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,            // kCopyDest,
    VK_IMAGE_LAYOUT_GENERAL,                         // kBuildAccel,
    VK_IMAGE_LAYOUT_GENERAL,                         // kCopyAccelSrc
    VK_IMAGE_LAYOUT_GENERAL,                         // kCopyAccelDst
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, //kDepthRead
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,//kDepthWrite
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,            //kDepthClear
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,            //kRenderTargetClear
    VK_IMAGE_LAYOUT_GENERAL,                         // kIndirectArgs
    VK_IMAGE_LAYOUT_GENERAL,                         //kVertexRead,
    VK_IMAGE_LAYOUT_GENERAL,                         //  kIndexRead,
    VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,              //kRenderTarget
    VK_IMAGE_LAYOUT_GENERAL,                         //kAccelInstanceBuffer
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // kRasterRead
    VK_IMAGE_LAYOUT_GENERAL,                         // kRasterAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // kRasterUAV,
};
static std::pair<VkAccessFlagBits2, VkImageLayout> combine(
    std::pair<VkAccessFlagBits2, VkImageLayout> first,
    std::pair<VkAccessFlagBits2, VkImageLayout> second) {
    VkAccessFlagBits2 access = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    bool first_is_write = (first.first & detail::kWriteAccess) != 0;
    bool second_is_write = (second.first & detail::kWriteAccess) != 0;
    if (first_is_write && second_is_write && (first.first != second.first)) {
        access = first.first | second.first;
        // LUISA_ERROR("Shader error, can not be writen in different way in same pass.");
    } else if (first_is_write) {
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

static VkImageLayout filter_layout(VkImageLayout last_layout, VkAccessFlagBits2 access) {
    switch (last_layout) {
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if (access != VK_ACCESS_2_SHADER_READ_BIT) {
                return VK_IMAGE_LAYOUT_GENERAL;
            }
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            if (access != VK_ACCESS_2_TRANSFER_READ_BIT) {
                return VK_IMAGE_LAYOUT_GENERAL;
            }
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            if (access != VK_ACCESS_2_TRANSFER_WRITE_BIT) {
                return VK_IMAGE_LAYOUT_GENERAL;
            }
            break;
        default: break;
    }
    return last_layout;
}
}// namespace detail
ResourceBarrier::ResourceBarrier() {}
void ResourceBarrier::record(
    ResourceView const &res,
    Usage usage) {
    record(
        res,
        detail::kBarrierSyncMap[luisa::to_underlying(usage)],
        detail::kBarrierAccessMap[luisa::to_underlying(usage)],
        detail::kBarrierLayoutMap[luisa::to_underlying(usage)]);
}
void ResourceBarrier::set_res(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout) {
    if (res.is_type_of<BufferView>()) {
        // If the buffer is host-visible, should not be recorded by resource-barrier
        if (res.get<0>().buffer->flush_host())
            return;
    }
    using SubResource = vstd::variant<
        BufferAfterRange,
        uint /*tex level*/>;
    ResourceStates::Type type;
    bool allow_simul_access = true;
    Resource const *vk_res;
    size_t size;
    VkImageLayout init_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    auto res_range = res.multi_visit_or(
        vstd::UndefEval<SubResource>{},
        [&](BufferView const &buffer_view) -> SubResource {
            type = ResourceStates::Type::kBuffer;
            vk_res = buffer_view.buffer;
            size = buffer_view.buffer->byte_size();
            return BufferAfterRange{
                // Range{buffer_view.offset, buffer_view.byteSize},
                stage,
                access};
        },
        [&](TexView const &tex_view) -> SubResource {
            // TODO: set init layout
            type = ResourceStates::Type::kTexture;
            size = tex_view.tex->mip();
            vk_res = tex_view.tex;
            init_layout = static_cast<Texture const *>(vk_res)->layout(tex_view.level);
            allow_simul_access = tex_view.tex->simultaneous_access();
            return tex_view.level;
        });
    auto ite = _frame_states.emplace(vk_res, type, size);
    auto &vec = ite.value().layer_states;

    vec.visit(
        [&]<typename T>(T &vec) {
            if constexpr (std::is_same_v<T, BufferRange>) {
                LUISA_DEBUG_ASSERT(res_range.index() == 0);
                auto &&current_range = res_range.template get<0>();
                vec = BufferRange{
                    current_range.stage,
                    VK_PIPELINE_STAGE_2_NONE,
                    current_range.access,
                    VK_ACCESS_2_NONE
                    // vec.init_access,
                };
            } else {
                LUISA_DEBUG_ASSERT(res_range.index() == 1);
                auto current_level = res_range.template get<1>();
                auto &tex_range = vec[current_level];
                tex_range.before_stage = stage;
                tex_range.before_access = access;
                tex_range.before_layout = layout;
            }
        });
}
void ResourceBarrier::record(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout) {
    if (res.is_type_of<BufferView>()) {
        // If the buffer is host-visible, should not be recorded by resource-barrier
        if (res.get<0>().buffer->flush_host())
            return;
    }
    using SubResource = vstd::variant<
        BufferAfterRange,
        uint /*tex level*/>;
    ResourceStates::Type type;
    bool allow_simul_access = true;
    Resource const *vk_res;
    size_t size;
    VkImageLayout init_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    auto res_range = res.multi_visit_or(
        vstd::UndefEval<SubResource>{},
        [&](BufferView const &buffer_view) -> SubResource {
            type = ResourceStates::Type::kBuffer;
            vk_res = buffer_view.buffer;
            size = buffer_view.buffer->byte_size();
            return BufferAfterRange{
                // Range{buffer_view.offset, buffer_view.byteSize},
                stage,
                access};
        },
        [&](TexView const &tex_view) -> SubResource {
            // TODO: set init layout
            type = ResourceStates::Type::kTexture;
            size = tex_view.tex->mip();
            vk_res = tex_view.tex;
            init_layout = static_cast<Texture const *>(vk_res)->layout(tex_view.level);
            allow_simul_access = tex_view.tex->simultaneous_access();
            return tex_view.level;
        });
    auto ite = _frame_states.emplace(vk_res, type, size);
    auto &vec = ite.value().layer_states;
    if (!ite.value().require_update) {
        _current_update_states.emplace_back(ite.key(), &ite.value());
    }
    ite.value().require_update = true;

    vec.visit(
        [&]<typename T>(T &vec) {
            if constexpr (std::is_same_v<T, BufferRange>) {
                LUISA_DEBUG_ASSERT(res_range.index() == 0);
                auto &&current_range = res_range.template get<0>();
                BufferRange new_range{
                    VK_PIPELINE_STAGE_2_NONE,
                    current_range.stage,
                    VK_ACCESS_2_NONE,
                    // vec.init_access,
                    current_range.access};
                auto result = detail::combine(
                    {vec.after_access, VK_IMAGE_LAYOUT_GENERAL},
                    {new_range.after_access, VK_IMAGE_LAYOUT_GENERAL});
                vec.after_access = result.first;
                vec.after_stage |= new_range.after_stage;
            } else {
                LUISA_DEBUG_ASSERT(res_range.index() == 1);
                auto current_level = res_range.template get<1>();
                auto &tex_range = vec[current_level];
                if (!tex_range.level_inited) {
                    tex_range.level_inited = true;
                    tex_range.before_layout = init_layout;
                }
                tex_range.level_require_update = true;
                tex_range.after_stage |= stage;
                // tex_range.after_access |= access;
                if ((access & (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
                    allow_simul_access = false;
                }
                auto result = detail::combine(
                    {tex_range.after_access, tex_range.after_layout},
                    {access, allow_simul_access ? VK_IMAGE_LAYOUT_GENERAL : layout});
                tex_range.after_access = result.first;
                tex_range.after_layout = result.second;
            }
        });
}

void ResourceBarrier::force_refresh_layout(
    Resource const *res, uint level,
    VkImageLayout before_layout) {
    auto iter = _frame_states.find(res);
    if (!(iter && iter.value().layer_states.index() == 1)) return;
    auto &v = iter.value();
    auto &ranges = v.layer_states.get<1>();
    LUISA_ASSERT(ranges.size() > level);
    ranges[level].before_layout = before_layout;
}

void ResourceBarrier::remove_resource(Resource const *res) noexcept {
    _frame_states.remove(res);
    _write_state_map.remove(res);
    saved_restore_states.remove(res);
    for (auto it = _current_update_states.begin(); it != _current_update_states.end();) {
        if (it->first == res) {
            it = _current_update_states.erase(it);
        } else {
            ++it;
        }
    }
}

ResourceBarrier::~ResourceBarrier() {
}

void ResourceBarrier::_update_state(Resource const *res_ptr, ResourceStates &state) {
    state.require_update = false;
    bool is_write = false;
    if (state.layer_states.index() == 0) {
        auto &bf = state.layer_states.get<0>();
        auto &barrier = _buffer_barriers.emplace_back();
        barrier.srcStageMask = bf.before_stage;
        barrier.dstStageMask = bf.after_stage;
        barrier.srcAccessMask = bf.before_access;
        barrier.dstAccessMask = bf.after_access;
        barrier.buffer = static_cast<Buffer const *>(res_ptr)->vk_buffer();
        barrier.offset = 0;
        barrier.size = std::numeric_limits<uint64_t>::max();
        is_write |= (barrier.dstAccessMask & detail::kWriteAccess) != 0;

        bf.before_stage = bf.after_stage;
        bf.after_stage = VK_PIPELINE_STAGE_2_NONE;
        bf.before_access = bf.after_access;
        bf.after_access = VK_ACCESS_2_NONE;
        // bf.after_access = bf.init_access;
    } else {// Texture
        auto &vec = state.layer_states.get<1>();
        auto tex = static_cast<Texture const *>(res_ptr);
        for (auto idx : vstd::range((int64_t)vec.size())) {
            auto &i = vec[idx];
            if (!i.level_require_update) continue;
            i.level_require_update = false;
            auto &barrier = _tex_barriers.emplace_back();
            i.after_layout = detail::filter_layout(i.after_layout, i.after_access);
            barrier.srcStageMask = i.before_stage;
            barrier.dstStageMask = i.after_stage;
            barrier.srcAccessMask = i.before_access;
            barrier.dstAccessMask = i.after_access;
            barrier.oldLayout = i.before_layout;
            barrier.newLayout = i.after_layout;
            barrier.image = tex->vk_image();
            barrier.subresourceRange = VkImageSubresourceRange{
                .aspectMask = tex->get_aspect(),
                .baseMipLevel = (uint)idx,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1};
            is_write |= (barrier.dstAccessMask & detail::kWriteAccess) != 0;
            i.before_stage = i.after_stage;
            i.after_stage = VK_PIPELINE_STAGE_2_NONE;
            i.before_access = i.after_access;
            i.after_access = VK_ACCESS_2_NONE;
            i.before_layout = i.after_layout;
        }
    }
    if (is_write) {
        _write_state_map.emplace(res_ptr, state.size);
    } else {
        _write_state_map.remove(res_ptr);
    }
}
namespace detail {
void filter_access(
    ResourceBarrier::QueueType type,
    VkPipelineStageFlagBits2 &sync,
    VkAccessFlagBits2 &access,
    VkImageLayout &layout) {
    switch (type) {
        case ResourceBarrier::QueueType::kCompute: {
            sync &= (VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_COPY_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_HOST_BIT);
            switch (layout) {
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                    layout = VK_IMAGE_LAYOUT_GENERAL;
                    break;
                default: break;
            }
        } break;
        case ResourceBarrier::QueueType::kCopy: {
            sync &= (VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_COPY_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_HOST_BIT);
            switch (layout) {
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                    layout = VK_IMAGE_LAYOUT_GENERAL;
                    break;
                default: break;
            }
        } break;
        default: break;
    }
    const auto tex_read_sync = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | kRasterStage;
    if ((access & (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
        sync &= ~tex_read_sync;
    }
}
}// namespace detail
VkImageLayout ResourceBarrier::get_layout(Resource const *res, uint level) const {
    auto iter = _frame_states.find(res);
    LUISA_ASSERT(iter && iter.value().layer_states.index() == 1);
    auto &v = iter.value();
    auto &ranges = v.layer_states.get<1>();
    LUISA_ASSERT(ranges.size() > level);
    return ranges[level].before_layout;
}

void ResourceBarrier::process_bindless(BindlessArray const *bdls_arr, Usage dst_usage) {
    for (auto iter = _write_state_map.begin(); iter != _write_state_map.end(); ++iter) {
        if (bdls_arr->is_ptr_in_bindless(reinterpret_cast<size_t>(iter->first))) {
            auto ite = _frame_states.find(iter->first);
            assert(ite);
            auto res = ite.key();
            if (res->tag() == Resource::Tag::kBuffer) {
                record(
                    BufferView(static_cast<Buffer const *>(res), 0, static_cast<Buffer const *>(res)->byte_size()),
                    dst_usage);
            } else if (res->tag() == Resource::Tag::kTexture) {
                auto tex = static_cast<Texture const *>(res);
                for (auto i : vstd::range(tex->mip())) {
                    record(
                        TexView(tex, i),
                        dst_usage);
                }
            }
        }
    }
}

void ResourceBarrier::update_states(VkCommandBuffer cmd_buffer) {
    _buffer_barriers.clear();
    _tex_barriers.clear();
    for (auto &i : _current_update_states) {
        _update_state(i.first, *i.second);
    }
    _current_update_states.clear();
    VkDependencyInfo info{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    if (!_tex_barriers.empty()) {
        for (auto &i : _tex_barriers) {
            barrier_filter(i);
        }
        info.imageMemoryBarrierCount = _tex_barriers.size();
        info.pImageMemoryBarriers = _tex_barriers.data();
    }
    if (!_buffer_barriers.empty()) {
        for (auto &i : _buffer_barriers) {
            barrier_filter(i);
        }
        info.bufferMemoryBarrierCount = _buffer_barriers.size();
        info.pBufferMemoryBarriers = _buffer_barriers.data();
    }
    vkCmdPipelineBarrier2(cmd_buffer, &info);
}

void ResourceBarrier::restore_states(VkCommandBuffer cmd_buffer) {
    _current_update_states.clear();
    _buffer_barriers.clear();
    _tex_barriers.clear();
    _write_state_map.clear();
    for (auto &i : _frame_states) {
        Resource const *res_ptr = i.first;
        ResourceStates &state = i.second;
        if (state.layer_states.index() == 0) {
            auto &bf = state.layer_states.get<0>();
            auto &barrier = _buffer_barriers.emplace_back();
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = bf.before_access;
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.buffer = static_cast<Buffer const *>(res_ptr)->vk_buffer();
            barrier.offset = 0;
            barrier.size = std::numeric_limits<uint64_t>::max();
            auto iter = saved_restore_states.find(i.first);
            if (iter) {
                auto &v = iter.value();
                barrier.dstStageMask = v.after_stage;
                barrier.dstAccessMask = v.after_access;
            }
        } else {// Texture
            auto &vec = state.layer_states.get<1>();
            auto init_layout = VK_IMAGE_LAYOUT_GENERAL;
            auto tex = static_cast<Texture const *>(res_ptr);
            auto iter = saved_restore_states.find(i.first);
            for (auto idx : vstd::range((int64_t)vec.size())) {
                auto &i = vec[idx];
                if (!i.level_inited) continue;
                auto &barrier = _tex_barriers.emplace_back();
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.srcAccessMask = i.before_access;
                barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                barrier.oldLayout = i.before_layout;
                barrier.newLayout = init_layout;
                tex->set_layout(idx, init_layout);
                barrier.image = tex->vk_image();
                barrier.subresourceRange = VkImageSubresourceRange{
                    .aspectMask = tex->get_aspect(),
                    .baseMipLevel = (uint)idx,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1};
                if (iter) {
                    auto &v = iter.value();
                    barrier.dstStageMask = v.after_stage;
                    barrier.newLayout = v.after_layout;
                    barrier.dstAccessMask = v.after_access;
                }
            }
        }
    }
    VkDependencyInfo info{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    if (!_tex_barriers.empty()) {
        for (auto &i : _tex_barriers) {
            barrier_filter(i);
        }
        info.imageMemoryBarrierCount = _tex_barriers.size();
        info.pImageMemoryBarriers = _tex_barriers.data();
    }
    if (!_buffer_barriers.empty()) {
        for (auto &i : _buffer_barriers) {
            barrier_filter(i);
        }
        info.bufferMemoryBarrierCount = _buffer_barriers.size();
        info.pBufferMemoryBarriers = _buffer_barriers.data();
    }
    vkCmdPipelineBarrier2(cmd_buffer, &info);
    _frame_states.clear();
}
ResourceBarrier::ResourceStates::ResourceStates(Type type, size_t size) : size(size) {
    if (type == Type::kTexture) {
        layer_states.reset_as<vstd::vector<TextureRange>>(size);
    } else {
        layer_states.reset_as<BufferRange>();
    }
}
void ResourceBarrier::barrier_filter(VkBufferMemoryBarrier2 &barrier) const {
    VkImageLayout layout;
    detail::filter_access(queue_type, barrier.srcStageMask, barrier.srcAccessMask, layout);
    detail::filter_access(queue_type, barrier.dstStageMask, barrier.dstAccessMask, layout);
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcQueueFamilyIndex = queue_index;
    barrier.dstQueueFamilyIndex = queue_index;
    barrier.pNext = nullptr;
}
void ResourceBarrier::barrier_filter(VkImageMemoryBarrier2 &barrier) const {
    detail::filter_access(queue_type, barrier.srcStageMask, barrier.srcAccessMask, barrier.oldLayout);
    detail::filter_access(queue_type, barrier.dstStageMask, barrier.dstAccessMask, barrier.newLayout);
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcQueueFamilyIndex = queue_index;
    barrier.dstQueueFamilyIndex = queue_index;
    barrier.pNext = nullptr;
}

}// namespace lc::vk
