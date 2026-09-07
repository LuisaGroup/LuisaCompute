#include <luisa/core/logging.h>

#include "resource_barrier.h"
#include "resource_barrier_contract.h"
#include "bindless_array.h"
#include "command_buffer_sync.h"

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

ResourceBarrier::TextureStates &ResourceBarrier::_texture_state(
    Texture const *texture) {
    LUISA_ASSERT(texture != nullptr,
                 "Vulkan texture barrier received a null wrapper.");
    auto identity = detail::native_image_identity(texture->vk_image());
    auto &state = _texture_frame_states.emplace(identity, texture).value();
    LUISA_ASSERT(
        state.native_state == texture->native_state(),
        "Vulkan image 0x{:016x} does not have one canonical native state.",
        identity);
    return state;
}

void ResourceBarrier::record(
    ResourceView const &res,
    Usage usage) {
    _record(
        res,
        detail::kBarrierSyncMap[luisa::to_underlying(usage)],
        detail::kBarrierAccessMap[luisa::to_underlying(usage)],
        detail::kBarrierLayoutMap[luisa::to_underlying(usage)],
        detail::TextureLayoutContract::GENERIC_USAGE);
}

void ResourceBarrier::record_texture_descriptor(
    Texture const *texture, uint32_t base_level,
    bool sampled, bool storage,
    Usage sampled_usage, Usage storage_usage) {
    LUISA_ASSERT(texture != nullptr,
                 "Vulkan texture descriptor barrier received a null texture.");
    LUISA_ASSERT(base_level < texture->mip(),
                 "Vulkan texture descriptor base mip {} is outside [0, {}).",
                 base_level, texture->mip());
    LUISA_ASSERT(sampled || storage,
                 "Vulkan texture descriptor has no sampled or storage role.");
    if (sampled) {
        auto level_count = texture->mip() - base_level;
        for (auto level = base_level; level < texture->mip(); ++level) {
            record(TexView{texture, level}, sampled_usage);
        }
        auto identity = detail::native_image_identity(texture->vk_image());
        auto &state = _texture_state(texture);
        _current_texture_descriptor_views.emplace_back(
            TextureDescriptorView{
                .identity = identity,
                .state = &state,
                .base_level = base_level,
                .level_count = level_count});
    }
    if (storage) {
        record(TexView{texture, base_level}, storage_usage);
    }
}
void ResourceBarrier::set_res(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout) {
    if (res.is_type_of<BufferView>()) {
        auto &buffer_view = res.get<0>();
        // If the buffer is host-visible, should not be recorded by resource-barrier
        if (buffer_view.buffer->flush_host()) { return; }
        auto buffer = buffer_view.buffer->vk_buffer();
        auto identity = detail::native_buffer_identity(buffer);
        auto &state = _buffer_frame_states.emplace(identity, buffer).value();
        LUISA_DEBUG_ASSERT(state.buffer == buffer);
        if (state.before_state_supplied) {
            state.range.before_stage |= stage;
            state.range.before_access |= access;
        } else {
            state.range = BufferRange{
                stage,
                VK_PIPELINE_STAGE_2_NONE,
                access,
                VK_ACCESS_2_NONE};
            state.before_state_supplied = true;
        }
        return;
    }

    auto &tex_view = res.get<1>();
    auto tex = tex_view.tex;
    auto &state = _texture_state(tex);
    LUISA_ASSERT(tex_view.level < state.native_state->mip_levels,
                 "Vulkan texture mip {} is outside [0, {}).",
                 tex_view.level, state.native_state->mip_levels);
    auto &tex_range = state.ranges[tex_view.level];
    tex_range.level_inited = true;
    if (tex_range.before_state_supplied) {
        LUISA_ASSERT(
            tex_range.before_layout == layout,
            "Vulkan before-states for aliases of image 0x{:016x}, mip {} "
            "declare conflicting layouts ({} and {}).",
            detail::native_image_identity(tex->vk_image()), tex_view.level,
            static_cast<uint32_t>(tex_range.before_layout),
            static_cast<uint32_t>(layout));
        tex_range.before_stage |= stage;
        tex_range.before_access |= access;
    } else {
        tex_range.before_stage = stage;
        tex_range.before_access = access;
        tex_range.before_layout = layout;
        tex_range.before_state_supplied = true;
    }
}
void ResourceBarrier::record(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout) {
    _record(
        res, stage, access, layout,
        detail::TextureLayoutContract::EXPLICIT_NATIVE);
}

void ResourceBarrier::_record(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout,
    detail::TextureLayoutContract layout_contract) {
    if (res.is_type_of<BufferView>()) {
        auto &buffer_view = res.get<0>();
        // If the buffer is host-visible, should not be recorded by resource-barrier
        if (buffer_view.buffer->flush_host()) { return; }
        auto buffer = buffer_view.buffer->vk_buffer();
        auto identity = detail::native_buffer_identity(buffer);
        auto &state = _buffer_frame_states.emplace(identity, buffer).value();
        LUISA_DEBUG_ASSERT(state.buffer == buffer);
        if (!state.require_update) {
            _current_buffer_update_states.emplace_back(identity, &state);
            state.require_update = true;
        }
        auto result = detail::combine_texture_access_layout(
            {state.range.after_access, VK_IMAGE_LAYOUT_GENERAL},
            {access, VK_IMAGE_LAYOUT_GENERAL});
        state.range.after_access = result.access;
        state.range.after_stage |= stage;
        return;
    }

    auto &tex_view = res.get<1>();
    auto tex = tex_view.tex;
    auto identity = detail::native_image_identity(tex->vk_image());
    auto &state = _texture_state(tex);
    LUISA_ASSERT(tex_view.level < state.native_state->mip_levels,
                 "Vulkan texture mip {} is outside [0, {}).",
                 tex_view.level, state.native_state->mip_levels);
    if (!state.require_update) {
        _current_texture_update_states.emplace_back(identity, &state);
        state.require_update = true;
    }
    auto &tex_range = state.ranges[tex_view.level];
    if (!tex_range.level_inited) {
        tex_range.level_inited = true;
        tex_range.before_layout = tex->layout(tex_view.level);
    }
    tex_range.level_require_update = true;
    tex_range.after_stage |= stage;
    auto resolved_layout = detail::resolve_texture_barrier_layout(
        state.native_state->simultaneous_access,
        access, layout, layout_contract);
    if (layout_contract ==
            detail::TextureLayoutContract::EXPLICIT_NATIVE &&
        tex_range.after_access != 0u) {
        LUISA_ASSERT(
            tex_range.after_layout == resolved_layout,
            "Vulkan native commands in one reorder layer declare "
            "conflicting layouts ({} and {}) for image 0x{:016x}, mip {}.",
            static_cast<int32_t>(tex_range.after_layout),
            static_cast<int32_t>(resolved_layout),
            identity, tex_view.level);
    }
    auto result = detail::combine_texture_access_layout(
        {tex_range.after_access, tex_range.after_layout},
        {access, resolved_layout});
    tex_range.after_access = result.access;
    tex_range.after_layout = result.layout;
}

void ResourceBarrier::force_refresh_layout(
    Resource const *res, uint level,
    VkImageLayout before_layout) {
    LUISA_ASSERT(res != nullptr && res->tag() == Resource::Tag::kTexture,
                 "Vulkan image-layout refresh requires a texture.");
    auto texture = static_cast<Texture const *>(res);
    auto identity = detail::native_image_identity(texture->vk_image());
    auto iter = _texture_frame_states.find(identity);
    if (!iter) return;
    auto &ranges = iter.value().ranges;
    LUISA_ASSERT(ranges.size() > level);
    ranges[level].before_layout = before_layout;
}

void ResourceBarrier::remove_resource(Resource const *res) noexcept {
    if (res->tag() == Resource::Tag::kBuffer) {
        auto buffer = static_cast<Buffer const *>(res)->vk_buffer();
        auto identity = detail::native_buffer_identity(buffer);
        for (auto it = _current_buffer_update_states.begin();
             it != _current_buffer_update_states.end();) {
            if (it->first == identity) {
                it = _current_buffer_update_states.erase(it);
            } else {
                ++it;
            }
        }
        _buffer_frame_states.remove(identity);
        _saved_buffer_restore_states.remove(identity);
        return;
    }

    if (res->tag() == Resource::Tag::kTexture) {
        auto texture = static_cast<Texture const *>(res);
        auto identity = detail::native_image_identity(texture->vk_image());
        for (auto it = _current_texture_descriptor_views.begin();
             it != _current_texture_descriptor_views.end();) {
            if (it->identity == identity) {
                it = _current_texture_descriptor_views.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = _current_texture_update_states.begin();
             it != _current_texture_update_states.end();) {
            if (it->first == identity) {
                it = _current_texture_update_states.erase(it);
            } else {
                ++it;
            }
        }
        _texture_frame_states.remove(identity);
        _saved_texture_restore_states.remove(identity);
    }
}

ResourceBarrier::~ResourceBarrier() {
}

void ResourceBarrier::_resolve_texture_descriptor_layouts() {
    // Overlapping sampled views form layout-equivalence groups. A later view
    // may promote a shared mip to GENERAL, so iterate until that promotion has
    // propagated through every transitive overlap. Layouts only move toward
    // GENERAL, which bounds convergence by the number of tracked mip ranges.
    auto changed = false;
    do {
        changed = false;
        for (auto &view : _current_texture_descriptor_views) {
            LUISA_ASSERT(view.state != nullptr && view.level_count != 0u &&
                             view.base_level < view.state->ranges.size() &&
                             view.level_count <=
                                 view.state->ranges.size() - view.base_level,
                         "Vulkan sampled-image descriptor view for image "
                         "0x{:016x} has an invalid mip range [{}, {}).",
                         view.identity, view.base_level,
                         view.base_level + view.level_count);
            auto layout = VK_IMAGE_LAYOUT_UNDEFINED;
            for (auto level = view.base_level;
                 level < view.base_level + view.level_count; ++level) {
                auto &range = view.state->ranges[level];
                LUISA_ASSERT(
                    range.level_require_update &&
                        range.after_layout != VK_IMAGE_LAYOUT_UNDEFINED,
                    "Vulkan sampled-image descriptor view for image "
                    "0x{:016x}, mip {} has no pending layout.",
                    view.identity, level);
                layout = detail::combine_texture_descriptor_view_layout(
                    layout, range.after_layout);
            }
            LUISA_ASSERT(layout != VK_IMAGE_LAYOUT_UNDEFINED);
            for (auto level = view.base_level;
                 level < view.base_level + view.level_count; ++level) {
                auto &range = view.state->ranges[level];
                if (range.after_layout != layout) {
                    LUISA_ASSERT(
                        layout == VK_IMAGE_LAYOUT_GENERAL,
                        "Vulkan sampled-image descriptor layout resolution "
                        "attempted a non-conservative promotion.");
                    range.after_layout = layout;
                    changed = true;
                }
            }
        }
    } while (changed);
}

void ResourceBarrier::_update_state(BufferStates &state) {
    state.require_update = false;
    auto &range = state.range;
    auto &barrier = _buffer_barriers.emplace_back();
    barrier.srcStageMask = range.before_stage;
    barrier.dstStageMask = range.after_stage;
    barrier.srcAccessMask = range.before_access;
    barrier.dstAccessMask = range.after_access;
    barrier.buffer = state.buffer;
    barrier.offset = 0;
    barrier.size = std::numeric_limits<uint64_t>::max();

    range.before_stage = range.after_stage;
    range.after_stage = VK_PIPELINE_STAGE_2_NONE;
    range.before_access = range.after_access;
    range.after_access = VK_ACCESS_2_NONE;
}

void ResourceBarrier::_update_state(TextureStates &state) {
    state.require_update = false;
    for (auto idx : vstd::range((int64_t)state.ranges.size())) {
        auto &range = state.ranges[idx];
        if (!range.level_require_update) continue;
        range.level_require_update = false;
        auto &barrier = _tex_barriers.emplace_back();
        range.after_layout = detail::filter_layout(range.after_layout, range.after_access);
        barrier.srcStageMask = range.before_stage;
        barrier.dstStageMask = range.after_stage;
        barrier.srcAccessMask = range.before_access;
        barrier.dstAccessMask = range.after_access;
        barrier.oldLayout = range.before_layout;
        barrier.newLayout = range.after_layout;
        barrier.image = state.native_state->image;
        barrier.subresourceRange = VkImageSubresourceRange{
            .aspectMask = Texture::get_aspect_from_format(
                state.native_state->format),
            .baseMipLevel = (uint)idx,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};
        barrier_filter(barrier);
        range.before_stage = barrier.dstStageMask;
        range.after_stage = VK_PIPELINE_STAGE_2_NONE;
        range.before_access = barrier.dstAccessMask;
        range.after_access = VK_ACCESS_2_NONE;
        range.before_layout = barrier.newLayout;
    }
}
VkImageLayout ResourceBarrier::get_layout(Resource const *res, uint level) {
    LUISA_ASSERT(res != nullptr && res->tag() == Resource::Tag::kTexture,
                 "Vulkan image-layout query requires a texture.");
    auto texture = static_cast<Texture const *>(res);
    auto identity = detail::native_image_identity(texture->vk_image());
    auto iter = _texture_frame_states.find(identity);
    LUISA_ASSERT(iter);
    auto &ranges = iter.value().ranges;
    LUISA_ASSERT(ranges.size() > level);
    return ranges[level].before_layout;
}

VkImageLayout ResourceBarrier::get_texture_descriptor_layout(
    Texture const *texture, uint32_t base_level,
    uint32_t level_count) {
    LUISA_ASSERT(texture != nullptr && level_count != 0u,
                 "Vulkan image descriptor layout query requires a nonempty "
                 "texture view.");
    auto identity = detail::native_image_identity(texture->vk_image());
    auto iter = _texture_frame_states.find(identity);
    LUISA_ASSERT(iter);
    auto &ranges = iter.value().ranges;
    LUISA_ASSERT(base_level < ranges.size() &&
                     level_count <= ranges.size() - base_level,
                 "Vulkan image descriptor mip range [{}, {}) is outside "
                 "[0, {}).",
                 base_level, base_level + level_count, ranges.size());
    auto layout = ranges[base_level].before_layout;
    for (auto level = base_level + 1u;
         level < base_level + level_count; ++level) {
        LUISA_ASSERT(
            ranges[level].before_layout == layout,
            "Vulkan sampled-image descriptor for image 0x{:016x} spans "
            "mips with different layouts (mip {} is {}, mip {} is {}).",
            identity, base_level, static_cast<int32_t>(layout), level,
            static_cast<int32_t>(ranges[level].before_layout));
    }
    return layout;
}

void ResourceBarrier::_apply_state(
    ResourceView const &view,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout,
    detail::TextureLayoutContract layout_contract,
    BindlessStateOperation operation) {
    switch (operation) {
        case BindlessStateOperation::RECORD:
            _record(view, stage, access, layout, layout_contract);
            break;
        case BindlessStateOperation::SET_BEFORE:
            set_res(view, stage, access, layout);
            break;
        case BindlessStateOperation::SET_RESTORE:
            set_restore_state(view, stage, access, layout);
            break;
    }
}

void ResourceBarrier::_process_bindless_members(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 buffer_stage,
    VkAccessFlagBits2 buffer_access,
    VkPipelineStageFlagBits2 texture_stage,
    VkAccessFlagBits2 texture_access,
    VkImageLayout texture_layout,
    detail::TextureLayoutContract layout_contract,
    BindlessStateOperation operation) {
    LUISA_ASSERT(bdls_arr != nullptr,
                 "Vulkan bindless barrier traversal received a null array.");
    std::lock_guard lock{bdls_arr->mtx};
    // A bindless access can name any currently encoded resource. Record every
    // unique one. The encoded map, rather than the pending host update map, is
    // the descriptor state observed by this dispatch.
    bdls_arr->traverse_encoded_resources([&](uint64_t handle) noexcept {
        auto res = reinterpret_cast<Resource const *>(handle);
        LUISA_ASSERT(
            res != nullptr &&
                (res->tag() == Resource::Tag::kBuffer ||
                 res->tag() == Resource::Tag::kTexture),
            "Vulkan bindless barrier snapshot contains an invalid resource.");
        if (res->tag() == Resource::Tag::kBuffer) {
            auto buffer = static_cast<Buffer const *>(res);
            _apply_state(
                BufferView(buffer, 0u, buffer->byte_size()),
                buffer_stage, buffer_access,
                VK_IMAGE_LAYOUT_UNDEFINED, layout_contract, operation);
        } else if (res->tag() == Resource::Tag::kTexture) {
            LUISA_ASSERT(
                texture_layout != VK_IMAGE_LAYOUT_UNDEFINED,
                "A bindless barrier that reaches textures requires an "
                "explicit destination layout.");
            auto tex = static_cast<Texture const *>(res);
            for (auto level : vstd::range(tex->mip())) {
                _apply_state(
                    TexView(tex, level),
                    texture_stage, texture_access, texture_layout,
                    layout_contract, operation);
            }
        }
    });
}

void ResourceBarrier::_apply_bindless_state(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout texture_layout,
    BindlessStateOperation operation) {
    LUISA_ASSERT(bdls_arr != nullptr,
                 "Vulkan bindless state expansion received a null array.");
    auto &indices = bdls_arr->indices_buffer();
    _apply_state(
        BufferView(&indices, 0u, indices.byte_size()),
        stage, access, VK_IMAGE_LAYOUT_UNDEFINED,
        detail::TextureLayoutContract::EXPLICIT_NATIVE, operation);
    _process_bindless_members(
        bdls_arr,
        stage, access,
        stage, access,
        texture_layout,
        detail::TextureLayoutContract::EXPLICIT_NATIVE,
        operation);
}

void ResourceBarrier::process_bindless(
    BindlessArray const *bdls_arr,
    Usage buffer_dst_usage,
    Usage texture_dst_usage) {
    // Bindless textures are sampled-only in the shader ABI. Keep their
    // barrier usage separate from writable bindless buffers even when both
    // are reached through the same array argument.
    _process_bindless_members(
        bdls_arr,
        detail::kBarrierSyncMap[luisa::to_underlying(buffer_dst_usage)],
        detail::kBarrierAccessMap[luisa::to_underlying(buffer_dst_usage)],
        detail::kBarrierSyncMap[luisa::to_underlying(texture_dst_usage)],
        detail::kBarrierAccessMap[luisa::to_underlying(texture_dst_usage)],
        VK_IMAGE_LAYOUT_GENERAL,
        // The bindless descriptor fixes the image layout to GENERAL, but this
        // remains backend-owned shader usage. Let it conservatively promote
        // an overlapping direct sampled descriptor to GENERAL instead of
        // treating the overlap as two conflicting native command contracts.
        detail::TextureLayoutContract::GENERIC_USAGE,
        BindlessStateOperation::RECORD);
}

void ResourceBarrier::process_bindless(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 dst_stage,
    VkAccessFlagBits2 dst_access,
    VkImageLayout texture_layout) {
    // A custom command supplies the native Vulkan access contract directly;
    // apply that exact contract to every member visible through the encoded
    // descriptor snapshot at this command boundary.
    _process_bindless_members(
        bdls_arr,
        dst_stage, dst_access,
        dst_stage, dst_access,
        texture_layout,
        detail::TextureLayoutContract::EXPLICIT_NATIVE,
        BindlessStateOperation::RECORD);
}

void ResourceBarrier::record_bindless(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 dst_stage,
    VkAccessFlagBits2 dst_access,
    VkImageLayout texture_layout) {
    _apply_bindless_state(
        bdls_arr, dst_stage, dst_access, texture_layout,
        BindlessStateOperation::RECORD);
}

void ResourceBarrier::set_bindless_before_state(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout texture_layout) {
    _apply_bindless_state(
        bdls_arr, stage, access, texture_layout,
        BindlessStateOperation::SET_BEFORE);
}

void ResourceBarrier::set_bindless_restore_state(
    BindlessArray const *bdls_arr,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout texture_layout) {
    _apply_bindless_state(
        bdls_arr, stage, access, texture_layout,
        BindlessStateOperation::SET_RESTORE);
}

void ResourceBarrier::clear_restore_states() noexcept {
    _saved_buffer_restore_states.clear();
    _saved_texture_restore_states.clear();
}

void ResourceBarrier::set_restore_state(
    ResourceView const &res,
    VkPipelineStageFlagBits2 stage,
    VkAccessFlagBits2 access,
    VkImageLayout layout) {
    auto state = RestoreStates{
        .after_stage = stage,
        .after_access = access,
        .after_layout = layout};
    if (res.is_type_of<BufferView>()) {
        auto buffer = res.get<0>().buffer->vk_buffer();
        auto identity = detail::native_buffer_identity(buffer);
        if (auto iter = _saved_buffer_restore_states.find(identity)) {
            iter.value().after_stage |= stage;
            iter.value().after_access |= access;
        } else {
            _saved_buffer_restore_states.emplace(identity, state);
        }
    } else {
        auto &tex_view = res.get<1>();
        auto &texture_state = _texture_state(tex_view.tex);
        LUISA_ASSERT(tex_view.level < texture_state.native_state->mip_levels,
                     "Vulkan texture restore mip {} is outside [0, {}).",
                     tex_view.level,
                     texture_state.native_state->mip_levels);
        auto identity = detail::native_image_identity(
            texture_state.native_state->image);
        auto &restore_state = _saved_texture_restore_states.emplace(
                                                               identity, texture_state.native_state->mip_levels)
                                  .value();
        LUISA_ASSERT(
            restore_state.ranges.size() ==
            texture_state.native_state->mip_levels);
        auto &range = restore_state.ranges[tex_view.level];
        if (range.valid) {
            LUISA_ASSERT(
                range.state.after_layout == layout,
                "Vulkan after-states for aliases of image 0x{:016x}, mip {} "
                "declare conflicting layouts ({} and {}).",
                identity, tex_view.level,
                static_cast<int32_t>(range.state.after_layout),
                static_cast<int32_t>(layout));
            range.state.after_stage |= stage;
            range.state.after_access |= access;
        } else {
            range = TextureRestoreRange{
                .valid = true,
                .state = state};
        }
    }
}

void ResourceBarrier::update_states(VkCommandBuffer cmd_buffer) {
    _buffer_barriers.clear();
    _tex_barriers.clear();
    _resolve_texture_descriptor_layouts();
    for (auto &i : _current_buffer_update_states) {
        _update_state(*i.second);
    }
    _current_buffer_update_states.clear();
    for (auto &i : _current_texture_update_states) {
        _update_state(*i.second);
    }
    _current_texture_update_states.clear();
    _current_texture_descriptor_views.clear();
    VkDependencyInfo info{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    if (!_tex_barriers.empty()) {
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
    detail::cmd_pipeline_barrier(cmd_buffer, device, &info);
}

void ResourceBarrier::restore_states(VkCommandBuffer cmd_buffer) {
    _current_buffer_update_states.clear();
    _current_texture_update_states.clear();
    _current_texture_descriptor_views.clear();
    _buffer_barriers.clear();
    _tex_barriers.clear();
    for (auto &entry : _buffer_frame_states) {
        auto identity = entry.first;
        auto &state = entry.second;
        auto &barrier = _buffer_barriers.emplace_back();
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = state.range.before_access;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.buffer = state.buffer;
        barrier.offset = 0;
        barrier.size = std::numeric_limits<uint64_t>::max();
        if (auto iter = _saved_buffer_restore_states.find(identity)) {
            auto &saved = iter.value();
            barrier.dstStageMask = saved.after_stage;
            barrier.dstAccessMask = saved.after_access;
        }
    }
    for (auto &entry : _texture_frame_states) {
        auto identity = entry.first;
        auto &state = entry.second;
        auto init_layout = VK_IMAGE_LAYOUT_GENERAL;
        auto restore_iter = _saved_texture_restore_states.find(identity);
        for (auto idx : vstd::range((int64_t)state.ranges.size())) {
            auto &range = state.ranges[idx];
            if (!range.level_inited) continue;
            auto &barrier = _tex_barriers.emplace_back();
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = range.before_access;
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.oldLayout = range.before_layout;
            barrier.newLayout = init_layout;
            barrier.image = state.native_state->image;
            barrier.subresourceRange = VkImageSubresourceRange{
                .aspectMask = Texture::get_aspect_from_format(
                    state.native_state->format),
                .baseMipLevel = (uint)idx,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1};
            if (restore_iter) {
                auto &saved = restore_iter.value().ranges[idx];
                if (saved.valid) {
                    barrier.dstStageMask = saved.state.after_stage;
                    barrier.newLayout = saved.state.after_layout;
                    barrier.dstAccessMask = saved.state.after_access;
                }
            }
            barrier_filter(barrier);
            state.native_state->set_layout(
                static_cast<uint>(idx), barrier.newLayout);
        }
    }
    VkDependencyInfo info{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    if (!_tex_barriers.empty()) {
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
    detail::cmd_pipeline_barrier(cmd_buffer, device, &info);
    _buffer_frame_states.clear();
    _texture_frame_states.clear();
    clear_restore_states();
}
void ResourceBarrier::barrier_filter(VkBufferMemoryBarrier2 &barrier) const {
    auto src = detail::normalize_queue_scope(
        queue_type, barrier.srcStageMask, barrier.srcAccessMask);
    auto dst = detail::normalize_queue_scope(
        queue_type, barrier.dstStageMask, barrier.dstAccessMask);
    barrier.srcStageMask = src.stages;
    barrier.srcAccessMask = src.access;
    barrier.dstStageMask = dst.stages;
    barrier.dstAccessMask = dst.access;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.pNext = nullptr;
}
void ResourceBarrier::barrier_filter(VkImageMemoryBarrier2 &barrier) const {
    auto src = detail::normalize_queue_scope(
        queue_type, barrier.srcStageMask, barrier.srcAccessMask);
    auto dst = detail::normalize_queue_scope(
        queue_type, barrier.dstStageMask, barrier.dstAccessMask);
    barrier.srcStageMask = src.stages;
    barrier.srcAccessMask = src.access;
    barrier.dstStageMask = dst.stages;
    barrier.dstAccessMask = dst.access;
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.pNext = nullptr;
}

}// namespace lc::vk
