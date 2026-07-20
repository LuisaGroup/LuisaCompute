#pragma once
#include <volk.h>
#include "buffer.h"
#include "resource_barrier_contract.h"
#include "texture.h"
#include <luisa/vstl/common.h>
namespace lc::vk {
class BindlessArray;
class Buffer;
class ResourceBarrier {
public:
    struct Range {
        int64_t min;
        int64_t max;
        Range() {
            min = std::numeric_limits<int64_t>::min();
            max = std::numeric_limits<int64_t>::max();
        }
        explicit Range(int64_t value) {
            min = value;
            max = value + 1;
        }
        Range(int64_t min, int64_t size)
            : min(min), max(size + min) {}
        bool collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    struct BufferRange {
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
    };
    struct TextureRange {
        bool level_inited{false};
        bool before_state_supplied{false};
        bool level_require_update{false};
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
        VkImageLayout before_layout{VK_IMAGE_LAYOUT_GENERAL};
        VkImageLayout after_layout{VK_IMAGE_LAYOUT_GENERAL};
    };
    struct BufferStates {
        VkBuffer buffer;
        BufferRange range;
        bool before_state_supplied{false};
        bool require_update{false};
        explicit BufferStates(VkBuffer buffer) noexcept
            : buffer{buffer} {}
    };
    struct TextureStates {
        luisa::shared_ptr<NativeImageState> native_state;
        vstd::vector<TextureRange> ranges;
        bool require_update{false};
        explicit TextureStates(Texture const *texture)
            : native_state{texture->native_state()},
              ranges(native_state->mip_levels) {}
    };
    struct TextureDescriptorView {
        detail::NativeImageIdentity identity;
        TextureStates *state;
        uint32_t base_level;
        uint32_t level_count;
    };
    using ResourceView = vstd::variant<
        BufferView,
        TexView>;
private:
    enum class BindlessStateOperation : uint8_t {
        RECORD,
        SET_BEFORE,
        SET_RESTORE,
    };
    vstd::HashMap<detail::NativeBufferIdentity, BufferStates> _buffer_frame_states;
    vstd::vector<std::pair<detail::NativeBufferIdentity, BufferStates *>> _current_buffer_update_states;
    vstd::HashMap<detail::NativeImageIdentity, TextureStates> _texture_frame_states;
    vstd::vector<std::pair<detail::NativeImageIdentity, TextureStates *>> _current_texture_update_states;
    vstd::vector<TextureDescriptorView> _current_texture_descriptor_views;
    vstd::vector<VkImageMemoryBarrier2> _tex_barriers;
    vstd::vector<VkBufferMemoryBarrier2> _buffer_barriers;
    TextureStates &_texture_state(Texture const *texture);
    void _update_state(BufferStates &states);
    void _update_state(TextureStates &states);
    void _resolve_texture_descriptor_layouts();
    void _apply_state(
        ResourceView const &view,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout,
        BindlessStateOperation operation);
    void _record(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout,
        detail::TextureLayoutContract layout_contract);
    void _process_bindless_members(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 buffer_stage,
        VkAccessFlagBits2 buffer_access,
        VkPipelineStageFlagBits2 texture_stage,
        VkAccessFlagBits2 texture_access,
        VkImageLayout texture_layout,
        BindlessStateOperation operation);
    void _apply_bindless_state(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout texture_layout,
        BindlessStateOperation operation);
public:
    enum class Usage : uint {
        kComputeRead,
        kComputeAccelRead,
        kComputeUAV,
        kCopySource,
        kCopyDest,
        kBuildAccel,
        kCopyAccelSrc,
        kCopyAccelDst,
        kDepthRead,
        kDepthWrite,
        kDepthClear,
        kRenderTargetClear,
        kIndirectArgs,
        kVertexRead,
        kIndexRead,
        kRenderTarget,
        kAccelInstanceBuffer,
        kRasterRead,
        kRasterAccelRead,
        kRasterUAV,
    };
    using QueueType = detail::QueueType;
    QueueType queue_type{QueueType::GRAPHICS};
    struct RestoreStates {
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 after_access{0};
        VkImageLayout after_layout{VK_IMAGE_LAYOUT_GENERAL};
    };
    struct TextureRestoreRange {
        bool valid{false};
        RestoreStates state;
    };
    struct TextureRestoreStates {
        vstd::vector<TextureRestoreRange> ranges;
        explicit TextureRestoreStates(size_t mip_levels)
            : ranges(mip_levels) {}
    };
private:
    vstd::HashMap<detail::NativeBufferIdentity, RestoreStates> _saved_buffer_restore_states;
    vstd::HashMap<detail::NativeImageIdentity, TextureRestoreStates> _saved_texture_restore_states;
public:
    ResourceBarrier();
    ~ResourceBarrier();
    void record(
        ResourceView const &res,
        Usage usage);
    // A sampled descriptor spans every mip at and after base_level, whereas a
    // storage descriptor names only base_level. Record both access roles and
    // retain the sampled view's single-layout constraint until update_states.
    void record_texture_descriptor(
        Texture const *texture, uint32_t base_level,
        bool sampled, bool storage,
        Usage sampled_usage, Usage storage_usage);
    void set_res(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout);
    void clear_restore_states() noexcept;
    void set_restore_state(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout);
    void record(
        ResourceView const &res,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout);
    // only used after render pass
    void force_refresh_layout(
        Resource const *res, uint level,
        VkImageLayout before_layout);
    void remove_resource(Resource const *res) noexcept;
    void update_states(
        VkCommandBuffer cmd_buffer);
    void restore_states(VkCommandBuffer cmd_buffer);
    void barrier_filter(VkBufferMemoryBarrier2 &barrier) const;
    void barrier_filter(VkImageMemoryBarrier2 &barrier) const;
    VkImageLayout get_layout(Resource const *res, uint level);
    VkImageLayout get_texture_descriptor_layout(
        Texture const *texture, uint32_t base_level,
        uint32_t level_count);
    // Shader preprocessing records the descriptor-index/metadata buffers
    // separately because their native scopes differ from member resources.
    // These overloads therefore expand only the encoded members.
    void process_bindless(
        BindlessArray const *bdls_arr,
        Usage buffer_dst_usage,
        Usage texture_dst_usage);
    void process_bindless(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 dst_stage,
        VkAccessFlagBits2 dst_access,
        VkImageLayout texture_layout);
    // Aggregate native contracts include the descriptor-index buffer and all
    // resources in the exact encoded snapshot; texture state is per mip.
    void record_bindless(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 dst_stage,
        VkAccessFlagBits2 dst_access,
        VkImageLayout texture_layout);
    void set_bindless_before_state(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout texture_layout);
    void set_bindless_restore_state(
        BindlessArray const *bdls_arr,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout texture_layout);
};
}// namespace lc::vk
