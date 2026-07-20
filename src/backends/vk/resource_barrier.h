#pragma once
#include <volk.h>
#include "buffer.h"
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
        // Range range;
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
        bool first_time{true};// used for backup
    };
    struct TextureRange {
        bool level_inited{false};
        bool level_require_update{false};
        bool first_time{true};// used for backup
        VkPipelineStageFlagBits2 before_stage{0};
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 before_access{0};
        VkAccessFlagBits2 after_access{0};
        VkImageLayout before_layout{VK_IMAGE_LAYOUT_GENERAL};
        VkImageLayout after_layout{VK_IMAGE_LAYOUT_GENERAL};
    };
    struct ResourceStates {
        vstd::variant<
            BufferRange,
            vstd::vector<TextureRange>>
            layer_states;

        enum class Type : uint8_t {
            kBuffer,
            kTexture
        };
        size_t size;
        bool require_update{false};
        ResourceStates(Type type, size_t size);
    };
    struct BufferAfterRange {
        // Range range;
        VkPipelineStageFlagBits2 stage;
        VkAccessFlagBits2 access;
    };
    using ResourceView = vstd::variant<
        BufferView,
        TexView>;
private:
    vstd::HashMap<Resource const *, ResourceStates> _frame_states;
    vstd::vector<std::pair<Resource const *, ResourceStates *>> _current_update_states;
    vstd::HashMap<Resource const *, size_t /* size */> _write_state_map;
    vstd::vector<VkImageMemoryBarrier2> _tex_barriers;
    vstd::vector<VkBufferMemoryBarrier2> _buffer_barriers;
    void _update_state(Resource const *res_ptr, ResourceStates &states);
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
    enum class QueueType {
        kGraphics,
        kCompute,
        kCopy
    };
    QueueType queue_type{QueueType::kGraphics};
    uint queue_index{0};
    struct RestoreStates {
        ResourceView res;
        VkPipelineStageFlagBits2 after_stage{0};
        VkAccessFlagBits2 after_access{0};
        VkImageLayout after_layout{VK_IMAGE_LAYOUT_GENERAL};
    };
    vstd::HashMap<Resource const *, RestoreStates> saved_restore_states;
    ResourceBarrier();
    ~ResourceBarrier();
    void record(
        ResourceView const &res,
        Usage usage);
    void set_res(
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
    VkImageLayout get_layout(Resource const *res, uint level) const;
    void process_bindless(BindlessArray const *bdls_arr, Usage dst_usage);
};
}// namespace lc::vk
