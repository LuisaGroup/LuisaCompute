#pragma once
#include "default_buffer.h"
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/first_fit.h>

namespace lc::vk {
using namespace luisa::compute;
class CommandBuffer;
class Texture;
class ResourceBarrier;
class BindlessArray : public Resource {
public:
    using Map = vstd::HashMap<size_t, size_t>;
    using MapIndex = typename Map::Index;
    struct BindlessStruct {
        static constexpr auto kInvalidPos = std::numeric_limits<uint>::max();
        static constexpr auto kMask = (1u << 28u) - 1;
        uint buffer = kInvalidPos;
        uint tex_2d = kInvalidPos;
        uint tex_3d = kInvalidPos;
        void write_samp2d(uint tex, uint s) {
            tex_2d = tex | (s << 28);
        }
        void write_samp3d(uint tex, uint s) {
            tex_3d = tex | (s << 28);
        }
    };
    struct MapIndices {
        MapIndex buffer;
        MapIndex tex_2d;
        MapIndex tex_3d;
    };
    struct TypedBufferBinding {
        uint descriptor_index = BindlessStruct::kInvalidPos;
        uint descriptor_bias_bytes{};
        uint logical_size_bytes{};
        uint reserved{};
    };
    static_assert(sizeof(TypedBufferBinding) == 16u);
private:
    struct FreeValue {
        uint _type : 2;
        uint _index : 30;
    };
    DefaultBuffer _indices_buffer;
    luisa::unique_ptr<DefaultBuffer> _buffer_metadata;
    BindlessSlotType _type;
    size_t _slot_count;
    vstd::vector<uint> _typed_descriptor_indices;
    vstd::vector<TypedBufferBinding> _typed_buffer_bindings;
    vstd::variant<
        vstd::vector<std::pair<BindlessStruct, MapIndices>>,
        vstd::vector<MapIndex>>
        _typed_binded;
    vstd::variant<
        vstd::vector<std::pair<BindlessStruct, MapIndices>>,
        vstd::vector<MapIndex>>
        _encoded_binded;
    Map _ptr_map;
    Map _encoded_ptr_map;
    static void _return_value(
        Map &resource_map, vstd::vector<FreeValue> &retired_slots,
        MapIndex &index, uint type, uint &origin_value);
    static void _deref(Map &resource_map, Map::Index &index);
    static Map::Index _add_index(Map &resource_map, size_t ptr);
    void _emplace_tex(
        VkImageView &img_view,
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        VkDescriptorSet tex_set,
        uint tex_idx,
        Texture const *tex) const;
    void _defer_descriptor_recycling(
        CommandBuffer *cmdbuffer,
        vstd::vector<FreeValue> &&retired_slots) const;

public:
    auto &indices_buffer() { return _indices_buffer; }
    auto const &indices_buffer() const { return _indices_buffer; }
    [[nodiscard]] auto buffer_metadata() const noexcept {
        return _buffer_metadata.get();
    }
    [[nodiscard]] auto size() const noexcept { return _slot_count; }
    mutable std::mutex mtx;
    BindlessArray(Device *device, BindlessSlotType type, size_t size);
    void pre_update(ResourceBarrier *barrier);
    bool is_ptr_in_bindless(size_t ptr) const {
        return _ptr_map.find(ptr);
    }
    bool is_ptr_in_encoded_bindless(size_t ptr) const {
        return _encoded_ptr_map.find(ptr);
    }
    // Caller must hold mtx. The pending map is updated in command-recording
    // order and therefore forms the exact resource snapshot seen by the
    // command reorder planner at a dispatch boundary.
    template<typename F>
    void traverse_pending_resources(F &&visitor) const noexcept {
        for (auto iter = _ptr_map.begin(); iter != _ptr_map.end(); ++iter) {
            visitor(static_cast<uint64_t>(iter->first));
        }
    }
    // Caller must hold mtx. This is the descriptor membership currently
    // visible to GPU commands, which can differ from the pending host map
    // until an update command is encoded.
    template<typename F>
    void traverse_encoded_resources(F &&visitor) const noexcept {
        for (auto iter = _encoded_ptr_map.begin();
             iter != _encoded_ptr_map.end(); ++iter) {
            visitor(static_cast<uint64_t>(iter->first));
        }
    }
    [[nodiscard]] bool contains_buffer_alias(
        const Buffer *source) const noexcept;
    // Checks the exact descriptor snapshot visible to the next encoded
    // dispatch. External buffer imports are rejected because their creation
    // flags are not attested by the current native-resource API.
    [[nodiscard]] bool encoded_buffers_support_device_address() const noexcept;
    void bind(luisa::span<BindlessArrayUpdateCommand::Modification const> mods);
    void bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods);
    void bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods);
    void bind(vstd::span<const BindlessArrayUpdateCommand::Texture3DModification> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::BufferModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Texture2DModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Texture3DModification const> mods);
    void update(
        CommandBuffer *cmdbuffer,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<BindlessArrayUpdateCommand::Modification const> mods);
    ~BindlessArray();
};
}// namespace lc::vk
