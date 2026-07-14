#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <DXRuntime/EnhancedBarrierTracker.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/core/first_fit.h>
namespace lc::dx {
using namespace luisa::compute;
class TextureBase;
class CommandBufferBuilder;
class EnhancedBarrierTracker;
class BindlessArray final : public Resource {
public:
    using Map = vstd::HashMap<size_t, size_t>;
    using MapIndex = typename Map::Index;
    struct BindlessStruct {
        static constexpr auto n_pos = std::numeric_limits<uint>::max();
        static constexpr auto mask = (1u << 28u) - 1;
        uint buffer = n_pos;
        uint tex2D = n_pos;
        uint tex3D = n_pos;
        void write_samp2d(uint tex, uint s) {
            tex2D = tex | (s << 28);
        }
        void write_samp3d(uint tex, uint s) {
            tex3D = tex | (s << 28);
        }
    };

    struct MapIndicies {
        MapIndex buffer;
        MapIndex tex2D;
        MapIndex tex3D;
    };

private:
    vstd::variant<
        vstd::vector<std::pair<BindlessStruct, MapIndicies>>,
        vstd::vector<MapIndex>>
        typed_binded;
    Map ptrMap;
    mutable std::mutex mtx;
    DefaultBuffer buffer;
    luisa::FirstFit::Node *_buffer_node = nullptr;
    void Deref(MapIndex &index);
    void TryReturnIndex(MapIndex &index, uint &originValue);
    void TryReturnIndexTex(MapIndex &index, uint &originValue);
    MapIndex AddIndex(size_t ptr);
    mutable vstd::vector<uint> freeQueue;
    mutable bool offset_setted = false;

public:
    void Lock() const {
        mtx.lock();
    }
    void Unlock() const {
        mtx.unlock();
    }
    bool IsPtrInBindless(size_t ptr) const {
        return ptrMap.find(ptr);
    }
    using Property = vstd::variant<
        BufferView,
        std::pair<TextureBase const *, Sampler>>;
    void Bind(vstd::span<const BindlessArrayUpdateCommand::Modification> mods);
    void Bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods);
    void Bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods);
    void PreProcessStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker) const;
    void UpdateStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const;
    template<typename T>
    void _UpdateStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        vstd::span<const T> mods) const;
    void UpdateStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) const;
    void UpdateStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) const;

    DefaultBuffer const *BindlessBuffer() const { return &buffer; }
    [[nodiscard]] uint size() const {
        if (auto *v = typed_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndicies>>>()) {
            return static_cast<uint>(v->size());
        } else if (auto *v = typed_binded.try_get<vstd::vector<MapIndex>>()) {
            return static_cast<uint>(v->size());
        }
        return 0;
    }
    Tag get_tag() const override { return Tag::BindlessArray; }
    BindlessArray(
        Device *device,
        uint arraySize,
        BindlessSlotType type);
    ~BindlessArray();
    ID3D12Resource *GetResource() const override {
        return buffer.GetResource();
    }
};
}// namespace lc::dx
