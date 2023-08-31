#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/runtime/rhi/command.h>
namespace lc::dx {
using namespace luisa::compute;
class TextureBase;
class CommandBufferBuilder;
class ResourceStateTracker;
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
    vstd::vector<std::pair<BindlessStruct, MapIndicies>> binded;
    Map ptrMap;
    mutable std::mutex mtx;
    DefaultBuffer buffer;
    void TryReturnIndex(MapIndex &index, uint &originValue);
    void TryReturnIndexTex(MapIndex &index, uint &originValue);
    MapIndex AddIndex(size_t ptr);
    mutable vstd::vector<uint> freeQueue;

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
    void PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const;
    void UpdateStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const;

    DefaultBuffer const *BindlessBuffer() const { return &buffer; }
    Tag GetTag() const override { return Tag::BindlessArray; }
    BindlessArray(
        Device *device,
        uint arraySize);
    ~BindlessArray();
    ID3D12Resource *GetResource() const override {
        return buffer.GetResource();
    }
};
}// namespace lc::dx
