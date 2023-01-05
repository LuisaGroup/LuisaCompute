#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <vstl/lockfree_array_queue.h>
#include <runtime/command.h>
using namespace luisa::compute;
namespace toolhub::directx {
class TextureBase;
class CommandBufferBuilder;
class ResourceStateTracker;
class BindlessArray final : public Resource {
public:
    enum class BindTag : uint8_t {
        Buffer,
        Tex2D,
        Tex3D
    };
    using Map = vstd::HashMap<size_t, size_t>;
    using MapIndex = typename Map::Index;
    struct BindlessStruct {
        static constexpr auto n_pos = std::numeric_limits<uint32_t>::max();
        uint32_t buffer = n_pos;
        uint32_t tex2D = n_pos;
        uint32_t tex3D = n_pos;
        uint16_t tex2DX;
        uint16_t tex2DY;
        uint16_t tex3DX;
        uint16_t tex3DY;
        uint16_t tex3DZ;
        uint8_t samp2D;
        uint8_t samp3D;
    };
    struct MapIndicies {
        MapIndex buffer;
        MapIndex tex2D;
        MapIndex tex3D;
    };

private:
    vstd::vector<std::pair<BindlessStruct, MapIndicies>> binded;
    Map ptrMap;
    DefaultBuffer buffer;
    void TryReturnIndex(MapIndex &index, uint32_t &originValue);
    MapIndex AddIndex(size_t ptr);
    mutable vstd::LockFreeArrayQueue<uint32_t> freeQueue;

public:
    bool IsPtrInBindless(size_t ptr) const {
        return ptrMap.Find(ptr);
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
    VSTD_SELF_PTR
};
}// namespace toolhub::directx