#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <vstl/LockFreeArrayQueue.h>
#include <runtime/sampler.h>
using namespace luisa::compute;
namespace toolhub::directx {
class TextureBase;
class CommandBufferBuilder;
class ResourceStateTracker;
class BindlessArray final : public Resource {
public:
    enum class BindTag : vbyte {
        Buffer,
        Tex2D,
        Tex3D
    };
    using Map = vstd::HashMap<size_t, size_t>;
    using MapIndex = typename Map::Index;
    struct BindlessStruct {
        static constexpr uint n_pos = std::numeric_limits<uint>::max();
        uint buffer = n_pos;
        uint tex2D = n_pos;
        uint tex3D = n_pos;
        uint16_t tex2DX;
        uint16_t tex2DY;
        uint16_t tex3DX;
        uint16_t tex3DY;
        uint16_t tex3DZ;
        vbyte samp2D;
        vbyte samp3D;
    };
    struct MapIndicies {
        MapIndex buffer;
        MapIndex tex2D;
        MapIndex tex3D;
    };

private:
    vstd::vector<std::pair<BindlessStruct, MapIndicies>> binded;
    mutable vstd::HashMap<uint, BindlessStruct> updateMap;
    Map ptrMap;
    DefaultBuffer buffer;
    uint GetNewIndex();
    void TryReturnIndex(MapIndex& index, uint& originValue);
    MapIndex AddIndex(size_t ptr);
    mutable vstd::LockFreeArrayQueue<uint> freeQueue;

public:
    using Property = vstd::variant<
        BufferView,
        std::pair<TextureBase const *, Sampler>>;
    void Bind(Property const &prop, uint index);
    void UnBind(BindTag type, uint index);
    bool IsPtrInBindless(size_t ptr) const;
    DefaultBuffer const *Buffer() const { return &buffer; }
    void PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker) const;
    void UpdateStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker) const;
    Tag GetTag() const override { return Tag::BindlessArray; }
    BindlessArray(
        Device *device,
        uint arraySize);
    ~BindlessArray();
    VSTD_SELF_PTR
};
}// namespace toolhub::directx