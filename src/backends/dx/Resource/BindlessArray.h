#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <vstl/LockFreeArrayQueue.h>
#include <runtime/sampler.h>
using namespace luisa::compute;
namespace toolhub::directx {
class TextureBase;
class CommandBufferBuilder;
class BindlessArray final : public Resource {
public:
    enum class BindTag : vbyte {
        Buffer,
        Tex2D,
        Tex3D
    };

private:
    struct BindlessStruct {
        static constexpr uint n_pos = std::numeric_limits<uint>::max();
        uint buffer;
        uint tex2D;
        uint sampler2D;
        uint tex3D;
        uint sampler3D;
    };
    vstd::vector<BindlessStruct> binded;
    vstd::HashMap<uint, BindlessStruct> updateMap;
    using Map = vstd::HashMap<size_t, size_t>;
    vstd::HashMap<std::pair<uint, BindTag>, Map::Index> indexMap;
    Map ptrMap;
    DefaultBuffer buffer;
    mutable std::mutex globalMtx;
    uint GetNewIndex();
    void TryReturnIndex(uint originValue);
    vstd::LockFreeArrayQueue<uint> freeQueue;
    void AddDepend(uint idx, BindTag tag, size_t ptr);
    void RemoveDepend(uint idx, BindTag tag);

public:
    using Property = vstd::variant<
        BufferView,
        std::pair<TextureBase const *, Sampler>>;
    void Bind(Property const &prop, uint index);
    void UnBind(BindTag type, uint index);
    bool IsPtrInBindless(size_t ptr) const;
    void Update(
        CommandBufferBuilder &builder);
    Tag GetTag() const override { return Tag::BindlessArray; }
    BindlessArray(
        Device *device,
        uint arraySize);
    ~BindlessArray();
    VSTD_SELF_PTR
};
}// namespace toolhub::directx