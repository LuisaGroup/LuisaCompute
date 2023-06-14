#pragma once

#include <DXRuntime/Device.h>
#include <luisa/vstl/functional.h>

namespace lc::dx {
using namespace Microsoft::WRL;
class SparseAllocator : public vstd::IOperatorNewBase {
public:
    struct Heap {
        vstd::vector<uint> waitingPools;
        uint64 capacity;
    };

private:
    struct PtrHash {
        size_t operator()(ComPtr<ID3D12Heap> const &p) const {
            return luisa::hash64(p.GetAddressOf(), sizeof(void *), luisa::hash64_default_seed);
        }
        size_t operator()(ID3D12Heap *p) const {
            return luisa::hash64(&p, sizeof(void *), luisa::hash64_default_seed);
        }
    };
    struct PtrEqual {
        int operator()(ComPtr<ID3D12Heap> const &a, ComPtr<ID3D12Heap> const &b) const {
            return memcmp(a.GetAddressOf(), b.GetAddressOf(), sizeof(void *));
        }
        int operator()(ComPtr<ID3D12Heap> const &a, ID3D12Heap *b) const {
            return memcmp(a.GetAddressOf(), &b, sizeof(void *));
        }
    };
    Device *device;
    using PtrMap = vstd::unordered_map<ID3D12Heap *, Heap *>;
    vstd::HashMap<ComPtr<ID3D12Heap>, Heap, PtrHash, PtrEqual> fullMaps;
    PtrMap waitingMaps;
    uint64 tileCapacity;
    bool isTex;
    PtrMap::iterator AllocateHeap();

public:
    SparseAllocator(Device *device, bool isTex, uint tileCapacity = 32);
    ~SparseAllocator();
    void AllocateTiles(
        uint tileCount,
        vstd::FuncRef<void(ID3D12Heap *heap, vstd::span<uint const> offsets)> const &allocate);
    void Deallocate(ID3D12Heap *heap, vstd::span<uint const> offsets);
    void ClearAllocate();
};
}// namespace lc::dx
