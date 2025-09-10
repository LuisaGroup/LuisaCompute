#pragma once
#include <d3d12.h>
#include <stdint.h>
struct SparseHeap {
    uint64_t allocation;
    ID3D12Heap *heap;
    uint64_t offset;
    uint64_t size_bytes;
};