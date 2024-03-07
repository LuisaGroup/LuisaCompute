#pragma once
#include <d3d12.h>
#include <stdint.h>
#include <wrl/client.h>
struct SparseHeap {
    uint64_t allocation;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    uint64_t size_bytes;
};