#pragma once

#include <cstddef>
#include <cstdint>

namespace lc {

// Shared host/shader ABI for a direct or bindless storage-buffer view. The
// runtime uploads these records and native SPIR-V addresses them as four
// little-endian uint32 words from the argument/metadata storage buffer.
struct StorageBufferMetadata {
    uint64_t descriptor_bias_bytes;
    uint64_t logical_size_bytes;
};

static_assert(sizeof(StorageBufferMetadata) == 16u);
static_assert(alignof(StorageBufferMetadata) == alignof(uint64_t));
static_assert(sizeof(StorageBufferMetadata) % sizeof(uint32_t) == 0u);

}// namespace lc
