#pragma once

#include <cstddef>
#include <cstdint>

namespace lc {

enum class StorageBufferMetadataField : uint8_t {
    DESCRIPTOR_BIAS_BYTES,
    LOGICAL_SIZE_BYTES,
    DEVICE_ADDRESS,
};

// Shared host/shader ABI for a direct or bindless storage-buffer view. The
// runtime uploads these records and native SPIR-V addresses them as
// little-endian uint32 words from the argument/metadata storage buffer.
struct StorageBufferMetadata {
    uint64_t descriptor_bias_bytes;
    uint64_t logical_size_bytes;
    uint64_t device_address;
};

[[nodiscard]] constexpr size_t storage_buffer_metadata_field_offset(
    StorageBufferMetadataField field) noexcept {
    switch (field) {
        case StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES:
            return offsetof(StorageBufferMetadata, descriptor_bias_bytes);
        case StorageBufferMetadataField::LOGICAL_SIZE_BYTES:
            return offsetof(StorageBufferMetadata, logical_size_bytes);
        case StorageBufferMetadataField::DEVICE_ADDRESS:
            return offsetof(StorageBufferMetadata, device_address);
    }
    return 0u;
}

static_assert(sizeof(StorageBufferMetadata) == 24u);
static_assert(alignof(StorageBufferMetadata) == alignof(uint64_t));
static_assert(sizeof(StorageBufferMetadata) % sizeof(uint32_t) == 0u);
static_assert(storage_buffer_metadata_field_offset(
                  StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES) == 0u);
static_assert(storage_buffer_metadata_field_offset(
                  StorageBufferMetadataField::LOGICAL_SIZE_BYTES) == 8u);
static_assert(storage_buffer_metadata_field_offset(
                  StorageBufferMetadataField::DEVICE_ADDRESS) == 16u);

}// namespace lc
