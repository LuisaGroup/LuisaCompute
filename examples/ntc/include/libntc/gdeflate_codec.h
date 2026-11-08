/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * GDeflate codec interface: CPU / GPU decompression & compression.
 */

#pragma once

#include <libntc/common.h>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace luisa::ntc {

// Compute standard CRC32 (matching ISO 3309 / GDeflate)
uint32_t compute_crc32(const void *data, size_t size) noexcept;

// Decompress GDeflate tile stream into destination buffer
bool decompress_gdeflate(const void *src, size_t src_size, void *dst, size_t dst_size, uint32_t expected_crc = 0);

// Compress uncompressed buffer into GDeflate tile stream
bool compress_gdeflate(const void *src, size_t src_size, std::vector<uint8_t> &dst, int level = 9, uint32_t *out_crc = nullptr);

} // namespace luisa::ntc
