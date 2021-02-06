//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <xxh3.h>

namespace luisa {

[[nodiscard]] uint32_t xxh32_hash32(const void *data, size_t size, uint32_t seed = 19980810u) noexcept;
[[nodiscard]] uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;
[[nodiscard]] XXH128_hash_t xxh3_hash128(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;

}

