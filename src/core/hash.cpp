//
// Created by Mike Smith on 2021/2/6.
//

#include "hash.h"

namespace luisa {

uint32_t xxh32_hash32(const void *data, size_t size, uint32_t seed) noexcept {
    return XXH_INLINE_XXH32(data, size, seed);
}

uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH_INLINE_XXH3_64bits_withSeed(data, size, seed);
}

XXH128_hash_t xxh3_hash128(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH_INLINE_XXH3_128bits_withSeed(data, size, seed);
}

}// namespace luisa
