//
// Created by Mike Smith on 2022/12/19.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <core/dll_export.h>

namespace luisa {

static constexpr uint64_t hash64_default_seed = 0x19980810ull;

[[nodiscard]] LC_CORE_API uint64_t hash64(const void *ptr, size_t size, uint64_t seed) noexcept;

template<typename T>
struct hash {};

}// namespace luisa
