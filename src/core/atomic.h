//
// Created by Mike Smith on 2021/3/8.
//

#pragma once

#include <atomic>

namespace luisa {

using atomic_int = std::atomic<int>;
using atomic_uint = std::atomic<unsigned int>;

static_assert(sizeof(atomic_int) == 4u);
static_assert(sizeof(atomic_uint) == 4u);

}// namespace luisa
