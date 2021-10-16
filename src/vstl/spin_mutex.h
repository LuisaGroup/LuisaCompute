#pragma once
#include <vstl/config.h>
#include <atomic>
#include <core/spin_mutex.h>
namespace vstd {
using spin_mutex = luisa::spin_mutex;
}// namespace vstd