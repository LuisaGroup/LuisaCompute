#include <luisa/core/logging.h>
#include <luisa/core/pool.h>

namespace luisa {

void LC_CORE_API detail::memory_pool_check_memory_leak(size_t expected, size_t actual) noexcept {
    if (expected != actual) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Leaks detected in pool: "
            "expected {} objects but got {}.",
            expected, actual);
    }
}

}// namespace luisa

