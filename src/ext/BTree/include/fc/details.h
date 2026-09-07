#ifndef FC_DETAILS_H
#define FC_DETAILS_H

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace frozenca {

template <typename T>
concept Containable = std::is_same_v<std::remove_cvref_t<T>, T>;

template <typename T>
concept DiskAllocable =
    std::is_same_v<std::remove_cvref_t<T>, T> &&
    std::is_trivially_copyable_v<T> && (sizeof(T) % alignof(T) == 0);

using attr_t = std::int32_t;

// Reports a fatal error and terminates the process.
// BTree never throws exceptions so that it can be compiled with
// exceptions disabled (-fno-exceptions).
[[noreturn]] inline void btree_fatal(const char *message) noexcept {
    std::fprintf(stderr, "frozenca::btree: %s\n", message);
    std::abort();
}

}  // namespace frozenca
#endif  // FC_DETAILS_H
