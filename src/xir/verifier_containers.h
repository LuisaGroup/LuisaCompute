#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_dense.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir::detail {

// XIR verification builds several immutable pointer-index relations whose
// lifetime is one verification boundary. They never retain iterators across
// insertion and do not need node address stability. Keep them independent of
// LUISA_USE_SYSTEM_STL: a node-based std::unordered_map turns every temporary
// relation entry into a separate allocation in Psycles' system-STL build.
//
// The raw pointer bits are intentionally not advertised as avalanching. The
// dense table applies its wyhash finalizer exactly once, preserving a uniform
// distribution after removing the allocator-heavy intermediate hash table.
struct VerifierPointerHash {
    using is_transparent = void;

    template<typename Pointer>
        requires std::is_pointer_v<Pointer>
    [[nodiscard]] size_t operator()(Pointer pointer) const noexcept {
        return static_cast<size_t>(
            reinterpret_cast<std::uintptr_t>(pointer));
    }
};

template<typename Key, typename Value>
using VerifierPointerMap = ankerl::unordered_dense::map<
    Key, Value, VerifierPointerHash, std::equal_to<>,
    luisa::allocator<std::pair<Key, Value>>,
    luisa::vector<std::pair<Key, Value>>>;

template<typename Key>
using VerifierPointerSet = ankerl::unordered_dense::set<
    Key, VerifierPointerHash, std::equal_to<>,
    luisa::allocator<Key>, luisa::vector<Key>>;

// Compile-time regression for the storage contract behind this optimization:
// iteration walks the contiguous value array instead of a chain of separately
// allocated nodes. Correctness tests separately exercise exact collision and
// wrong-owner rejection semantics through the public verifier API.
static_assert(std::random_access_iterator<
              VerifierPointerMap<const void *, size_t>::iterator>);

}// namespace luisa::compute::xir::detail
