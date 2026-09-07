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

// Short-lived XIR analyses build immutable pointer-index relations. They never
// retain iterators across insertion and do not need node address stability.
// Keep these tables independent of LUISA_USE_SYSTEM_STL: a node-based
// std::unordered_map turns every temporary relation entry into a separate
// allocation in system-STL builds.
//
// Raw pointer bits are intentionally not advertised as avalanching. The dense
// table applies its wyhash finalizer exactly once. Hashes only select candidate
// buckets; exact pointer equality remains the correctness predicate.
struct DensePointerHash {
    using is_transparent = void;

    template<typename Pointer>
        requires std::is_pointer_v<Pointer>
    [[nodiscard]] size_t operator()(Pointer pointer) const noexcept {
        return static_cast<size_t>(
            reinterpret_cast<std::uintptr_t>(pointer));
    }
};

template<typename Key, typename Value>
using DensePointerMap = ankerl::unordered_dense::map<
    Key, Value, DensePointerHash, std::equal_to<>,
    luisa::allocator<std::pair<Key, Value>>,
    luisa::vector<std::pair<Key, Value>>>;

template<typename Key>
using DensePointerSet = ankerl::unordered_dense::set<
    Key, DensePointerHash, std::equal_to<>,
    luisa::allocator<Key>, luisa::vector<Key>>;

// Compile-time regression for the storage contract: iteration walks the
// contiguous value array instead of separately allocated nodes.
static_assert(std::random_access_iterator<
              DensePointerMap<const void *, size_t>::iterator>);

}// namespace luisa::compute::xir::detail
