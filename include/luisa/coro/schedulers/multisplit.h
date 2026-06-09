#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::coro {

/// Result of stable multi-split clustering.
///
/// Maps each target token (subroutine identifier) to the list of
/// instance indices that should be dispatched for that subroutine.
/// Within each group, indices preserve the original order of alive
/// instances (stable clustering).
struct MultiSplitResult {
    luisa::unordered_map<uint32_t, luisa::vector<size_t>> groups;
};

/// Cluster alive coroutine instances by their target_token.
///
/// For each index i where alive[i] is true, the index is appended to
/// groups[target_tokens[i]]. The original order within each group is
/// preserved, making this a stable clustering.
///
/// @param target_tokens  Token values for each instance (identifies the
///                       target subroutine). Size must match alive.size().
/// @param alive          Whether each instance is alive (not completed).
/// @return               Grouped indices by token. Empty groups for tokens
///                       with no alive instances are not present in the map.
[[nodiscard]] inline MultiSplitResult stable_multisplit(
    const luisa::vector<uint32_t> &target_tokens,
    const luisa::vector<bool> &alive) noexcept {

    MultiSplitResult result;
    const size_t n = target_tokens.size();

    for (size_t i = 0u; i < n; ++i) {
        if (alive[i]) {
            result.groups[target_tokens[i]].push_back(i);
        }
    }

    return result;
}

}// namespace luisa::compute::coro
