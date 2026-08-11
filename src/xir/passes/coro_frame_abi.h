#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "coro_frame_access.h"

namespace luisa::compute::xir::detail {

// Coroutine schedulers prepend dispatch identity, dispatch size, and the
// continuation token before compiler-managed user state.
inline constexpr size_t CORO_FRAME_RESERVED_FIELD_COUNT = 7u;

// ABI decomposition is deliberately bounded. The vector is intentionally
// simple; if profiling later identifies planning allocation as material, this
// is the single container to specialize without changing the analysis model.
inline constexpr size_t CORO_FRAME_ABI_FIELD_LIMIT = 32u;

struct CoroFrameAbiField {
    luisa::vector<uint32_t> access_chain;
    const Type *type{nullptr};
};

struct CoroFrameAbiPlan {
    luisa::vector<CoroFrameAbiField> fields;
    size_t payload_size{0u};
    size_t max_alignment{0u};
    bool decomposed{false};
};

// Computes a storage-preserving partition of one dataflow atom. Whole values
// remain one field unless recursively decomposing them strictly reduces the
// sum of represented type sizes. Packed children are retained as aggregates,
// so the plan removes padding with the fewest fields found by this recursion
// rather than blindly scalarizing every primitive. Local-allocation atoms and
// complete non-lvalue SSA aggregates may be decomposed; plans exceeding
// field_limit remain whole. An SSA plan always partitions the complete value,
// so split can reconstruct it without introducing an undefined component.
[[nodiscard]] CoroFrameAbiPlan plan_coro_frame_atom_abi(
    const CoroFrameAtomDomain::Atom &atom,
    size_t field_limit = CORO_FRAME_ABI_FIELD_LIMIT) noexcept;

}// namespace luisa::compute::xir::detail
